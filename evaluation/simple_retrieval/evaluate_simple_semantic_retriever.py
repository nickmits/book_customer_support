"""
RAGAS Evaluation for Simple Semantic PDF Retriever
Evaluates the SimpleSemanticPDFRetriever with SemanticChunker
"""
from dotenv import load_dotenv
import os
import asyncio
import sys
from typing import List, Dict, Any, Optional
from ragas import EvaluationDataset, evaluate, RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    LLMContextRecall,
    Faithfulness,
    FactualCorrectness,
    AnswerRelevancy,
    ContextPrecision
)
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers import (
    SingleHopSpecificQuerySynthesizer,
    MultiHopAbstractQuerySynthesizer,
    MultiHopSpecificQuerySynthesizer
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
import pandas as pd
import nest_asyncio

# Apply nest_asyncio FIRST
nest_asyncio.apply()

# Fix for Windows event loop issues
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

load_dotenv()

# Import the SimpleSemanticPDFRetriever
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from book_research.simple_semantic_pdf_retriever import SimpleSemanticPDFRetriever

# SEMANTIC CHUNKING PARAMETERS
BREAKPOINT_THRESHOLD_TYPE = "percentile"
BREAKPOINT_THRESHOLD_AMOUNT = 95

def to_text(content):
    """Helper to convert content to text."""
    if isinstance(content, str):
        return content
    return str(content)

def reset_eval_fields(dataset):
    """Reset evaluation fields in dataset."""
    for row in dataset:
        if hasattr(row, 'eval_sample'):
            row.eval_sample.response = None
            row.eval_sample.retrieved_contexts = []
        else:
            row.response = None
            row.retrieved_contexts = []

def evaluate_current_dataset(ds, evaluator_llm, evaluator_embeddings):
    """
    RAGAS evaluation with all required certification metrics.
    Includes: Faithfulness, Context Recall, Context Precision, Answer Relevancy
    """
    evaluation_dataset = EvaluationDataset.from_pandas(ds.to_pandas())

    # All 4 required RAGAS metrics for certification
    metrics = [
        LLMContextRecall(),
        Faithfulness(),
        ContextPrecision(),
        AnswerRelevancy(),
        FactualCorrectness()  # Bonus metric
    ]

    return evaluate(
        dataset=evaluation_dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=RunConfig(timeout=360),
        raise_exceptions=False,
    )

class SimpleRetrieverChain:
    """
    Direct retriever wrapper for RAGAS evaluation.
    Uses SimpleSemanticPDFRetriever from simple_semantic_pdf_retriever.py
    """

    def __init__(self, pdf_retriever: SimpleSemanticPDFRetriever):
        self.pdf_retriever = pdf_retriever
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Direct retrieval and QA using SimpleSemanticPDFRetriever."""
        question = inputs.get("question", "")

        try:
            # Step 1: Direct retrieval from PDF using SimpleSemanticPDFRetriever
            print(f"    > Retrieving for: {question[:50]}...")
            raw_contexts = self.pdf_retriever.search(question, k=10)

            # Log retrieval details
            print(f"    > Retrieved {len(raw_contexts)} documents")
            if raw_contexts:
                print(f"    > First context preview: {raw_contexts[0].page_content[:100]}...")

            # Step 2: Generate answer based on retrieved contexts
            context_text = "\n\n".join([
                f"[Page {doc.metadata.get('page_number', 'Unknown')}]: {doc.page_content}"
                for doc in raw_contexts[:5]  # Use top 5 for answer generation
            ])

            # Create a focused prompt
            prompt = f"""Based on the following context from the book "Thief of Sorrows", answer the question.

Context:
{context_text}

Question: {question}

Answer (be specific and use information from the context):"""

            response = self.llm.invoke(prompt)

            return {
                "response": response.content,
                "context": raw_contexts[:10]  # Return raw documents for RAGAS
            }

        except Exception as e:
            print(f"    > Error in retrieval: {e}")
            return {
                "response": f"Error: {str(e)}",
                "context": []
            }

def load_and_chunk_pdf(
    pdf_path: str,
    breakpoint_threshold_type: str = BREAKPOINT_THRESHOLD_TYPE,
    breakpoint_threshold_amount: int = BREAKPOINT_THRESHOLD_AMOUNT
):
    """
    Load PDF and create chunks using SemanticChunker.
    Matches the chunking method in simple_semantic_pdf_retriever.py
    """
    print(f"[Loading] PDF: {pdf_path}")
    print(f"[Parameters] Using SemanticChunker with {breakpoint_threshold_type}={breakpoint_threshold_amount}")

    # Load PDF
    loader = PyMuPDFLoader(pdf_path)
    pages = loader.load()

    # Add metadata
    for i, page in enumerate(pages):
        page.metadata.update({
            "page_number": i + 1,
            "book_title": "Thief of Sorrows",
            "author": "Kristen Long",
            "source_type": "pdf_fulltext"
        })

    # Split with SemanticChunker (same as simple_semantic_pdf_retriever.py)
    embeddings = OpenAIEmbeddings()
    text_splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type=breakpoint_threshold_type,
        breakpoint_threshold_amount=breakpoint_threshold_amount
    )

    # Use all pages for test generation
    pdf_chunks = text_splitter.split_documents(pages)

    print(f"[Created] {len(pdf_chunks)} semantic chunks from {len(pages)} pages")
    return pdf_chunks

def generate_test_questions(pdf_chunks, testset_size: int = 20, use_multiHop: bool = False):
    """
    Generate test questions from PDF chunks using RAGAS TestsetGenerator.
    """
    print(f"\n[Generating] {testset_size} test questions from PDF...")
    print(f"[Using] {len(pdf_chunks)} chunks for generation")

    # Initialize generator LLMs
    generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
    generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

    # Create TestsetGenerator
    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings
    )

    # Define query distribution
    if use_multiHop:
        query_distribution = [
            (SingleHopSpecificQuerySynthesizer(llm=generator_llm), 0.5),
            (MultiHopAbstractQuerySynthesizer(llm=generator_llm), 0.25),
            (MultiHopSpecificQuerySynthesizer(llm=generator_llm), 0.25)
        ]
    else:
        query_distribution = [
            (SingleHopSpecificQuerySynthesizer(llm=generator_llm), 1.0)
        ]

    # Generate dataset - use more chunks for better coverage
    chunks_to_use = min(20, len(pdf_chunks))  # Use up to 20 chunks
    pdf_dataset = generator.generate_with_langchain_docs(
        pdf_chunks[:chunks_to_use],
        testset_size=testset_size,
        query_distribution=query_distribution
    )

    print(f"[Generated] {len(pdf_dataset)} questions")
    return pdf_dataset

def test_retriever_directly(pdf_retriever: SimpleSemanticPDFRetriever, test_questions: List[str]):
    """
    Quick test to verify SimpleSemanticPDFRetriever is working.
    """
    print("\n[RETRIEVER TEST]")
    print("="*50)

    for q in test_questions:
        docs = pdf_retriever.search(q, k=5)
        print(f"\nQuestion: {q}")
        print(f"Retrieved: {len(docs)} documents")
        if docs:
            print(f"Top result preview: {docs[0].page_content[:150]}...")
            print(f"Page: {docs[0].metadata.get('page_number', 'Unknown')}")

    print("="*50)

def run_full_evaluation_pipeline(
    pdf_path: str,
    testset_size: int = 10,
    breakpoint_threshold_type: str = BREAKPOINT_THRESHOLD_TYPE,
    breakpoint_threshold_amount: int = BREAKPOINT_THRESHOLD_AMOUNT,
    use_multiHop: bool = False
):
    """
    Complete evaluation pipeline using SimpleSemanticPDFRetriever.
    """
    print("="*70)
    print("RAGAS EVALUATION - SIMPLE SEMANTIC PDF RETRIEVER")
    print("="*70)
    print(f"Configuration:")
    print(f"  - Chunking Strategy: SemanticChunker")
    print(f"  - Threshold Type: {breakpoint_threshold_type}")
    print(f"  - Threshold Amount: {breakpoint_threshold_amount}")
    print(f"  - Test Set Size: {testset_size}")
    print(f"  - Multi-hop: {use_multiHop}")
    print("="*70)

    # Step 1: Load and chunk PDF with SemanticChunker
    pdf_chunks = load_and_chunk_pdf(pdf_path, breakpoint_threshold_type, breakpoint_threshold_amount)

    # Step 2: Generate test questions
    test_dataset = generate_test_questions(pdf_chunks, testset_size, use_multiHop)

    # Convert to pandas
    df_questions = test_dataset.to_pandas()
    print(f"\n[Sample Questions Generated]:")
    for i, row in df_questions.head(5).iterrows():
        print(f"  {i+1}. {row['user_input'][:80]}...")

    # Step 3: Initialize SimpleSemanticPDFRetriever
    print(f"\n[Initializing] SimpleSemanticPDFRetriever...")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    pdf_retriever = SimpleSemanticPDFRetriever(
        pdf_path,
        openai_api_key,
        breakpoint_threshold_type=breakpoint_threshold_type,
        breakpoint_threshold_amount=breakpoint_threshold_amount
    ).initialize(k=10)

    # Test retriever directly first
    test_questions = [
        "Who is the author of Thief of Sorrows?",
        "What is the main character's name?",
        "What happens in the story?"
    ]
    test_retriever_directly(pdf_retriever, test_questions)

    # Step 4: Use SimpleSemanticPDFRetriever with SimpleRetrieverChain
    print("\n[Mode] Using SimpleSemanticPDFRetriever with SimpleRetrieverChain (no agent routing)")
    chain = SimpleRetrieverChain(pdf_retriever)

    # Step 5: Initialize evaluator LLM and embeddings
    chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    evaluator_llm = LangchainLLMWrapper(chat_model)
    evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

    # Step 6: Prepare dataset for evaluation
    dataset = EvaluationDataset.from_pandas(df_questions)
    reset_eval_fields(dataset)

    # Step 7: Process each generated question with diagnostics
    print(f"\n[Processing] {len(dataset)} questions...")
    print("-"*70)

    for i, row in enumerate(dataset):
        q = getattr(row, "user_input", None) or getattr(row, "question", None)
        if not q:
            continue

        print(f"\n[{i+1}/{len(dataset)}] Question: {q[:80]}...")

        # Invoke the chain
        out = chain.invoke({"question": q})

        # Store results
        row.response = to_text(out["response"])
        row.retrieved_contexts = [d.page_content for d in out["context"]][:10]

        # Diagnostics
        print(f"  - Response length: {len(row.response)} chars")
        print(f"  - Retrieved contexts: {len(row.retrieved_contexts)}")
        if row.retrieved_contexts:
            print(f"  - First context sample: {row.retrieved_contexts[0][:80]}...")

    print("-"*70)

    # Step 8: Evaluate using RAGAS
    print("\n[Evaluating] Running RAGAS metrics...")
    results = evaluate_current_dataset(dataset, evaluator_llm, evaluator_embeddings)

    return results, df_questions

def save_metrics_to_markdown(metric_scores: Dict[str, float], output_path: str):
    """
    Save RAGAS metrics results to a Markdown table.

    Args:
        metric_scores: Dictionary of metric names and their average scores
        output_path: Path to save the markdown file
    """
    from datetime import datetime

    # Create markdown content
    markdown_content = f"""# RAGAS Evaluation Metrics - Simple Semantic PDF Retriever

**Evaluation Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

**Configuration:**
- Chunking Method: SemanticChunker
- Threshold Type: {BREAKPOINT_THRESHOLD_TYPE}
- Threshold Amount: {BREAKPOINT_THRESHOLD_AMOUNT}
- Model: gpt-4o-mini

## Metrics Results

| Metric | Score |
|--------|-------|
"""

    # Add each metric to the table
    for metric_name, score in metric_scores.items():
        # Format metric name for display
        display_name = metric_name.replace('_', ' ').title()
        markdown_content += f"| {display_name} | {score:.4f} |\n"

    # Add summary statistics
    avg_score = sum(metric_scores.values()) / len(metric_scores)
    markdown_content += f"\n**Average Score:** {avg_score:.4f}\n"

    # Add interpretation guide
    markdown_content += """
## Metric Descriptions

- **Faithfulness**: Measures factual consistency of the answer with the context
- **Context Recall**: Measures how well retrieved context aligns with ground truth
- **Context Precision**: Measures signal-to-noise ratio of retrieved contexts
- **Answer Relevancy**: Measures how relevant the answer is to the question
- **Factual Correctness**: Measures factual overlap between answer and ground truth

*Score Range: 0.0 (worst) to 1.0 (best)*
"""

    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)

# Main execution
if __name__ == "__main__":
    # Configuration
    PDF_PATH = "book_research/data/thief_of_sorrows.pdf"
    TESTSET_SIZE = 5  # Start small for testing

    # Run the complete pipeline with SemanticChunker
    results, questions_df = run_full_evaluation_pipeline(
        pdf_path=PDF_PATH,
        testset_size=TESTSET_SIZE,
        breakpoint_threshold_type=BREAKPOINT_THRESHOLD_TYPE,
        breakpoint_threshold_amount=BREAKPOINT_THRESHOLD_AMOUNT,
        use_multiHop=False
    )

    # Display results
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(results)

    if hasattr(results, 'to_pandas'):
        df_results = results.to_pandas()

        # Calculate averages
        print("\n[Average Scores]:")
        metrics = [
            'faithfulness',
            'context_recall',
            'context_precision',
            'answer_relevancy',
            'factual_correctness(mode=f1)'
        ]

        metric_scores = {}
        for metric in metrics:
            if metric in df_results.columns:
                avg_score = df_results[metric].mean()
                metric_scores[metric] = avg_score
                print(f"  {metric}: {avg_score:.3f}")

        save_metrics_to_markdown(metric_scores, "evaluation/simple_semantic_retrieval_metrics.md")
