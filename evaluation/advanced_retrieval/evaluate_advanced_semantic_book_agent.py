"""
RAGAS Evaluation for Advanced Semantic Retrieval System
Evaluates the advanced retrieval system from semantic_main.py
Uses SemanticChunker with Ensemble, BM25, Multi-Query, and Cohere reranking
"""

from dotenv import load_dotenv
import os
import asyncio
import sys
import nest_asyncio
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

# Import the semantic retrieval system
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from semantic_main import initialize_semantic_retrievers, BREAKPOINT_THRESHOLD_TYPE, BREAKPOINT_THRESHOLD_AMOUNT

# Apply nest_asyncio for event loop management
nest_asyncio.apply()
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

load_dotenv()

def test_retriever_directly(retriever, test_questions: List[str]):
    """Quick test to verify retriever is working."""
    print("\n[RETRIEVER TEST]")
    print("="*50)

    for q in test_questions:
        docs = retriever.get_relevant_documents(q)
        print(f"\nQuestion: {q}")
        print(f"Retrieved: {len(docs)} documents")
        if docs:
            print(f"Top result preview: {docs[0].page_content[:150]}...")
            print(f"Metadata: {docs[0].metadata}")

    print("="*50)

class PDFOnlyAdvancedChain:
    """Direct PDF retrieval bypassing agent routing for fair evaluation."""
    
    def __init__(self, pdf_retriever):
        self.pdf_retriever = pdf_retriever
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Direct retrieval from PDF only."""
        question = inputs.get("question", "")
        
        try:
            # Direct retrieval from PDF
            print(f"    → PDF-only retrieving for: {question[:50]}...")
            raw_contexts = self.pdf_retriever.get_relevant_documents(question)
            
            print(f"    → Retrieved {len(raw_contexts)} documents")
            if raw_contexts:
                print(f"    → First context: {raw_contexts[0].page_content[:100]}...")
            
            # Generate answer from contexts
            context_text = "\n\n".join([
                f"[Page {doc.metadata.get('page', doc.metadata.get('page_number', 'Unknown'))}]: {doc.page_content}"
                for doc in raw_contexts[:5]
            ])
            
            prompt = f"""Based on the following context from the book "Thief of Sorrows", answer the question.
            
Context:
{context_text}

Question: {question}

Answer (be specific and use information from the context):"""
            
            response = self.llm.invoke(prompt)
            
            return {
                "response": response.content,
                "context": raw_contexts[:10]  # Return raw documents
            }
            
        except Exception as e:
            print(f"    → Error in retrieval: {e}")
            return {"response": f"Error: {str(e)}", "context": []}


def to_text(content):
    """Convert content to text."""
    return str(content) if content else ""

def reset_eval_fields(dataset):
    """Reset evaluation fields in dataset."""
    for row in dataset:
        row.response = None
        row.retrieved_contexts = []

def evaluate_current_dataset(ds, evaluator_llm, evaluator_embeddings):
    """
    RAGAS evaluation with all required certification metrics.
    Includes: Faithfulness, Context Recall, Context Precision, Answer Relevancy
    """
    evaluation_dataset = EvaluationDataset.from_pandas(ds.to_pandas())
    return evaluate(
        dataset=evaluation_dataset,
        metrics=[
            LLMContextRecall(),
            Faithfulness(),
            ContextPrecision(),
            AnswerRelevancy(),
            FactualCorrectness()  # Bonus metric
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=RunConfig(timeout=360),
        raise_exceptions=False,
    )

def load_and_chunk_pdf(pdf_path: str):
    """Load PDF and create semantic chunks for test generation."""
    from langchain_openai import OpenAIEmbeddings

    print(f"[Loading] PDF for test generation: {pdf_path}")
    print(f"[Using] SemanticChunker with {BREAKPOINT_THRESHOLD_TYPE} threshold")

    loader = PyMuPDFLoader(pdf_path)
    pages = loader.load()

    for i, page in enumerate(pages):
        page.metadata.update({
            "page_number": i + 1,
            "book_title": "Thief of Sorrows",
            "author": "Kristen Long",
            "source_type": "pdf_fulltext"
        })

    # Limit to first 20 pages for faster test generation
    pages_to_use = pages[:20]
    print(f"[Using] First {len(pages_to_use)} pages for test generation")

    embeddings = OpenAIEmbeddings()
    text_splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type=BREAKPOINT_THRESHOLD_TYPE,
        breakpoint_threshold_amount=BREAKPOINT_THRESHOLD_AMOUNT
    )
    pdf_chunks = text_splitter.split_documents(pages_to_use)

    print(f"[Created] {len(pdf_chunks)} semantic chunks from {len(pages_to_use)} pages")
    return pdf_chunks

def generate_test_questions(pdf_chunks, testset_size: int = 10, use_multiHop: bool = False):
    """Generate test questions from PDF chunks using RAGAS."""
    print(f"\n[Generating] {testset_size} test questions from PDF...")
    print(f"[Using] {len(pdf_chunks)} chunks for generation")
    
    generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
    generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
    
    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings
    )
    
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
    
    # Use more chunks for better coverage
    chunks_to_use = min(20, len(pdf_chunks))
    pdf_dataset = generator.generate_with_langchain_docs(
        pdf_chunks[:chunks_to_use],
        testset_size=testset_size,
        query_distribution=query_distribution
    )
    
    print(f"[Generated] {len(pdf_dataset)} questions")
    return pdf_dataset

def run_advanced_evaluation_pipeline(
    pdf_path: str,
    testset_size: int = 10,
    use_multiHop: bool = False
):
    """
    Complete evaluation pipeline for the advanced semantic retrieval system.
    Uses retrievers from semantic_main.py
    """
    print("="*70)
    print("ADVANCED SEMANTIC RETRIEVAL SYSTEM - EVALUATION")
    print("="*70)
    print(f"Configuration:")
    print(f"  - Chunking Strategy: SemanticChunker")
    print(f"  - Threshold Type: {BREAKPOINT_THRESHOLD_TYPE}")
    print(f"  - Threshold Amount: {BREAKPOINT_THRESHOLD_AMOUNT}")
    print(f"  - Test Set Size: {testset_size}")
    print(f"  - Multi-hop: {use_multiHop}")
    print("="*70)

    # Step 1: Initialize semantic retrievers from semantic_main.py
    print("\n[Importing] Retrievers from semantic_main.py...")
    csv_retriever, pdf_retriever = initialize_semantic_retrievers()

    if not pdf_retriever:
        print("[ERROR] PDF retriever not initialized. Check your data files.")
        return None, None

    # Test the retriever
    test_questions = [
        "Who is the author of Thief of Sorrows?",
        "What is the main character's name?"
    ]
    test_retriever_directly(pdf_retriever, test_questions)
    
    # Step 2: Load and chunk PDF for test generation with semantic chunking
    pdf_chunks = load_and_chunk_pdf(pdf_path)
    
    # Step 3: Generate test questions
    test_dataset = generate_test_questions(pdf_chunks, testset_size, use_multiHop)
    
    # Convert to pandas
    df_questions = test_dataset.to_pandas()
    print(f"\n[Sample Questions Generated]:")
    for i, row in df_questions.head(5).iterrows():
        print(f"  {i+1}. {row['user_input'][:80]}...")

    # Step 4: Initialize PDF-only chain
    print("\n[Mode] Using Advanced Retrieval System (PDF-only)")
    chain = PDFOnlyAdvancedChain(pdf_retriever)
    
    # Step 5: Initialize evaluator LLM and embeddings
    chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    evaluator_llm = LangchainLLMWrapper(chat_model)
    evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

    # Step 6: Prepare dataset for evaluation
    dataset = EvaluationDataset.from_pandas(df_questions)
    reset_eval_fields(dataset)
    
    # Step 7: Process each question
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

    # Step 8: Evaluate with RAGAS
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
    markdown_content = f"""# RAGAS Evaluation Metrics - Advanced Semantic Retrieval System

**Evaluation Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

**Configuration:**
- Chunking Method: SemanticChunker
- Threshold Type: {BREAKPOINT_THRESHOLD_TYPE}
- Threshold Amount: {BREAKPOINT_THRESHOLD_AMOUNT}
- Retrieval Components: Ensemble (BM25 + Multi-Query + Cohere Rerank)
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

## System Architecture

This evaluation uses the **Advanced Semantic Retrieval System** with:
- **SemanticChunker** for intelligent text splitting
- **Ensemble Retriever** combining BM25, Multi-Query, and Cohere Rerank
- **Qdrant** vector store for efficient similarity search
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)


if __name__ == "__main__":
    PDF_PATH = "book_research/data/thief_of_sorrows.pdf"
    TESTSET_SIZE = 10  

    print("\n" + "="*70)
    print("ADVANCED RETRIEVAL SYSTEM EVALUATION")
    print("="*70)

    results, questions_df = run_advanced_evaluation_pipeline(
        pdf_path=PDF_PATH,
        testset_size=TESTSET_SIZE,
        use_multiHop=False
    )


    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(results)

    if results and hasattr(results, 'to_pandas'):
        df_results = results.to_pandas()


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
        save_metrics_to_markdown(metric_scores, "evaluation/advanced_semantic_retrieval_metrics.md")
