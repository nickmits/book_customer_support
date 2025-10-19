"""
RAGAS Evaluation for Book Agent with Automatic Test Generation
Using TestsetGenerator to create questions from PDF chunks
"""
from dotenv import load_dotenv
import os
import asyncio
from typing import List, Dict, Any, Optional
from ragas import EvaluationDataset, evaluate, RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    LLMContextRecall, Faithfulness, FactualCorrectness,
)
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers import (
    default_query_distribution, 
    SingleHopSpecificQuerySynthesizer,
    MultiHopAbstractQuerySynthesizer,
    MultiHopSpecificQuerySynthesizer
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd
import os

# Import your components
from book_research.book_agent import build_book_agent
from book_research.simple_pdf_retriever import create_simple_pdf_retriever
# Replace lines 29-33 with:
import asyncio
import sys
import nest_asyncio

# Apply nest_asyncio FIRST
nest_asyncio.apply()

# Fix for Windows event loop issues  
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
load_dotenv()

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


def evaluate_current_dataset(ds, evaluator_llm):
    """
    Exact function from your pattern for RAGAS evaluation.
    """
    evaluation_dataset = EvaluationDataset.from_pandas(ds.to_pandas())
    return evaluate(
        dataset=evaluation_dataset,
        metrics=[
            LLMContextRecall(), Faithfulness(), FactualCorrectness()
        ],
        llm=evaluator_llm,
        run_config=RunConfig(timeout=360),
        raise_exceptions=False,
    )


class BookAgentChain:
    """
    Wrapper to make build_book_agent compatible with the chain.invoke() pattern.
    """
    
    def __init__(self, pdf_retriever, csv_retriever=None):
        # Use provided retrievers
        self.pdf_retriever = pdf_retriever
        self.csv_retriever = csv_retriever
        
        # Build agent
        self.agent = build_book_agent(
            csv_retriever=self.csv_retriever,
            pdf_retriever=self.pdf_retriever
        )
        
        # Agent configuration
        self.config = {
            "configurable": {
                "chat_model": "gpt-4o-mini",
                "max_tokens": 1000,
                "temperature": 0.1,
                "allow_clarification": False,
                "max_web_search_results": 3,
                "openai_api_key": os.getenv("OPENAI_API_KEY"),
                "tavily_api_key": os.getenv("TAVILY_API_KEY"),
            }
        }
    
    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous invoke matching the chain pattern."""
        return asyncio.run(self.ainvoke(inputs))
    
    async def ainvoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process question and return response + context."""
        question = inputs.get("question", "")
        
        try:
            # Invoke agent
            result = await self.agent.ainvoke(
                {"messages": [{"role": "user", "content": question}]},
                self.config
            )
            
            # Extract response
            final_response = result.get("final_response", "")
            
            # Extract contexts from search results
            contexts = []
            search_results = result.get("search_results", {})
            books = search_results.get("books", [])
            
            for book in books:
                if isinstance(book, dict):
                    if "content" in book:  # PDF passage
                        content = book.get("content", "")
                        page = book.get("page", "Unknown")
                        context_doc = MockDocument(f"[Page {page}] {content}")
                    elif "description" in book:  # CSV book
                        title = book.get("title", "Unknown")
                        author = book.get("author", "Unknown")
                        desc = book.get("description", "")
                        context_doc = MockDocument(f"{title} by {author}: {desc}")
                    else:
                        context_doc = MockDocument(str(book))
                    contexts.append(context_doc)
            
            return {
                "response": final_response,
                "context": contexts[:10]
            }
            
        except Exception as e:
            print(f"Error: {e}")
            return {"response": f"Error: {str(e)}", "context": []}


class MockDocument:
    """Document class for compatibility."""
    def __init__(self, content: str):
        self.page_content = content
        self.metadata = {}


def load_and_chunk_pdf(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Load PDF and create chunks for test generation.
    
    Args:
        pdf_path: Path to PDF file
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
    
    Returns:
        List of document chunks
    """
    print(f"[Loading] PDF: {pdf_path}")
    
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
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    pdf_chunks = text_splitter.split_documents(pages)
    
    print(f"[Created] {len(pdf_chunks)} chunks from {len(pages)} pages")
    return pdf_chunks


def generate_test_questions(
    pdf_chunks,
    testset_size: int = 20,
    use_multiHop: bool = False
):
    """
    Generate test questions from PDF chunks using RAGAS TestsetGenerator.
    
    Args:
        pdf_chunks: List of document chunks
        testset_size: Number of questions to generate
        use_multiHop: Whether to include multi-hop queries
    
    Returns:
        Generated dataset with questions
    """
    print(f"\n[Generating] {testset_size} test questions from PDF...")
    
    # Initialize generator LLMs following your pattern
    generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
    generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
    
    # Create TestsetGenerator
    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings
    )
    
    # Define query distribution
    if use_multiHop:
        # Mix of single and multi-hop questions
        query_distribution = [
            (SingleHopSpecificQuerySynthesizer(llm=generator_llm), 0.5),
            (MultiHopAbstractQuerySynthesizer(llm=generator_llm), 0.25),
            (MultiHopSpecificQuerySynthesizer(llm=generator_llm), 0.25)
        ]
    else:
        # Only single-hop questions (your pattern)
        query_distribution = [
            (SingleHopSpecificQuerySynthesizer(llm=generator_llm), 1.0)
        ]
    
    # Generate dataset
    pdf_dataset = generator.generate_with_langchain_docs(
        pdf_chunks[:5],
        testset_size=testset_size,
        query_distribution=query_distribution
    )
    
    print(f"[Generated] {len(pdf_dataset)} questions")
    return pdf_dataset


def run_full_evaluation_pipeline(
    pdf_path: str,
    testset_size: int = 20,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    use_multiHop: bool = False
):
    """
    Complete pipeline: Load PDF → Generate Questions → Evaluate Agent.
    
    Args:
        pdf_path: Path to PDF file
        testset_size: Number of test questions to generate
        chunk_size: Size of text chunks for question generation
        chunk_overlap: Overlap between chunks
        use_multiHop: Whether to include multi-hop queries
    
    Returns:
        RAGAS evaluation results
    """
    print("="*70)
    print("RAGAS EVALUATION PIPELINE")
    print("="*70)
    
    # Step 1: Load and chunk PDF
    pdf_chunks = load_and_chunk_pdf(pdf_path, chunk_size, chunk_overlap)
    
    # Step 2: Generate test questions
    test_dataset = generate_test_questions(pdf_chunks, testset_size, use_multiHop)
    
    # Convert to pandas to see the questions
    df_questions = test_dataset.to_pandas()
    print(f"\n[Sample Questions Generated]:")
    for i, row in df_questions.head(5).iterrows():
        print(f"  {i+1}. {row['user_input'][:80]}...")
    
    # Step 3: Initialize the book agent chain
    pdf_retriever = create_simple_pdf_retriever(pdf_path, k=10)
    book_agent_chain = BookAgentChain(pdf_retriever)
    
    # Step 4: Initialize evaluator LLM
    chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    evaluator_llm = LangchainLLMWrapper(chat_model)
    
    # Step 5: Prepare dataset for evaluation
    dataset = EvaluationDataset.from_pandas(df_questions)
    
    # Step 6: Reset evaluation fields
    reset_eval_fields(dataset)
    
    # Step 7: Process each generated question
    print(f"\n[Processing] {len(dataset)} questions through agent...")
    for i, row in enumerate(dataset):
    # The row IS the sample, not row.eval_sample
        q = getattr(row, "user_input", None) or getattr(row, "question", None)
        if not q: 
            continue
        
        print(f"  [{i+1}/{len(dataset)}] {q[:60]}...")
        
        # Invoke the book agent chain
        out = book_agent_chain.invoke({"question": q})
        
        # Store results
        row.response = to_text(out["response"])
        row.retrieved_contexts = [d.page_content for d in out["context"]][:10]
    
    # Step 8: Evaluate using your exact function
    print("\n[Evaluating] Running RAGAS metrics...")
    results = evaluate_current_dataset(dataset, evaluator_llm)
    
    return results, df_questions


# Main execution
if __name__ == "__main__":
    # Configuration
    PDF_PATH = "book_research/data/thief_of_sorrows.pdf"
    TESTSET_SIZE = 5  # Number of questions to generate
    
    # Run the complete pipeline
    results, questions_df = run_full_evaluation_pipeline(
        pdf_path=PDF_PATH,
        testset_size=TESTSET_SIZE,
        chunk_size=1000,
        chunk_overlap=200,
        use_multiHop=False  
    )
    
    # Display results
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(results)
    
    # Detailed results if available
    if hasattr(results, 'to_pandas'):
        df_results = results.to_pandas()
        
        # Show some examples
        print("\n[Sample Evaluated Questions]:")
        
        # Calculate averages
        print("\n[Average Scores]:")
        for metric in ['context_recall', 'faithfulness', 'factual_correctness(mode=f1)']:  
            avg_score = df_results[metric].mean()
            print(f"  {metric}: {avg_score:.3f}")
        
        # Save results
        df_results.to_csv("ragas_evaluation_results.csv", index=False)
        print("\n[Saved] Results to 'ragas_evaluation_results.csv'")