"""
RAGAS Evaluation for Advanced Book Agent with Multiple Retrievers
Tests the full agent with CSV catalog, PDF retriever, and intelligent routing
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
    LLMContextRecall, Faithfulness, FactualCorrectness,
)
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers import (
    SingleHopSpecificQuerySynthesizer,
    MultiHopAbstractQuerySynthesizer,
    MultiHopSpecificQuerySynthesizer
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd

# Import the advanced retrieval components from main.py
from langchain_community.vectorstores import Qdrant
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain.chat_models import init_chat_model

# Import your book agent components
from book_research.book_agent import build_book_agent
from book_research.configuration import Configuration
from book_research.state import AgentState

# Apply nest_asyncio for event loop management
nest_asyncio.apply()
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

load_dotenv()


def initialize_advanced_retrievers():
    """
    Initialize the advanced CSV and PDF retrievers with multiple strategies.
    Based on the implementation in main.py
    """
    print("[INIT] Initializing advanced retrievers...")
    
    # Get API keys
    openai_api_key = os.getenv("OPENAI_API_KEY")
    cohere_api_key = os.getenv("COHERE_API_KEY")
    
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in environment")
    
    # Initialize models
    chat_model = init_chat_model(
        model="openai:gpt-4o-mini",
        api_key=openai_api_key,
        temperature=0.1,
        max_tokens=1000
    )
    
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=openai_api_key
    )
    
    # ============================================================================
    # Load CSV Data (Book Catalog)
    # ============================================================================
    csv_path = "book_research/data/space_exploration_books.csv"
    csv_retriever = None
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"[INFO] Loaded {len(df)} books from CSV catalog")
        
        csv_documents = []
        for index, row in df.iterrows():
            book_text = f"""Title: {row['title']}
Author: {row['author']}
Work Key: {row['work_key']}
Description: {row['description']}
Subjects: {row['subjects']}"""
            
            metadata = {
                "title": row['title'],
                "author": row['author'],
                "work_key": row['work_key'],
                "subjects": row['subjects'],
                "source_type": "csv_metadata",
                "row_index": index
            }
            
            doc = Document(page_content=book_text.strip(), metadata=metadata)
            csv_documents.append(doc)
        
        # Create advanced CSV retriever
        csv_vectorstore = Qdrant.from_documents(
            documents=csv_documents,
            embedding=embedding_model,
            location=":memory:",
            collection_name="csv_catalog"
        )
        
        # BM25 for keyword search
        csv_bm25 = BM25Retriever.from_documents(csv_documents)
        csv_bm25.k = 5
        
        # Multi-query for query expansion
        csv_multi_query = MultiQueryRetriever.from_llm(
            retriever=csv_vectorstore.as_retriever(search_kwargs={"k": 10}),
            llm=chat_model
        )
        
        # Ensemble with optional reranking
        if cohere_api_key:
            csv_compression = ContextualCompressionRetriever(
                base_retriever=csv_multi_query,
                base_compressor=CohereRerank(model="rerank-v3.5", cohere_api_key=cohere_api_key)
            )
            csv_retriever = EnsembleRetriever(
                retrievers=[csv_bm25, csv_multi_query, csv_compression],
                weights=[0.2, 0.3, 0.5]
            )
            print("[OK] CSV retriever with Cohere reranking created")
        else:
            csv_retriever = EnsembleRetriever(
                retrievers=[csv_bm25, csv_multi_query],
                weights=[0.4, 0.6]
            )
            print("[OK] CSV retriever created (no reranking)")
    
    # ============================================================================
    # Load PDF Data (Thief of Sorrows) - Full document
    # ============================================================================
    pdf_path = "book_research/data/thief_of_sorrows.pdf"
    pdf_retriever = None
    
    if os.path.exists(pdf_path):
        loader = PyMuPDFLoader(pdf_path)
        pdf_pages = loader.load()
        print(f"[INFO] Loaded {len(pdf_pages)} pages from PDF")
        
        # Add metadata
        for i, page in enumerate(pdf_pages):
            page.metadata.update({
                "book_title": "Thief of Sorrows",
                "author": "Kristen Long",
                "source_type": "pdf_fulltext",
                "page_number": i + 1,
                "total_pages": len(pdf_pages)
            })
        
        # Split into chunks with smaller size for better retrieval
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Smaller chunks for better precision
            chunk_overlap=100
        )
        pdf_chunks = text_splitter.split_documents(pdf_pages)
        print(f"[INFO] Created {len(pdf_chunks)} chunks from PDF")
        
        # Create advanced PDF retriever
        pdf_vectorstore = Qdrant.from_documents(
            documents=pdf_chunks,
            embedding=embedding_model,
            location=":memory:",
            collection_name="pdf_fulltext"
        )
        
        # BM25 for keyword search
        pdf_bm25 = BM25Retriever.from_documents(pdf_chunks)
        pdf_bm25.k = 10  # More results for PDF
        
        # Multi-query for better coverage
        pdf_multi_query = MultiQueryRetriever.from_llm(
            retriever=pdf_vectorstore.as_retriever(search_kwargs={"k": 20}),  # More results
            llm=chat_model
        )
        
        # Ensemble with optional reranking
        if cohere_api_key:
            pdf_compression = ContextualCompressionRetriever(
                base_retriever=pdf_multi_query,
                base_compressor=CohereRerank(model="rerank-v3.5", cohere_api_key=cohere_api_key)
            )
            pdf_retriever = EnsembleRetriever(
                retrievers=[pdf_bm25, pdf_multi_query, pdf_compression],
                weights=[0.3, 0.3, 0.4]
            )
            print("[OK] PDF retriever with Cohere reranking created")
        else:
            pdf_retriever = EnsembleRetriever(
                retrievers=[pdf_bm25, pdf_multi_query],
                weights=[0.5, 0.5]
            )
            print("[OK] PDF retriever created (no reranking)")
    
    return csv_retriever, pdf_retriever


class AdvancedBookAgentChain:
    """
    Wrapper for the advanced book agent with intelligent routing.
    """
    
    def __init__(self, csv_retriever, pdf_retriever):
        self.csv_retriever = csv_retriever
        self.pdf_retriever = pdf_retriever
        
        # Build the full agent with both retrievers
        self.agent = build_book_agent(
            csv_retriever=self.csv_retriever,
            pdf_retriever=self.pdf_retriever
        )
        
        # Configuration for the agent
        self.config = {
            "configurable": {
                "chat_model": "gpt-4o-mini",
                "max_tokens": 1000,
                "temperature": 0.1,
                "allow_clarification": False,
                "max_web_search_results": 2,
                "openai_api_key": os.getenv("OPENAI_API_KEY"),
                "tavily_api_key": os.getenv("TAVILY_API_KEY"),
            }
        }
    
    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous invoke for RAGAS compatibility."""
        return asyncio.run(self.ainvoke(inputs))
    
    async def ainvoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process question through the advanced agent."""
        question = inputs.get("question", "")
        
        try:
            # Invoke the full agent with routing
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
            
            # Convert search results to document format
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
            
            # Debug: Show routing decision
            routing = search_results.get("routing_reasoning", "")
            if routing:
                print(f"    → Routing: {search_results.get('search_target', 'unknown')}")
            
            return {
                "response": final_response,
                "context": contexts[:10]
            }
            
        except Exception as e:
            print(f"    → Error: {e}")
            return {"response": f"Error: {str(e)}", "context": []}


class MockDocument:
    """Document class for RAGAS compatibility."""
    def __init__(self, content: str):
        self.page_content = content
        self.metadata = {}


def to_text(content):
    """Convert content to text."""
    return str(content) if content else ""


def reset_eval_fields(dataset):
    """Reset evaluation fields in dataset."""
    for row in dataset:
        row.response = None
        row.retrieved_contexts = []


def evaluate_current_dataset(ds, evaluator_llm):
    """Standard RAGAS evaluation function."""
    evaluation_dataset = EvaluationDataset.from_pandas(ds.to_pandas())
    return evaluate(
        dataset=evaluation_dataset,
        metrics=[
            LLMContextRecall(), 
            Faithfulness(), 
            FactualCorrectness()
        ],
        llm=evaluator_llm,
        run_config=RunConfig(timeout=360),
        raise_exceptions=False,
    )


def load_and_chunk_pdf(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Load PDF and create chunks for test generation."""
    print(f"[Loading] PDF for test generation: {pdf_path}")
    
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
    
    print(f"[Created] {len(pdf_chunks)} chunks from {len(pages)} pages for test generation")
    return pdf_chunks


def generate_test_questions(pdf_chunks, testset_size: int = 5, use_multiHop: bool = False):
    """Generate test questions from PDF chunks using RAGAS."""
    print(f"\n[Generating] {testset_size} test questions from PDF...")
    
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
    
    # Generate dataset - use only first 5 chunks for speed
    pdf_dataset = generator.generate_with_langchain_docs(
        pdf_chunks[:5],  
        testset_size=testset_size,
        query_distribution=query_distribution
    )
    
    print(f"[Generated] {len(pdf_dataset)} questions")
    return pdf_dataset


def run_advanced_evaluation_pipeline(
    pdf_path: str,
    testset_size: int = 5,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    use_multiHop: bool = False
):
    """
    Complete evaluation pipeline for the advanced book agent.
    """
    print("="*70)
    print("ADVANCED BOOK AGENT - RAGAS EVALUATION")
    print("="*70)
    
    # Step 1: Initialize advanced retrievers
    csv_retriever, pdf_retriever = initialize_advanced_retrievers()
    
    if not csv_retriever and not pdf_retriever:
        print("[ERROR] No retrievers initialized. Check your data files.")
        return None, None
    
    # Step 2: Load and chunk PDF for test generation
    pdf_chunks = load_and_chunk_pdf(pdf_path, chunk_size, chunk_overlap)
    
    # Step 3: Generate test questions
    test_dataset = generate_test_questions(pdf_chunks, testset_size, use_multiHop)
    
    # Convert to pandas
    df_questions = test_dataset.to_pandas()
    print(f"\n[Sample Questions Generated]:")
    for i, row in df_questions.head(5).iterrows():
        print(f"  {i+1}. {row['user_input'][:80]}...")
    
    # Step 4: Initialize the advanced book agent
    advanced_agent_chain = AdvancedBookAgentChain(csv_retriever, pdf_retriever)
    
    # Step 5: Initialize evaluator LLM
    chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    evaluator_llm = LangchainLLMWrapper(chat_model)
    
    # Step 6: Prepare dataset for evaluation
    dataset = EvaluationDataset.from_pandas(df_questions)
    reset_eval_fields(dataset)
    
    # Step 7: Process each question through the advanced agent
    print(f"\n[Processing] {len(dataset)} questions through advanced agent...")
    for i, row in enumerate(dataset):
        q = getattr(row, "user_input", None) or getattr(row, "question", None)
        if not q:
            continue
        
        print(f"  [{i+1}/{len(dataset)}] {q[:60]}...")
        
        # Invoke the advanced agent
        out = advanced_agent_chain.invoke({"question": q})
        
        # Store results
        row.response = to_text(out["response"])
        row.retrieved_contexts = [d.page_content for d in out["context"]][:10]
    
    # Step 8: Evaluate with RAGAS
    print("\n[Evaluating] Running RAGAS metrics...")
    results = evaluate_current_dataset(dataset, evaluator_llm)
    
    return results, df_questions


# Main execution
if __name__ == "__main__":
    # Configuration
    PDF_PATH = "book_research/data/thief_of_sorrows.pdf"
    TESTSET_SIZE = 5  # Start with 5 questions
    
    # Run the evaluation pipeline
    results, questions_df = run_advanced_evaluation_pipeline(
        pdf_path=PDF_PATH,
        testset_size=TESTSET_SIZE,
        chunk_size=1000,
        chunk_overlap=200,
        use_multiHop=False
    )
    
    # Display results
    print("\n" + "="*70)
    print("EVALUATION RESULTS - ADVANCED AGENT")
    print("="*70)
    print(results)
    
    # Detailed results
    if results and hasattr(results, 'to_pandas'):
        df_results = results.to_pandas()
        
        # Calculate averages
        print("\n[Average Scores]:")
        for metric in ['context_recall', 'faithfulness', 'factual_correctness(mode=f1)']:
            if metric in df_results.columns:
                avg_score = df_results[metric].mean()
                print(f"  {metric}: {avg_score:.3f}")
        
        # Save results
        df_results.to_csv("ragas_advanced_evaluation_results.csv", index=False)
        print("\n[Saved] Results to 'ragas_advanced_evaluation_results.csv'")
        
        # Compare with simple retriever if available
        if os.path.exists("ragas_evaluation_results.csv"):
            simple_df = pd.read_csv("ragas_evaluation_results.csv")
            print("\n[Comparison with Simple Retriever]:")
            print(f"  Simple - Context Recall: {simple_df['context_recall'].mean():.3f}")
            print(f"  Advanced - Context Recall: {df_results['context_recall'].mean():.3f}")
            print(f"  Improvement: {(df_results['context_recall'].mean() - simple_df['context_recall'].mean()):.3f}")