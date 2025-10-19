"""
RAGAS Evaluation for Advanced Retrieval System
Evaluates the advanced retrieval system with Ensemble, BM25, Multi-Query, and Cohere reranking
FIXED VERSION - Aligned chunk sizes for fair evaluation
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

# Import the advanced retrieval components
from langchain_community.vectorstores import Qdrant
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain.chat_models import init_chat_model

# Apply nest_asyncio for event loop management
nest_asyncio.apply()
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

load_dotenv()

# CRITICAL: ALIGNED CHUNK PARAMETERS - MUST BE SAME EVERYWHERE
CHUNK_SIZE = 800  # Same as simple retriever
CHUNK_OVERLAP = 200  # Same as simple retriever

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

def initialize_advanced_retrievers(chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
    """
    Initialize the advanced CSV and PDF retrievers with aligned chunk sizes.
    """
    print("[INIT] Initializing advanced retrievers...")
    print(f"[PARAMS] chunk_size={chunk_size}, overlap={chunk_overlap}")
    
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
    # Load CSV Data (Book Catalog) - Optional for this evaluation
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
        
        csv_bm25 = BM25Retriever.from_documents(csv_documents)
        csv_bm25.k = 5
        
        csv_multi_query = MultiQueryRetriever.from_llm(
            retriever=csv_vectorstore.as_retriever(search_kwargs={"k": 10}),
            llm=chat_model
        )
        
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
    # Load PDF Data (Thief of Sorrows) - CRITICAL: Use aligned chunk sizes
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
                "total_pages": len(pdf_pages),
                "page": i + 1  # Add for compatibility
            })
        
        # CRITICAL FIX: Use aligned chunk sizes
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,  # ALIGNED with test generation
            chunk_overlap=chunk_overlap,  # ALIGNED with test generation
            length_function=len
        )
        pdf_chunks = text_splitter.split_documents(pdf_pages)
        print(f"[INFO] Created {len(pdf_chunks)} chunks from PDF with size {chunk_size}")
        
        # Create advanced PDF retriever
        pdf_vectorstore = Qdrant.from_documents(
            documents=pdf_chunks,
            embedding=embedding_model,
            location=":memory:",
            collection_name="pdf_fulltext"
        )
        
        # BM25 for keyword search
        pdf_bm25 = BM25Retriever.from_documents(pdf_chunks)
        pdf_bm25.k = 10
        
        # Multi-query for better coverage
        pdf_multi_query = MultiQueryRetriever.from_llm(
            retriever=pdf_vectorstore.as_retriever(search_kwargs={"k": 15}),
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
        
        # Test the retriever immediately
        test_questions = [
            "Who is the author of Thief of Sorrows?",
            "What is the main character's name?"
        ]
        test_retriever_directly(pdf_retriever, test_questions)
    
    return csv_retriever, pdf_retriever

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

def load_and_chunk_pdf(pdf_path: str, chunk_size: int = CHUNK_SIZE, 
                      chunk_overlap: int = CHUNK_OVERLAP):
    """Load PDF and create chunks with ALIGNED parameters."""
    print(f"[Loading] PDF for test generation: {pdf_path}")
    print(f"[CRITICAL] Using chunk_size={chunk_size}, overlap={chunk_overlap}")
    
    loader = PyMuPDFLoader(pdf_path)
    pages = loader.load()
    
    for i, page in enumerate(pages):
        page.metadata.update({
            "page_number": i + 1,
            "book_title": "Thief of Sorrows",
            "author": "Kristen Long",
            "source_type": "pdf_fulltext"
        })
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    pdf_chunks = text_splitter.split_documents(pages)
    
    print(f"[Created] {len(pdf_chunks)} chunks from {len(pages)} pages")
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
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    use_multiHop: bool = False
):
    """
    Complete evaluation pipeline for the advanced retrieval system.
    """
    print("="*70)
    print("ADVANCED RETRIEVAL SYSTEM - RAGAS EVALUATION")
    print("="*70)
    print(f"Configuration:")
    print(f"  - Chunk Size: {chunk_size}")
    print(f"  - Chunk Overlap: {chunk_overlap}")
    print(f"  - Test Set Size: {testset_size}")
    print(f"  - Multi-hop: {use_multiHop}")
    print("="*70)
    
    # Step 1: Initialize advanced retrievers with aligned sizes
    csv_retriever, pdf_retriever = initialize_advanced_retrievers(chunk_size, chunk_overlap)
    
    if not pdf_retriever:
        print("[ERROR] PDF retriever not initialized. Check your data files.")
        return None, None
    
    # Step 2: Load and chunk PDF for test generation with SAME parameters
    pdf_chunks = load_and_chunk_pdf(pdf_path, chunk_size, chunk_overlap)
    
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
    
    # Step 5: Initialize evaluator LLM
    chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    evaluator_llm = LangchainLLMWrapper(chat_model)
    
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
    results = evaluate_current_dataset(dataset, evaluator_llm)
    
    return results, df_questions

# Main execution
if __name__ == "__main__":
    # Configuration
    PDF_PATH = "book_research/data/thief_of_sorrows.pdf"
    TESTSET_SIZE = 10  # Increased for better evaluation

    # Run advanced retrieval evaluation
    print("\n" + "="*70)
    print("ADVANCED RETRIEVAL SYSTEM EVALUATION")
    print("="*70)

    results, questions_df = run_advanced_evaluation_pipeline(
        pdf_path=PDF_PATH,
        testset_size=TESTSET_SIZE,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        use_multiHop=False
    )

    # Display results
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(results)

    if results and hasattr(results, 'to_pandas'):
        df_results = results.to_pandas()

        print("\n[Average Scores]:")
        for metric in ['context_recall', 'faithfulness', 'factual_correctness(mode=f1)']:
            if metric in df_results.columns:
                avg_score = df_results[metric].mean()
                print(f"  {metric}: {avg_score:.3f}")

        df_results.to_csv("evaluation/advanced_retrieval_results.csv", index=False)
        print("\n[Saved] Results to 'evaluation/advanced_retrieval_results.csv'")

        # Compare with simple retriever if available
        simple_results_path = "evaluation/simple_retrieval_results.csv"
        if os.path.exists(simple_results_path):
            simple_df = pd.read_csv(simple_results_path)
            print("\n[Comparison with Simple Retriever]:")
            print(f"  Simple - Context Recall: {simple_df['context_recall'].mean():.3f}")
            print(f"  Advanced - Context Recall: {df_results['context_recall'].mean():.3f}")
            improvement = df_results['context_recall'].mean() - simple_df['context_recall'].mean()
            print(f"  Improvement: {improvement:+.3f}")