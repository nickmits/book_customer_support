"""
RAGAS Evaluation for Book Agent with Automatic Test Generation
Using TestsetGenerator to create questions from PDF chunks
FIXED VERSION - Compatible with your RAGAS installation
"""
from dotenv import load_dotenv
import os
import asyncio
from typing import List, Dict, Any, Optional
from ragas import EvaluationDataset, evaluate, RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    LLMContextRecall, 
    Faithfulness, 
    FactualCorrectness,
    # Removed the metrics that aren't available in your version
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
import sys
import nest_asyncio

# Apply nest_asyncio FIRST
nest_asyncio.apply()

# Fix for Windows event loop issues  
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
load_dotenv()

# ALIGNED CHUNK PARAMETERS - USE SAME EVERYWHERE
CHUNK_SIZE = 800  # Aligned size
CHUNK_OVERLAP = 200  # Aligned overlap

# Since you might not have simple_pdf_retriever.py working, let's create it inline
def create_simple_pdf_retriever(pdf_path: str, chunk_size: int = CHUNK_SIZE, 
                               chunk_overlap: int = CHUNK_OVERLAP, k: int = 10):
    """
    Create a simple PDF retriever inline to avoid import issues.
    """
    from langchain_community.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings
    
    print(f"[Creating Retriever] chunk_size={chunk_size}, overlap={chunk_overlap}")
    
    # Load PDF
    loader = PyMuPDFLoader(pdf_path)
    pages = loader.load()
    
    # Add metadata
    for i, page in enumerate(pages):
        page.metadata.update({
            "page": i + 1,
            "source": pdf_path,
            "total_pages": len(pages)
        })
    
    # Split with aligned parameters
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = text_splitter.split_documents(pages)
    print(f"[Retriever] Created {len(chunks)} chunks from {len(pages)} pages")
    
    # Create vectorstore
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="pdf_chunks"
    )
    
    # Return as retriever
    return vectorstore.as_retriever(search_kwargs={"k": k})

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
    RAGAS evaluation with available metrics.
    """
    evaluation_dataset = EvaluationDataset.from_pandas(ds.to_pandas())
    
    # Only use metrics that are available in your version
    metrics = [
        LLMContextRecall(),
        Faithfulness(),
        FactualCorrectness()
    ]
    
    return evaluate(
        dataset=evaluation_dataset,
        metrics=metrics,
        llm=evaluator_llm,
        run_config=RunConfig(timeout=360),
        raise_exceptions=False,
    )

class SimpleRetrieverChain:
    """
    Direct retriever wrapper for better RAGAS compatibility.
    Tests retriever directly without agent routing.
    """
    
    def __init__(self, pdf_retriever):
        self.pdf_retriever = pdf_retriever
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Direct retrieval and QA without agent complexity."""
        question = inputs.get("question", "")
        
        try:
            # Step 1: Direct retrieval from PDF
            print(f"    → Retrieving for: {question[:50]}...")
            raw_contexts = self.pdf_retriever.get_relevant_documents(question)
            
            # Log retrieval details
            print(f"    → Retrieved {len(raw_contexts)} documents")
            if raw_contexts:
                print(f"    → First context preview: {raw_contexts[0].page_content[:100]}...")
            
            # Step 2: Generate answer based on retrieved contexts
            context_text = "\n\n".join([
                f"[Page {doc.metadata.get('page', 'Unknown')}]: {doc.page_content}"
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
            print(f"    → Error in retrieval: {e}")
            return {
                "response": f"Error: {str(e)}", 
                "context": []
            }

def load_and_chunk_pdf(pdf_path: str, chunk_size: int = CHUNK_SIZE, 
                      chunk_overlap: int = CHUNK_OVERLAP):
    """
    Load PDF and create chunks for test generation with aligned parameters.
    """
    print(f"[Loading] PDF: {pdf_path}")
    print(f"[Parameters] chunk_size={chunk_size}, overlap={chunk_overlap}")
    
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
    
    # Split into chunks with aligned parameters
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    pdf_chunks = text_splitter.split_documents(pages)
    
    print(f"[Created] {len(pdf_chunks)} chunks from {len(pages)} pages")
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

def test_retriever_directly(pdf_retriever, test_questions: List[str]):
    """
    Quick test to verify retriever is working.
    """
    print("\n[RETRIEVER TEST]")
    print("="*50)
    
    for q in test_questions:
        docs = pdf_retriever.get_relevant_documents(q)
        print(f"\nQuestion: {q}")
        print(f"Retrieved: {len(docs)} documents")
        if docs:
            print(f"Top result preview: {docs[0].page_content[:150]}...")
            print(f"Page: {docs[0].metadata.get('page', 'Unknown')}")
    
    print("="*50)

def run_full_evaluation_pipeline(
    pdf_path: str,
    testset_size: int = 10,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    use_multiHop: bool = False
):
    """
    Complete pipeline with aligned parameters and diagnostics.
    """
    print("="*70)
    print("RAGAS EVALUATION - SIMPLE RETRIEVER")
    print("="*70)
    print(f"Configuration:")
    print(f"  - Chunk Size: {chunk_size}")
    print(f"  - Chunk Overlap: {chunk_overlap}")
    print(f"  - Test Set Size: {testset_size}")
    print(f"  - Multi-hop: {use_multiHop}")
    print("="*70)
    
    # Step 1: Load and chunk PDF with aligned parameters
    pdf_chunks = load_and_chunk_pdf(pdf_path, chunk_size, chunk_overlap)
    
    # Step 2: Generate test questions
    test_dataset = generate_test_questions(pdf_chunks, testset_size, use_multiHop)
    
    # Convert to pandas
    df_questions = test_dataset.to_pandas()
    print(f"\n[Sample Questions Generated]:")
    for i, row in df_questions.head(5).iterrows():
        print(f"  {i+1}. {row['user_input'][:80]}...")
    
    # Step 3: Initialize retriever with aligned parameters
    pdf_retriever = create_simple_pdf_retriever(pdf_path, chunk_size, chunk_overlap, k=10)
    
    # Test retriever directly first
    test_questions = [
        "Who is the author of Thief of Sorrows?",
        "What is the main character's name?",
        "What happens in the story?"
    ]
    test_retriever_directly(pdf_retriever, test_questions)
    
    # Step 4: Use Simple Retriever Chain
    print("\n[Mode] Using Simple Retriever Chain (no agent routing)")
    chain = SimpleRetrieverChain(pdf_retriever)
    
    # Step 5: Initialize evaluator LLM
    chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    evaluator_llm = LangchainLLMWrapper(chat_model)
    
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
    results = evaluate_current_dataset(dataset, evaluator_llm)
    
    return results, df_questions

# Main execution
if __name__ == "__main__":
    # Configuration
    PDF_PATH = "book_research/data/thief_of_sorrows.pdf"
    TESTSET_SIZE = 5  # Start small for testing
    
    # Run the complete pipeline
    results, questions_df = run_full_evaluation_pipeline(
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
    
    if hasattr(results, 'to_pandas'):
        df_results = results.to_pandas()
        
        # Calculate averages
        print("\n[Average Scores]:")
        metrics = ['context_recall', 'faithfulness', 'factual_correctness(mode=f1)']
        
        for metric in metrics:
            if metric in df_results.columns:
                avg_score = df_results[metric].mean()
                print(f"  {metric}: {avg_score:.3f}")
        
        # Save results
        df_results.to_csv("evaluation/simple_retrieval_results.csv", index=False)
        print("\n[Saved] Results to 'evaluation/simple_retrieval_results.csv'")