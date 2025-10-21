"""
Production Retriever System - Best Performing Configuration
Based on RAGAS evaluation: Simple Retrieval + Semantic Chunking (Score: 0.7829)

This module provides the highest-performing retrieval system for the bookstore API.
Uses SemanticChunker for intelligent context-aware splitting with basic vector search.
"""

from typing import List, Optional, Tuple
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document
import pandas as pd
import os


def extract_pdf_metadata(pdf_path: str) -> dict:
    """Extract metadata from PDF file."""
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(pdf_path)
        metadata = doc.metadata

        pdf_metadata = {
            "title": metadata.get("title", "Unknown Title"),
            "author": metadata.get("author", "Unknown Author"),
            "subject": metadata.get("subject", ""),
            "keywords": metadata.get("keywords", ""),
        }

        doc.close()

        print(f"[INFO] Extracted PDF metadata:")
        print(f"  Title: {pdf_metadata['title']}")
        print(f"  Author: {pdf_metadata['author']}")

        return pdf_metadata

    except Exception as e:
        print(f"[WARNING] Could not extract PDF metadata: {e}")
        return {
            "title": "Unknown Title",
            "author": "Unknown Author",
            "subject": "",
            "keywords": "",
        }


def initialize_production_retrievers(
    csv_path: str = "book_research/data/space_exploration_books.csv",
    pdf_path: str = "book_research/data/thief_of_sorrows.pdf",
    openai_api_key: Optional[str] = None,
    breakpoint_threshold_type: str = "percentile",
    breakpoint_threshold_amount: int = 95,
    csv_search_k: int = 5,
    pdf_search_k: int = 10
) -> Tuple[Optional[object], Optional[object]]:
    """
    Initialize production-grade retrievers using the best performing configuration.

    Based on evaluation results:
    - Simple Retrieval + Semantic Chunking achieved 0.7829 average score
    - Significantly outperforms Advanced Ensemble retrieval (0.6935)

    Args:
        csv_path: Path to CSV book catalog
        pdf_path: Path to PDF book file
        openai_api_key: OpenAI API key (or will use env variable)
        breakpoint_threshold_type: Semantic chunker threshold type ("percentile", "standard_deviation", "interquartile")
        breakpoint_threshold_amount: Threshold value for semantic splitting
        csv_search_k: Number of results to return for CSV searches
        pdf_search_k: Number of results to return for PDF searches

    Returns:
        Tuple of (csv_retriever, pdf_retriever)
    """

    if not openai_api_key:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            print("[ERROR] OPENAI_API_KEY not found in environment")
            return None, None

    try:
        print("\n" + "="*70)
        print("Initializing Production Retriever System")
        print("   Configuration: Simple Retrieval + Semantic Chunking")
        print(f"   Performance: 0.7829 average (Best in evaluation)")
        print("="*70 + "\n")

        # Initialize embedding model (shared for both retrievers)
        embedding_model = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=openai_api_key
        )

        csv_retriever = None
        pdf_retriever = None

        # ============================================================================
        # CSV Retriever: Book Catalog Metadata
        # ============================================================================
        if os.path.exists(csv_path):
            print(f"[CSV] Loading book catalog from: {csv_path}")

            df = pd.read_csv(csv_path)
            print(f"[CSV] Loaded {len(df)} books from catalog")

            csv_documents = []
            for index, row in df.iterrows():
                book_text = f"""Title: {row['title']}
Author: {row['author']}
Work Key: {row['work_key']}

Description: {row['description']}

Subjects: {row['subjects']}
"""
                metadata = {
                    "title": row['title'],
                    "author": row['author'],
                    "work_key": row['work_key'],
                    "subjects": row['subjects'],
                    "source_type": "csv_metadata",
                    "has_full_text": False,
                    "row_index": index
                }

                doc = Document(page_content=book_text.strip(), metadata=metadata)
                csv_documents.append(doc)

            # Create CSV vector store with simple retrieval
            print(f"[CSV] Creating vector store with {len(csv_documents)} documents...")
            csv_vectorstore = Qdrant.from_documents(
                documents=csv_documents,
                embedding=embedding_model,
                location=":memory:",
                collection_name="csv_catalog_semantic"
            )

            csv_retriever = csv_vectorstore.as_retriever(
                search_kwargs={"k": csv_search_k}
            )

            print(f"[CSV] Retriever ready (returns top {csv_search_k} results)")
        else:
            print(f"[CSV] WARNING: File not found: {csv_path}")

        # ============================================================================
        # PDF Retriever: Full Text with Semantic Chunking
        # ============================================================================
        if os.path.exists(pdf_path):
            print(f"\n[PDF] Loading full text from: {pdf_path}")

            # Extract PDF metadata
            pdf_metadata = extract_pdf_metadata(pdf_path)

            # Load PDF pages
            loader = PyMuPDFLoader(pdf_path)
            pdf_pages = loader.load()
            print(f"[PDF] Loaded {len(pdf_pages)} pages")

            # Add metadata to pages
            for i, page in enumerate(pdf_pages):
                page.metadata.update({
                    "book_title": pdf_metadata["title"],
                    "author": pdf_metadata["author"],
                    "source_type": "pdf_fulltext",
                    "has_full_text": True,
                    "page_number": i + 1,
                    "total_pages": len(pdf_pages),
                })

            # SEMANTIC CHUNKING: Intelligent context-aware splitting
            print(f"[PDF] Using SemanticChunker ({breakpoint_threshold_type}={breakpoint_threshold_amount})")
            print(f"[PDF] This creates coherent chunks based on semantic similarity")

            text_splitter = SemanticChunker(
                embedding_model,
                breakpoint_threshold_type=breakpoint_threshold_type,
                breakpoint_threshold_amount=breakpoint_threshold_amount
            )

            pdf_chunks = text_splitter.split_documents(pdf_pages)
            print(f"[PDF] Created {len(pdf_chunks)} semantic chunks from {len(pdf_pages)} pages")

            # Create PDF vector store with simple retrieval
            print(f"[PDF] Creating vector store with semantic chunks...")
            pdf_vectorstore = Qdrant.from_documents(
                documents=pdf_chunks,
                embedding=embedding_model,
                location=":memory:",
                collection_name="pdf_fulltext_semantic"
            )

            pdf_retriever = pdf_vectorstore.as_retriever(
                search_kwargs={"k": pdf_search_k}
            )

            print(f"[PDF] Retriever ready (returns top {pdf_search_k} results)")
        else:
            print(f"[PDF] WARNING: File not found: {pdf_path}")

        print("\n" + "="*70)
        print("Production Retriever System Initialized Successfully")
        print(f"   CSV Retriever: {'Ready' if csv_retriever else 'Not Available'}")
        print(f"   PDF Retriever: {'Ready' if pdf_retriever else 'Not Available'}")
        print("="*70 + "\n")

        return csv_retriever, pdf_retriever

    except Exception as e:
        print(f"[ERROR] Failed to initialize production retrievers: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    """Test the production retriever system"""
    from dotenv import load_dotenv
    load_dotenv()

    print("Testing Production Retriever System...\n")

    csv_ret, pdf_ret = initialize_production_retrievers()

    if csv_ret:
        print("\n[TEST] CSV Retriever:")
        results = csv_ret.get_relevant_documents("space exploration books")
        for i, doc in enumerate(results[:3], 1):
            print(f"{i}. {doc.metadata.get('title', 'Unknown')}")

    if pdf_ret:
        print("\n[TEST] PDF Retriever:")
        results = pdf_ret.get_relevant_documents("What is the main character's name?")
        for i, doc in enumerate(results[:3], 1):
            page = doc.metadata.get('page_number', '?')
            preview = doc.page_content[:100].replace('\n', ' ')
            print(f"{i}. Page {page}: {preview}...")
