"""
Advanced Recursive Retrieval System
Previous production system using RecursiveCharacterTextSplitter with Ensemble retrieval
RAGAS Score: 0.6935

This module contains the previous retrieval system for reference and comparison.
Uses Ensemble (BM25 + Multi-Query + Cohere Rerank) with recursive character chunking.
"""

from typing import List, Optional, Tuple
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain.chat_models import init_chat_model
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


def initialize_advanced_recursive_retrievers(
    csv_path: str = "book_research/data/space_exploration_books.csv",
    pdf_path: str = "book_research/data/thief_of_sorrows.pdf",
    openai_api_key: Optional[str] = None,
    cohere_api_key: Optional[str] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> Tuple[Optional[object], Optional[object]]:
    """
    Initialize Advanced Recursive retrieval system.

    Uses:
    - RecursiveCharacterTextSplitter for chunking
    - Ensemble retrieval (BM25 + Multi-Query + Cohere Rerank)
    - Qdrant vector store

    Performance: 0.6935 average score

    Args:
        csv_path: Path to CSV book catalog
        pdf_path: Path to PDF book file
        openai_api_key: OpenAI API key
        cohere_api_key: Cohere API key (optional, for reranking)
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks

    Returns:
        Tuple of (csv_retriever, pdf_retriever)
    """

    if not openai_api_key:
        openai_api_key = os.getenv("OPENAI_API_KEY")
    if not cohere_api_key:
        cohere_api_key = os.getenv("COHERE_API_KEY")

    if not openai_api_key:
        print("[ERROR] OPENAI_API_KEY not found in environment")
        return None, None

    try:
        print("\n" + "="*70)
        print("Initializing Advanced Recursive Retrieval System")
        print("   Configuration: Ensemble + RecursiveCharacterTextSplitter")
        print(f"   Performance: 0.6935 average")
        print("="*70 + "\n")

        # Initialize chat model for multi-query
        chat_model = init_chat_model(
            model="openai:gpt-4o-mini",
            api_key=openai_api_key,
            temperature=0.1,
            max_tokens=1000
        )

        # Initialize embedding model
        embedding_model = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=openai_api_key
        )

        csv_retriever = None
        pdf_retriever = None

        # ============================================================================
        # CSV Retriever: Book Catalog
        # ============================================================================
        if os.path.exists(csv_path):
            print(f"[CSV] Loading book catalog from: {csv_path}")

            df = pd.read_csv(csv_path)
            print(f"[CSV] Loaded {len(df)} books from CSV")

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

            # Build ensemble CSV retriever
            print("[CSV] Building ensemble retriever...")
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
            else:
                csv_retriever = EnsembleRetriever(
                    retrievers=[csv_bm25, csv_multi_query],
                    weights=[0.4, 0.6]
                )

            print("[CSV] Ensemble retriever created")
        else:
            print(f"[CSV] ⚠️  File not found: {csv_path}")

        # ============================================================================
        # PDF Retriever: Full Text with Recursive Chunking
        # ============================================================================
        if os.path.exists(pdf_path):
            print(f"\n[PDF] Loading full text from: {pdf_path}")

            pdf_metadata = extract_pdf_metadata(pdf_path)

            loader = PyMuPDFLoader(pdf_path)
            pdf_pages = loader.load()
            print(f"[PDF] Loaded {len(pdf_pages)} pages from PDF")

            for i, page in enumerate(pdf_pages):
                page.metadata.update({
                    "book_title": pdf_metadata["title"],
                    "author": pdf_metadata["author"],
                    "source_type": "pdf_fulltext",
                    "has_full_text": True,
                    "page_number": i + 1,
                    "total_pages": len(pdf_pages),
                    "page": i + 1
                })

            # RECURSIVE CHARACTER SPLITTING
            print(f"[PDF] Using RecursiveCharacterTextSplitter (size={chunk_size}, overlap={chunk_overlap})")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )

            pdf_chunks = text_splitter.split_documents(pdf_pages)
            print(f"[PDF] Created {len(pdf_chunks)} chunks from {len(pdf_pages)} pages")

            # Build ensemble PDF retriever
            print(f"[PDF] Building ensemble retriever with {len(pdf_chunks)} chunks...")
            pdf_vectorstore = Qdrant.from_documents(
                documents=pdf_chunks,
                embedding=embedding_model,
                location=":memory:",
                collection_name="pdf_fulltext"
            )

            pdf_bm25 = BM25Retriever.from_documents(pdf_chunks)
            pdf_bm25.k = 10

            pdf_multi_query = MultiQueryRetriever.from_llm(
                retriever=pdf_vectorstore.as_retriever(search_kwargs={"k": 15}),
                llm=chat_model
            )

            if cohere_api_key:
                pdf_compression = ContextualCompressionRetriever(
                    base_retriever=pdf_multi_query,
                    base_compressor=CohereRerank(model="rerank-v3.5", cohere_api_key=cohere_api_key)
                )
                pdf_retriever = EnsembleRetriever(
                    retrievers=[pdf_bm25, pdf_multi_query, pdf_compression],
                    weights=[0.3, 0.3, 0.4]
                )
            else:
                pdf_retriever = EnsembleRetriever(
                    retrievers=[pdf_bm25, pdf_multi_query],
                    weights=[0.5, 0.5]
                )

            print(f"[PDF] Ensemble retriever created")
        else:
            print(f"[PDF] ⚠️  File not found: {pdf_path}")

        print("\n" + "="*70)
        print("Advanced Recursive Retrieval System Initialized")
        print(f"   CSV Retriever: {'Ready' if csv_retriever else 'Not Available'}")
        print(f"   PDF Retriever: {'Ready' if pdf_retriever else 'Not Available'}")
        print("="*70 + "\n")

        return csv_retriever, pdf_retriever

    except Exception as e:
        print(f"[ERROR] Failed to initialize advanced recursive retrievers: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    """Test the advanced recursive retriever system"""
    from dotenv import load_dotenv
    load_dotenv()

    print("Testing Advanced Recursive Retrieval System...\n")

    csv_ret, pdf_ret = initialize_advanced_recursive_retrievers()

    if pdf_ret:
        print("\n[TEST] PDF Retriever:")
        results = pdf_ret.get_relevant_documents("What is the main character's name?")
        for i, doc in enumerate(results[:3], 1):
            page = doc.metadata.get('page_number', '?')
            preview = doc.page_content[:100].replace('\n', ' ')
            print(f"{i}. Page {page}: {preview}...")
