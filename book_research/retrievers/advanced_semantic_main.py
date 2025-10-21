"""
Semantic Retrieval System
Same as main.py but uses SemanticChunker instead of RecursiveCharacterTextSplitter
"""

from dotenv import load_dotenv
import os

load_dotenv()

# Global retriever state
csv_retriever = None
pdf_retriever = None

RETRIEVAL_AVAILABLE = False
try:
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import Qdrant
    from langchain.retrievers import EnsembleRetriever
    from langchain.retrievers.multi_query import MultiQueryRetriever
    from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
    from langchain_cohere import CohereRerank
    from langchain_community.retrievers import BM25Retriever
    from langchain_core.documents import Document
    from langchain.chat_models import init_chat_model
    from langchain_community.document_loaders import PyMuPDFLoader
    from langchain_experimental.text_splitter import SemanticChunker
    import pandas as pd
    RETRIEVAL_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] Advanced retrieval not available: {e}")
    RETRIEVAL_AVAILABLE = False


# SEMANTIC CHUNKING PARAMETERS
BREAKPOINT_THRESHOLD_TYPE = "percentile"
BREAKPOINT_THRESHOLD_AMOUNT = 95


def extract_pdf_metadata(pdf_path: str) -> dict:
    """Extract metadata from PDF file."""
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(pdf_path)
        metadata = doc.metadata

        # Extract common metadata fields
        pdf_metadata = {
            "title": metadata.get("title", "Unknown Title"),
            "author": metadata.get("author", "Unknown Author"),
            "subject": metadata.get("subject", ""),
            "keywords": metadata.get("keywords", ""),
            "creator": metadata.get("creator", ""),
            "producer": metadata.get("producer", ""),
            "creation_date": metadata.get("creationDate", ""),
            "modification_date": metadata.get("modDate", ""),
        }

        doc.close()

        print(f"[INFO] Extracted PDF metadata:")
        print(f"  Title: {pdf_metadata['title']}")
        print(f"  Author: {pdf_metadata['author']}")
        print(f"  Subject: {pdf_metadata['subject']}")

        return pdf_metadata

    except Exception as e:
        print(f"[WARNING] Could not extract PDF metadata: {e}")
        return {
            "title": "Unknown Title",
            "author": "Unknown Author",
            "subject": "",
            "keywords": "",
        }


def initialize_semantic_retrievers():
    """Initialize CSV and PDF retrievers using SemanticChunker for the PDF."""
    global csv_retriever, pdf_retriever

    if not RETRIEVAL_AVAILABLE:
        print("[WARNING] Retrieval dependencies not available")
        return None, None

    try:
        # Get API keys from environment
        openai_api_key = os.getenv("OPENAI_API_KEY")
        cohere_api_key = os.getenv("COHERE_API_KEY")

        if not openai_api_key:
            print("[ERROR] OPENAI_API_KEY not found in environment")
            return None, None

        # Initialize models
        chat_model = init_chat_model(
            model="openai:gpt-4o-mini",
            api_key=openai_api_key,
            temperature=0.1,
            max_tokens=1000
        )

        embedding_model = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=openai_api_key
        )

        # ============================================================================
        # Load CSV Data (Book Catalog)
        # ============================================================================
        csv_path = "book_research/data/space_exploration_books.csv"
        if not os.path.exists(csv_path):
            print(f"[WARNING] CSV file not found: {csv_path}")
            csv_documents = []
        else:
            df = pd.read_csv(csv_path)
            print(f"[INFO] Loaded {len(df)} books from CSV")

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

        # ============================================================================
        # Load PDF Data with SEMANTIC CHUNKING
        # ============================================================================
        pdf_path = "book_research/data/thief_of_sorrows.pdf"
        if not os.path.exists(pdf_path):
            print(f"[WARNING] PDF file not found: {pdf_path}")
            pdf_chunks = []
            pdf_book_metadata = {"title": "Unknown", "author": "Unknown"}
        else:
            # Extract metadata from PDF
            pdf_book_metadata = extract_pdf_metadata(pdf_path)

            loader = PyMuPDFLoader(pdf_path)
            pdf_pages = loader.load()
            print(f"[INFO] Loaded {len(pdf_pages)} pages from PDF")

            # Add metadata using extracted info
            for i, page in enumerate(pdf_pages):
                page.metadata.update({
                    "book_title": pdf_book_metadata["title"],
                    "author": pdf_book_metadata["author"],
                    "source_type": "pdf_fulltext",
                    "has_full_text": True,
                    "page_number": i + 1,
                    "total_pages": len(pdf_pages),
                    "page": i + 1  # Add for compatibility
                })

            # SEMANTIC CHUNKING: Intelligent chunking based on semantic similarity
            print(f"[INFO] Using SemanticChunker with {BREAKPOINT_THRESHOLD_TYPE}={BREAKPOINT_THRESHOLD_AMOUNT}")
            text_splitter = SemanticChunker(
                embedding_model,
                breakpoint_threshold_type=BREAKPOINT_THRESHOLD_TYPE,
                breakpoint_threshold_amount=BREAKPOINT_THRESHOLD_AMOUNT
            )

            # Split ALL pages using semantic chunking
            pdf_chunks = text_splitter.split_documents(pdf_pages)
            print(f"[INFO] Created {len(pdf_chunks)} semantic chunks from {len(pdf_pages)} pages")


        # ============================================================================
        # Create CSV Retriever
        # ============================================================================
        if csv_documents:
            print("[INIT] Building CSV retriever...")
            csv_vectorstore = Qdrant.from_documents(
                documents=csv_documents,
                embedding=embedding_model,
                location=":memory:",
                collection_name="csv_catalog_semantic"
            )

            csv_bm25 = BM25Retriever.from_documents(csv_documents)
            csv_bm25.k = 5

            csv_multi_query = MultiQueryRetriever.from_llm(
                retriever=csv_vectorstore.as_retriever(search_kwargs={"k": 10}),
                llm=chat_model
            )

            # Cohere reranking (optional if API key available)
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

            print("[OK] CSV retriever created")
        else:
            print("[WARNING] No CSV documents loaded")
            csv_retriever = None

        # ============================================================================
        # Create PDF Retriever with semantic chunks
        # ============================================================================
        if pdf_chunks:
            print(f"[INIT] Building PDF retriever with {len(pdf_chunks)} semantic chunks...")
            pdf_vectorstore = Qdrant.from_documents(
                documents=pdf_chunks,
                embedding=embedding_model,
                location=":memory:",
                collection_name="pdf_fulltext_semantic"
            )

            pdf_bm25 = BM25Retriever.from_documents(pdf_chunks)
            pdf_bm25.k = 10

            pdf_multi_query = MultiQueryRetriever.from_llm(
                retriever=pdf_vectorstore.as_retriever(search_kwargs={"k": 15}),
                llm=chat_model
            )

            # Cohere reranking (optional if API key available)
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

            print(f"[OK] PDF retriever created with {len(pdf_chunks)} semantic chunks")
        else:
            print("[WARNING] No PDF documents loaded")
            pdf_retriever = None

        return csv_retriever, pdf_retriever

    except Exception as e:
        print(f"[ERROR] Error initializing retrievers: {e}")
        import traceback
        traceback.print_exc()
        return None, None


# For direct execution testing
if __name__ == "__main__":
    print("=" * 70)
    print("SEMANTIC RETRIEVAL SYSTEM - Initialization Test")
    print("=" * 70)

    csv_ret, pdf_ret = initialize_semantic_retrievers()

    if pdf_ret:
        print("\n[Testing] PDF Retriever...")
        test_query = "What is the main character's name?"
        results = pdf_ret.get_relevant_documents(test_query)
        print(f"Query: {test_query}")
        print(f"Retrieved {len(results)} documents")
        if results:
            print(f"First result preview: {results[0].page_content[:200]}...")

    print("\n[Done] Semantic retrieval system initialized successfully!")
