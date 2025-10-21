"""
Simple PDF Retriever with Semantic Chunking
Uses SemanticChunker for more intelligent context-aware splitting
"""

from typing import List, Optional
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document
import os


class SimpleSemanticPDFRetriever:
    def __init__(
        self,
        pdf_path: str,
        openai_api_key: str,
        collection_name: str = "PDF_Documents_Semantic",
        breakpoint_threshold_type: str = "percentile",
        breakpoint_threshold_amount: int = 95
    ):
        self.pdf_path = pdf_path
        self.openai_api_key = openai_api_key
        self.collection_name = collection_name
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.breakpoint_threshold_amount = breakpoint_threshold_amount
        self.retriever = None

    def initialize(self, k: int = 10) -> 'SimpleSemanticPDFRetriever':
        """Load PDF, create vector store with semantic chunking, and initialize retriever."""
        print(f"[INIT] Loading PDF: {self.pdf_path}")
        print(f"[INIT] Using SemanticChunker with {self.breakpoint_threshold_type}={self.breakpoint_threshold_amount}")

        loader = PyMuPDFLoader(self.pdf_path)
        pages = loader.load()

        for i, page in enumerate(pages):
            page.metadata.update({
                "page_number": i + 1,
                "book_title": "Thief of Sorrows",
                "source_type": "pdf_fulltext"
            })

        # Use SemanticChunker for intelligent splitting
        pdf_embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=self.openai_api_key
        )

        text_splitter = SemanticChunker(
            pdf_embeddings,
            breakpoint_threshold_type=self.breakpoint_threshold_type,
            breakpoint_threshold_amount=self.breakpoint_threshold_amount
        )

        pdf_docs = text_splitter.split_documents(pages)

        print(f"[INIT] Created {len(pdf_docs)} semantic chunks from {len(pages)} pages")

        pdf_vectorstore = Qdrant.from_documents(
            pdf_docs,
            pdf_embeddings,
            location=":memory:",
            collection_name=self.collection_name
        )

        self.retriever = pdf_vectorstore.as_retriever(search_kwargs={"k": k})

        print(f"[OK] Semantic PDF retriever ready!")
        return self

    def search(self, query: str, k: int = 10) -> List[Document]:
        """Search for relevant documents."""
        if not self.retriever:
            raise ValueError("Call initialize() first")
        return self.retriever.get_relevant_documents(query)[:k]

    def format_results(self, results: List[Document]) -> str:
        """Format results for display."""
        output = []
        for i, doc in enumerate(results, 1):
            page = doc.metadata.get("page_number", "?")
            content = doc.page_content[:200] + "..."
            output.append(f"{i}. Page {page}: {content}")
        return "\n".join(output)


def create_simple_semantic_pdf_retriever(
    pdf_path: str,
    k: int = 10,
    breakpoint_threshold_type: str = "percentile",
    breakpoint_threshold_amount: int = 95
):
    """Factory function - one line usage for semantic retriever."""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("Set OPENAI_API_KEY environment variable")

    return SimpleSemanticPDFRetriever(
        pdf_path,
        openai_api_key,
        breakpoint_threshold_type=breakpoint_threshold_type,
        breakpoint_threshold_amount=breakpoint_threshold_amount
    ).initialize(k)


if __name__ == "__main__":
    retriever = create_simple_semantic_pdf_retriever(
        "book_research/data/thief_of_sorrows.pdf",
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95
    )
    results = retriever.search("What happens in chapter 2?", k=3)
    print(retriever.format_results(results))
