"""
Test script for the Simple PDF Retriever
Demonstrates how to use RecursiveCharacterTextSplitter and vector search
"""

from book_research.simple_pdf_retriever import create_simple_pdf_retriever
import os

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


def main():
    """Test the simple PDF retriever with various queries."""

    print("="*70)
    print("SIMPLE PDF RETRIEVER TEST")
    print("="*70)

    # Path to PDF
    pdf_path = "book_research/data/thief_of_sorrows.pdf"

    if not os.path.exists(pdf_path):
        print(f"❌ Error: PDF file not found at {pdf_path}")
        return

    # Create retriever with RecursiveCharacterTextSplitter
    print("\n[INIT] Creating PDF retriever...")
    print(f"  - PDF: {pdf_path}")
    print(f"  - Chunk size: 1000 characters")
    print(f"  - Chunk overlap: 200 characters")
    print(f"  - Retrieval: Top 5 results with reranking")
    print()

    retriever = create_simple_pdf_retriever(
        pdf_path=pdf_path,
        chunk_size=1000,      # 1000 character chunks
        chunk_overlap=200,    # 200 character overlap
        k=5,                  # Retrieve top 5 results
        use_reranking=True    # Use Cohere reranking if available
    )

    # Test queries
    test_queries = [
        "What happens in chapter 2?",
        "Who are the main characters?",
        "Tell me about the plot and themes",
        "What is the setting of the story?"
    ]

    # Run each query
    for i, query in enumerate(test_queries, 1):
        print("\n" + "="*70)
        print(f"TEST QUERY {i}: {query}")
        print("="*70)

        # Search
        results = retriever.search(query, k=3)

        # Display results
        print(f"\nFound {len(results)} relevant passages:\n")
        print(retriever.format_results(results))

        # Separator
        print("\n" + "-"*70)

    print("\n[OK] Test completed!")
    print("\n" + "="*70)
    print("USAGE SUMMARY")
    print("="*70)
    print("""
The SimplePDFRetriever uses:

1. RecursiveCharacterTextSplitter:
   - Intelligently splits text at natural boundaries
   - Preserves context with overlapping chunks
   - Handles paragraphs, sentences, and words

2. Chroma Vector Store:
   - Fast in-memory vector database
   - Efficient similarity search
   - Easy to use and lightweight

3. OpenAI Embeddings:
   - text-embedding-3-small model
   - High-quality semantic search
   - Cost-effective

4. Optional Cohere Reranking:
   - Improves result relevance
   - Reranks top candidates
   - Better precision

To use in your code:

    from book_research.simple_pdf_retriever import create_simple_pdf_retriever

    # Create retriever
    retriever = create_simple_pdf_retriever("path/to/book.pdf")

    # Search
    results = retriever.search("your query here", k=5)

    # Display
    for result in results:
        print(result.page_content)
        print(f"Page: {result.metadata['page_number']}")
""")


if __name__ == "__main__":
    main()
