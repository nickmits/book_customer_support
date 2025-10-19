# Book Research System Setup Guide

This guide explains how to set up the complete book recommendation system with actual retrievers.

## Prerequisites

1. **OpenAI API Key**: For the language model
2. **Tavily API Key**: For web search (optional)
3. **Book Data**: CSV catalog and PDF files for "Thief of Sorrows"

## Required Data Setup

### 1. CSV Book Catalog
Create a CSV file with book metadata:
```csv
title,author,isbn,description,subjects
"The Great Gatsby","F. Scott Fitzgerald","9780743273565","A masterpiece...","Classic Literature"
"1984","George Orwell","9780451524935","A dystopian novel...","Dystopian Fiction"
```

### 2. PDF Content
Ensure you have the full text of "Thief of Sorrows" by Kristen Long in PDF format.

## Setup Steps

### 1. Initialize Vector Stores

```python
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import CSVLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Setup embeddings
embeddings = OpenAIEmbeddings()

# Load CSV catalog
csv_loader = CSVLoader("path/to/book_catalog.csv")
csv_docs = csv_loader.load()
csv_text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
csv_docs_split = csv_text_splitter.split_documents(csv_docs)

# Create CSV vector store
csv_vectorstore = Chroma.from_documents(
    documents=csv_docs_split,
    embedding=embeddings,
    collection_name="book_catalog"
)

# Load PDF content
pdf_loader = PyPDFLoader("path/to/thief_of_sorrows.pdf")
pdf_docs = pdf_loader.load()
pdf_text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
pdf_docs_split = pdf_text_splitter.split_documents(pdf_docs)

# Create PDF vector store
pdf_vectorstore = Chroma.from_documents(
    documents=pdf_docs_split,
    embedding=embeddings,
    collection_name="thief_of_sorrows"
)
```

### 2. Create Ensemble Retrievers

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# Create retrievers
csv_retriever = csv_vectorstore.as_retriever(search_kwargs={"k": 10})
pdf_retriever = pdf_vectorstore.as_retriever(search_kwargs={"k": 5})

# Create BM25 retriever for hybrid search (optional)
bm25_retriever = BM25Retriever.from_documents(csv_docs_split + pdf_docs_split)
bm25_retriever.k = 5

# Create ensemble retriever
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, csv_retriever, pdf_retriever],
    weights=[0.2, 0.3, 0.5]
)
```

### 3. Integrate with FastAPI

Update your `main.py` to use actual retrievers:

```python
# In main.py, replace the mock implementation with:
async def initialize_book_agent():
    global book_agent
    
    if book_agent is None:
        # Initialize retrievers (as shown above)
        csv_retriever, pdf_retriever = setup_retrievers()
        
        # Build the actual agent
        from book_research.book_agent import build_book_agent
        book_agent = build_book_agent(csv_retriever, pdf_retriever)
    
    return book_agent

# Update the recommendation endpoint to use the real agent:
@app.post("/recommendations")
async def get_book_recommendation(request: BookRecommendationRequest, config: Configuration = Depends(get_configuration)):
    agent = await initialize_book_agent()
    
    # Create initial state
    initial_state = {
        "messages": [HumanMessage(content=request.query)],
        "search_iterations": 0,
        "post_revisions": 0
    }
    
    # Run the agent
    result = await agent.ainvoke(initial_state, config={
        "configurable": {
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            **config.model_dump()
        }
    })
    
    return BookRecommendationResponse(
        response=result.get("final_response", "No results found"),
        search_results=result.get("search_results", {}),
        final_response=result.get("final_response", "")
    )
```

## Environment Variables

Set these environment variables:

```bash
export OPENAI_API_KEY="your-openai-key"
export TAVILY_API_KEY="your-tavily-key"  # Optional
export CHAT_MODEL="gpt-4o-mini"
export MAX_TOKENS="1000"
export TEMPERATURE="0.1"
```

## Testing the Integration

Once set up, the system will provide:

1. **Intelligent Routing**: Automatically determines whether to search catalog, book content, or both
2. **Multi-Modal Search**: Can search different data sources based on query type
3. **Post Generation**: Creates formatted social media posts using actual book content
4. **Satisfaction Loops**: Iterates searches until finding satisfactory results

The current implementation provides mock responses that demonstrate the API structure and can be replaced with the actual agent implementation following this guide.
