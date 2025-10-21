"""
Bookstore Customer Support API
A clean, dependency-safe FastAPI implementation with AI chat capabilities.
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import uvicorn
import os
import json

from dotenv import load_dotenv
load_dotenv()


AGENT_AVAILABLE = False
Configuration = None
AgentState = None
build_book_agent = None

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
    import pandas as pd
    RETRIEVAL_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] Advanced retrieval not available: {e}")
    RETRIEVAL_AVAILABLE = False

try:
    from book_research.configuration import Configuration
    from book_research.state import AgentState
    from book_research.book_agent import build_book_agent
    from book_research.retrievers.advanced_recursive_retriever import initialize_advanced_recursive_retrievers
    AGENT_AVAILABLE = True
    print("[OK] Book agent components loaded successfully")
except ImportError as e:
    print(f"[WARNING] Book agent components not available: {e}")
    print("[INFO] Running in basic mode with smart AI responses")

    class Configuration:
        def __init__(self):
            self.chat_model = "gpt-4o-mini"
            self.max_tokens = 1000
            self.temperature = 0.1
            self.allow_clarification = True
            self.max_web_search_results = 3

    class AgentState:
        pass

    def build_book_agent(*args, **kwargs):
        return None



class Book(BaseModel):
    """Book model for our inventory database."""
    id: Optional[int] = None
    title: str
    author: str
    isbn: str
    genre: str
    price: float
    stock: int = 1
    description: Optional[str] = None
    add_date: Optional[datetime] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class BooksResponse(BaseModel):
    """Response model for book listings."""
    message: str
    books: List[Book]

class BookUpdate(BaseModel):
    """Model for partial book updates."""
    title: Optional[str] = None
    author: Optional[str] = None
    isbn: Optional[str] = None
    genre: Optional[str] = None
    price: Optional[float] = None
    stock: Optional[int] = None
    description: Optional[str] = None

class AIAgentRequest(BaseModel):
    """Request model for AI agent chat."""
    query: str
    request_type: Optional[str] = None  

class AIAgentResponse(BaseModel):
    """Response model for AI agent chat."""
    response: str
    search_results: Optional[Dict[str, Any]] = None
    final_response: str

class BookstoreStats(BaseModel):
    """Statistics model for bookstore analytics."""
    total_books: int
    total_genres: int
    genres: List[str]
    total_inventory_value: float
    average_price: float


app = FastAPI(
    title="BookStore API",
    description="A cozy bookstore management system with elegant endpoints for book lovers",
    version="1.0.0",
    docs_url="/library",
    redoc_url="/archive",
    openapi_tags=[
        {"name": "Welcome", "description": "Welcome and health check endpoints"},
        {"name": "Inventory", "description": "Book inventory management"},
        {"name": "Management", "description": "Book collection management"},
        {"name": "Analytics", "description": "Bookstore analytics and statistics"},
        {"name": "AI Agent", "description": "AI-powered book recommendation chat"},
    ]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# DATA STORAGE
# ============================================================================

# In-memory bookstore inventory - will be populated from CSV on startup
bookstore_inventory: List[Book] = []

# Global agent state
book_agent = None
csv_retriever = None
pdf_retriever = None


# ============================================================================
# CSV DATA LOADING
# ============================================================================

def load_csv_books_into_inventory():
    """Load books from CSV file into bookstore_inventory."""
    global bookstore_inventory

    csv_path = "book_research/data/space_exploration_books.csv"

    if not os.path.exists(csv_path):
        print(f"[WARNING] CSV file not found: {csv_path}")
        print("[INFO] Using fallback inventory")

        # Try to add PDF book to fallback inventory
        pdf_path = "book_research/data/thief_of_sorrows.pdf"
        if os.path.exists(pdf_path):
            pdf_metadata = extract_pdf_metadata(pdf_path)
            bookstore_inventory = [
                Book(
                    id=1,
                    title=pdf_metadata["title"],
                    author=pdf_metadata["author"],
                    isbn=f"978{abs(hash(pdf_metadata['title'])) % 10000000000:010d}",
                    genre=pdf_metadata.get("subject", "Fiction") or "Fiction",
                    price=16.99,
                    stock=5,
                    description=f"Full text available. {pdf_metadata.get('subject', '')}".strip(),
                    add_date=datetime.now()
                )
            ]
        else:
            # Ultimate fallback - empty inventory
            bookstore_inventory = []
            print("[WARNING] No CSV or PDF found. Starting with empty inventory.")
        return

    try:
        if not RETRIEVAL_AVAILABLE:
            print("[WARNING] Pandas not available, cannot load CSV")
            return

        import pandas as pd
        df = pd.read_csv(csv_path)
        print(f"[INFO] Loading {len(df)} books from CSV into inventory...")

        for index, row in df.iterrows():
            # Extract genre from subjects (take first subject as primary genre)
            subjects = str(row.get('subjects', 'General Fiction'))
            genre = subjects.split(';')[0].strip() if subjects else "General Fiction"

            # Generate ISBN from work_key if not present
            work_key = str(row.get('work_key', ''))
            isbn = f"978{abs(hash(work_key)) % 10000000000:010d}"

            # Set default price based on book type
            price = 14.99  # Default price
            stock = 10  # Default stock

            book = Book(
                id=index + 1,
                title=str(row.get('title', 'Unknown Title')),
                author=str(row.get('author', 'Unknown Author')),
                isbn=isbn,
                genre=genre,
                price=price,
                stock=stock,
                description=str(row.get('description', 'No description available.')),
                add_date=datetime.now()
            )
            bookstore_inventory.append(book)

        # Add PDF book to inventory (extract metadata from PDF)
        pdf_path = "book_research/data/thief_of_sorrows.pdf"
        if os.path.exists(pdf_path):
            pdf_metadata = extract_pdf_metadata(pdf_path)

            # Generate ISBN from PDF title
            pdf_isbn = f"978{abs(hash(pdf_metadata['title'])) % 10000000000:010d}"

            bookstore_inventory.append(
                Book(
                    id=len(bookstore_inventory) + 1,
                    title=pdf_metadata["title"],
                    author=pdf_metadata["author"],
                    isbn=pdf_isbn,
                    genre=pdf_metadata.get("subject", "Fiction") or "Fiction",
                    price=16.99,
                    stock=5,
                    description=f"Full text available. {pdf_metadata.get('subject', '')}".strip(),
                    add_date=datetime.now()
                )
            )
            print(f"[OK] Added PDF book '{pdf_metadata['title']}' to inventory")

        print(f"[OK] Loaded {len(bookstore_inventory)} books into inventory")

    except Exception as e:
        print(f"[ERROR] Failed to load CSV books: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# PDF METADATA EXTRACTION
# ============================================================================

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


# ============================================================================
# RETRIEVER INITIALIZATION
# ============================================================================

def initialize_retrievers():
    """
    Initialize CSV and PDF retrievers using Advanced Recursive Retrieval.

    RAGAS Performance: 0.6935 average (13.1% improvement over simple baseline)

    Configuration:
    - RecursiveCharacterTextSplitter (1000 chars, 200 overlap) - Fast chunking
    - Ensemble Retrieval: BM25 + Multi-Query + Cohere Rerank
    - Initialization: ~60 seconds (pre-loaded at startup to avoid request timeout)

    Performance Improvements vs Simple Retrieval:
    - Context Recall: +66.7% (0.50 → 0.83) - Retrieves much more relevant information
    - Context Precision: +113.3% (0.27 → 0.57) - Dramatically less noise in results
    - Overall Score: +13.1% (0.61 → 0.69) - Better quality answers

    Evaluation Source: evaluation/advanced_retrieval/advanced_recursive_retrieval_metrics.md
    """
    global csv_retriever, pdf_retriever

    if not RETRIEVAL_AVAILABLE or not AGENT_AVAILABLE:
        print("[WARNING] Retrieval dependencies not available")
        return None, None

    try:
        print("\n" + "="*70)
        print("Initializing ADVANCED Recursive Retrieval System")
        print("   RAGAS Score: 0.6935 (13.1% improvement over baseline)")
        print("   Configuration: Ensemble (BM25 + Multi-Query + Cohere Rerank)")
        print("="*70 + "\n")

        # Use advanced retriever from separate module
        csv_retriever, pdf_retriever = initialize_advanced_recursive_retrievers(
            csv_path="book_research/data/space_exploration_books.csv",
            pdf_path="book_research/data/thief_of_sorrows.pdf",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            cohere_api_key=os.getenv("COHERE_API_KEY"),
            chunk_size=1000,
            chunk_overlap=200
        )

        print("\n" + "="*70)
        print("Advanced Retrieval System Ready")
        print(f"   CSV Retriever: {'Loaded' if csv_retriever else 'Failed'}")
        print(f"   PDF Retriever: {'Loaded' if pdf_retriever else 'Failed'}")
        print("="*70 + "\n")

        return csv_retriever, pdf_retriever

    except Exception as e:
        print(f"[ERROR] Error initializing advanced retrievers: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def calculate_stats() -> Dict[str, Any]:
    """Calculate bookstore statistics."""
    total_books = len(bookstore_inventory)
    total_value = sum(book.price * book.stock for book in bookstore_inventory)
    genres = list(set(book.genre for book in bookstore_inventory))
    
    return {
        "total_books": total_books,
        "total_genres": len(genres),
        "genres": genres,
        "total_inventory_value": round(total_value, 2),
        "average_price": round(sum(book.price for book in bookstore_inventory) / total_books, 2) if total_books > 0 else 0
    }

async def get_intelligent_response(query: str) -> str:
    """Generate intelligent AI responses based on query analysis."""
    query_lower = query.lower()
    
    # Check for specific book mentions
    if any(word in query_lower for word in ["thief", "sorrows"]):
        book = next((b for b in bookstore_inventory if "thief" in b.title.lower()), None)
        if book:
            return f"""📖 **{book.title}** by {book.author}

{book.description}

**Genre:** {book.genre}
**Price:** ${book.price}
**Stock:** {book.stock} copies available

This dark fantasy masterpiece delivers everything you want in epic fantasy - complex characters, intricate world-building, and plot twists that keep you on the edge of your seat. Perfect for fans of morally complex characters and immersive storytelling!"""
    
    # Check for recommendation requests
    elif any(word in query_lower for word in ["recommend", "suggest", "similar", "like", "what should i read"]):
        stats = calculate_stats()
        return f"""📚 **Personalized Book Recommendations**

Based on your interests, here are some great options from our collection:

**🏛️ Classic Literature:**
- "The Great Gatsby" by F. Scott Fitzgerald - Jazz Age masterpiece
- "To Kill a Mockingbird" by Harper Lee - Powerful story of justice

**🌃 Dystopian Fiction:**
- "1984" by George Orwell - Visionary tale of totalitarian control

**🐉 Dark Fantasy:**
- "Thief of Sorrows" by Kristen Long - Complex characters and world-building

**📊 Our Collection:** {stats['total_books']} books across {stats['total_genres']} genres

What genre or mood are you interested in? I can provide more specific recommendations!"""
    
    # Check for statistics requests
    elif any(word in query_lower for word in ["stats", "statistics", "analytics", "how many books", "inventory"]):
        stats = calculate_stats()
        return f"""📊 **Bookstore Statistics**

📚 **Total Books:** {stats['total_books']}
📖 **Genres Available:** {stats['total_genres']}
💰 **Total Inventory Value:** ${stats['total_inventory_value']}
💵 **Average Book Price:** ${stats['average_price']}

**📋 Available Genres:**
{', '.join(f'• {genre}' for genre in stats['genres'])}

Our collection is carefully curated to provide diverse reading experiences for every type of reader!"""
    
    # Check for author searches
    elif any(word in query_lower for word in ["by ", "author"]):
        # Extract author name if possible
        author_books = []
        for book in bookstore_inventory:
            if any(word in query_lower for word in book.author.lower().split()):
                author_books.append(f"• **{book.title}** - ${book.price}")
        
        if author_books:
            return f"""✍️ **Books by Author:**

{chr(10).join(author_books[:5])}

Found these titles in our collection! Would you like more details about any specific book?"""
    
    # Help/greeting responses
    elif any(word in query_lower for word in ["help", "hello", "hi", "what can you do"]):
        return """🤖 **AI Bookstore Assistant**

I'm your personal book recommendation assistant! Here's what I can help you with:

**🔍 Search & Discover:**
- Find books by title, author, or genre
- Get personalized recommendations
- Learn about specific books

**📊 Store Information:**
- View bookstore statistics
- Check inventory details
- Explore available genres

**💬 Just ask me naturally:**
- "Tell me about Thief of Sorrows"
- "Recommend fantasy books"
- "Books by F. Scott Fitzgerald"
- "Show me statistics"

What would you like to explore today?"""
    
    # Default response
    else:
        return f"""🤔 I understand you're asking about: "{query}"

I can help you with book recommendations, find specific titles or authors, show bookstore statistics, or tell you about our collection. 

Try asking me:
- About a specific book title or author
- For recommendations in a particular genre
- About our bookstore statistics
- Or just say "help" for more guidance!

What would you like to know about our books?"""

async def initialize_ai_agent():
    """Initialize the AI agent with proper error handling."""
    global book_agent, csv_retriever, pdf_retriever

    if not AGENT_AVAILABLE:
        print("[WARNING] Agent not available - using fallback")
        return None

    # If agent was pre-initialized on startup, just return it
    if book_agent is not None:
        return book_agent

    # Otherwise, initialize it now (fallback for edge cases)
    if book_agent is None:
        try:
            print("[AGENT] Initializing AI agent (not pre-initialized)...")

            # Initialize retrievers if not already done
            if csv_retriever is None or pdf_retriever is None:
                csv_retriever, pdf_retriever = initialize_retrievers()

            if csv_retriever is not None and pdf_retriever is not None:
                # Build the agent with retrievers
                book_agent = build_book_agent(
                    csv_retriever=csv_retriever,
                    pdf_retriever=pdf_retriever
                )
                print("[OK] AI agent initialized successfully!")
            else:
                print("[WARNING] Retrievers not available - using fallback")
                book_agent = None

        except Exception as e:
            print(f"[WARNING] Agent initialization failed: {e}")
            import traceback
            traceback.print_exc()
            book_agent = None

    return book_agent


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", tags=["Welcome"], response_model=Dict[str, Any])
async def welcome():
    """Welcome to our cozy digital bookstore! 📚"""
    return {
        "message": "Welcome to the BookStore API! 📖",
        "description": "Your gateway to a world of books and literary treasures enhanced with AI-powered recommendations",
        "version": "1.0.0",
        "status": "operational",
        "ai_agent_available": AGENT_AVAILABLE,
        "endpoints": {
            "documentation": {
                "swagger_ui": "/library",
                "redoc": "/archive"
            },
            "inventory": {
                "books": "GET /books - Browse complete inventory",
                "book_by_id": "GET /books/{id} - Find specific book",
                "stats": "GET /stats - View analytics"
            },
            "management": {
                "add_book": "POST /books - Add new book",
                "update_book": "PUT /books/{id} - Update book details",
                "delete_book": "DELETE /books/{id} - Remove book"
            },
            "ai_features": {
                "agent_chat": "POST /ai-agent - Chat with AI assistant"
            }
        }
    }

@app.get("/health", tags=["Welcome"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/books", response_model=BooksResponse, tags=["Inventory"])
async def browse_inventory(
    genre: Optional[str] = None,
    author: Optional[str] = None,
    limit: Optional[int] = 100
):
    """Browse our curated collection with optional filtering."""
    filtered_books = bookstore_inventory
    
    if genre:
        filtered_books = [book for book in filtered_books 
                         if genre.lower() in book.genre.lower()]
    
    if author:
        filtered_books = [book for book in filtered_books 
                         if author.lower() in book.author.lower()]
    
    # Apply limit
    if limit:
        filtered_books = filtered_books[:limit]
    
    return BooksResponse(
        message=f"Found {len(filtered_books)} book(s) matching your criteria",
        books=filtered_books
    )

@app.get("/books/{book_id}", response_model=Book, tags=["Inventory"])
async def get_book_by_id(book_id: int):
    """Get a specific book by its ID."""
    book = next((book for book in bookstore_inventory if book.id == book_id), None)
    if not book:
        raise HTTPException(
            status_code=404,
            detail=f"Book with ID {book_id} not found in our collection"
        )
    return book

@app.post("/books", response_model=Dict[str, Any], tags=["Management"])
async def add_book(book_data: Book):
    """Add a new book to our collection."""
    # Check for duplicate ISBN
    if any(existing.isbn == book_data.isbn for existing in bookstore_inventory):
        raise HTTPException(
            status_code=400,
            detail="A book with this ISBN already exists"
        )
    
    # Generate new ID
    book_data.id = max([b.id for b in bookstore_inventory], default=0) + 1
    book_data.add_date = datetime.now()
    
    bookstore_inventory.append(book_data)
    
    return {
        "message": f"Successfully added '{book_data.title}' to our collection!",
        "book": book_data
    }

@app.put("/books/{book_id}", response_model=Dict[str, Any], tags=["Management"])
async def update_book(book_id: int, book_update: BookUpdate):
    """Update book information."""
    book_index = next((i for i, book in enumerate(bookstore_inventory) 
                      if book.id == book_id), None)
    
    if book_index is None:
        raise HTTPException(
            status_code=404,
            detail=f"Book with ID {book_id} not found"
        )
    
    book = bookstore_inventory[book_index]
    update_data = book_update.dict(exclude_unset=True)
    
    for field, value in update_data.items():
        setattr(book, field, value)
    
    return {
        "message": f"Successfully updated '{book.title}'",
        "book": book
    }



@app.post("/ai-agent", response_model=AIAgentResponse, tags=["AI Agent"])
async def chat_with_ai_agent(request: AIAgentRequest):
    """Chat with our AI-powered book recommendation assistant."""
    try:
        # Try to initialize full agent
        agent = await initialize_ai_agent()

        # If agent is available, use it
        if agent is not None:
            print(f"[PROCESS] Processing query with LangGraph agent: {request.query}")

            # Get API keys from environment
            openai_api_key = os.getenv("OPENAI_API_KEY")
            tavily_api_key = os.getenv("TAVILY_API_KEY")

            # Configure agent with faster settings
            config = {
                "configurable": {
                    "chat_model": "gpt-4o-mini",
                    "max_tokens": 800,  # Reduced for faster responses
                    "temperature": 0.5,
                    "allow_clarification": False,
                    "max_web_search_results": 2,  # Reduced web search results
                    "openai_api_key": openai_api_key,
                    "tavily_api_key": tavily_api_key,
                }
            }

            # Invoke the agent
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": request.query}]},
                config
            )

            # Extract the final response
            final_response = result.get("final_response", "")
            if not final_response:
                # Try to get from messages
                messages = result.get("messages", [])
                if messages:
                    last_msg = messages[-1]
                    final_response = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)

            # Extract search results info
            search_results = result.get("search_results", {})

            return AIAgentResponse(
                response="LangGraph agent processed your request successfully",
                search_results={
                    "mode": "langgraph_agent",
                    "query": request.query,
                    "agent_available": True,
                    "search_target": search_results.get("search_target", "unknown"),
                    "routing_reasoning": search_results.get("routing_reasoning", ""),
                    "timestamp": datetime.now().isoformat()
                },
                final_response=final_response
            )
        else:
            # Fallback to intelligent response
            print(f"[PROCESS] Processing query with fallback: {request.query}")
            final_response = await get_intelligent_response(request.query)

            return AIAgentResponse(
                response="AI assistant processed your request successfully (fallback mode)",
                search_results={
                    "mode": "intelligent_response_fallback",
                    "query": request.query,
                    "agent_available": False,
                    "timestamp": datetime.now().isoformat()
                },
                final_response=final_response
            )

    except Exception as e:
        print(f"Error in AI agent: {e}")
        import traceback
        traceback.print_exc()

        # Try fallback
        try:
            final_response = await get_intelligent_response(request.query)
            return AIAgentResponse(
                response="Processed with fallback due to error",
                search_results={
                    "mode": "fallback",
                    "query": request.query,
                    "agent_available": False,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                },
                final_response=final_response
            )
        except:
            return AIAgentResponse(
                response="Error processing request",
                search_results={
                    "mode": "fallback",
                    "query": request.query,
                    "agent_available": False,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                },
                final_response="I apologize, but I encountered a technical issue. Please try rephrasing your question or try again later!"
            )


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("[START] Starting BookStore API...")

    # Load books from CSV into inventory
    load_csv_books_into_inventory()

    print(f"[INFO] Loaded {len(bookstore_inventory)} books in inventory")
    print(f"[INFO] AI Agent available: {AGENT_AVAILABLE}")

    # Initialize retrievers on startup to avoid timeout on first request
    if AGENT_AVAILABLE:
        print("[STARTUP] Pre-initializing retrievers to avoid first-request timeout...")
        csv_retriever, pdf_retriever = initialize_retrievers()
        if csv_retriever and pdf_retriever:
            print("[STARTUP] Building agent with pre-initialized retrievers...")
            book_agent = build_book_agent(
                csv_retriever=csv_retriever,
                pdf_retriever=pdf_retriever
            )
            print("[OK] Agent ready! No timeout will occur on first request.")
        else:
            print("[WARNING] Retrievers failed to initialize")

    print("[INFO] Server starting at http://localhost:8000")
    print("[INFO] API documentation available at http://localhost:8000/library")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )