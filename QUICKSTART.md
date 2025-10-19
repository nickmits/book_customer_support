# Quick Start Guide - Book Recommendation API

## Overview

Your FastAPI application now integrates the complete book recommendation agent from your Jupyter notebook! Here's how everything works together:

## Architecture

```
Frontend (React)
    ↓
FastAPI (main.py)
    ↓
Book Agent (book_agent.py) with Intelligent Routing
    ↓
├── CSV Retriever (catalog search)
├── PDF Retriever (Thief of Sorrows full-text search)
└── Combined Search (both sources)
```

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Environment Variables

Create a `.env` file in the project root with your API keys:

```env
OPENAI_API_KEY=your_openai_key_here
TAVILY_API_KEY=your_tavily_key_here
COHERE_API_KEY=your_cohere_key_here
```

### 3. Verify Data Files

Make sure these files exist:
- `book_research/data/space_exploration_books.csv`
- `book_research/data/thief_of_sorrows.pdf`

### 4. Start the API Server

```bash
python main.py
```

The server will start at `http://localhost:8000`

You should see output like:
```
Setting up retrievers...
✅ Models created
📖 Loaded 120 pages from PDF
✅ Created 8 PDF chunks from 10 pages
📚 Loaded 5 books from CSV
✅ Prepared 5 CSV documents
🔧 Building CSV Retriever...
✅ CSV Retriever created!
🔧 Building PDF Retriever...
✅ PDF Retriever created!
```

## How to Use the API

### Option 1: Interactive API Documentation

Open your browser to:
- `http://localhost:8000/library` - Swagger UI
- `http://localhost:8000/archive` - ReDoc UI

### Option 2: Test Script

Run the automated test script:

```bash
python test_api.py
```

This will test:
1. Basic endpoints (books, stats)
2. Recommendation queries
3. Social media post generation

### Option 3: Direct API Calls

#### Get Book Recommendations

```bash
curl -X POST "http://localhost:8000/recommendations" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What books about space exploration do you have?",
    "request_type": "recommendation"
  }'
```

#### Ask About Thief of Sorrows

```bash
curl -X POST "http://localhost:8000/recommendations" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Tell me about the plot of Thief of Sorrows",
    "request_type": "specific_book"
  }'
```

#### Generate Social Media Post

```bash
curl -X POST "http://localhost:8000/social-post" \
  -H "Content-Type: application/json" \
  -d '{
    "platform": "general",
    "include_hashtags": true
  }'
```

### Option 4: Use Your React Frontend

Your frontend is already set up! Just start it:

```bash
cd frontend
npm install
npm run dev
```

Then open `http://localhost:5173` and interact with the chatbot UI.

## How the Routing Works

The agent automatically routes queries to the right data source:

| Query Type | Route | Example |
|------------|-------|---------|
| Catalog search | CSV | "What space books do you have?" |
| Book content | PDF | "What's the plot of Thief of Sorrows?" |
| Similar books | Both | "Books like Thief of Sorrows" |
| Social post | Post Writer | "Write a post about Thief of Sorrows" |

## API Endpoints

### Book Management
- `GET /` - Welcome message
- `GET /books` - List all books
- `GET /books/{id}` - Get specific book
- `POST /books` - Add new book
- `PUT /books/{id}` - Update book
- `DELETE /books/{id}` - Remove book
- `GET /stats` - Get inventory statistics

### AI Features
- `POST /recommendations` - Get AI-powered book recommendations
- `POST /social-post` - Generate social media posts
- `GET /recommendations/config` - View AI configuration

## Troubleshooting

### "Retrievers not set up" Warning

This means the PDF or CSV files couldn't be loaded. Check:
1. Files exist at `book_research/data/`
2. You have the required dependencies installed
3. Check the console for detailed error messages

### API Keys Missing

Make sure all three API keys are set in your `.env` file:
- OPENAI_API_KEY (required)
- TAVILY_API_KEY (optional, for web search fallback)
- COHERE_API_KEY (required, for reranking)

### Slow First Request

The first request after starting the server will be slow because:
1. Loading and chunking PDF pages
2. Creating vector embeddings
3. Building BM25 indexes
4. Initializing the agent graph

Subsequent requests will be much faster!

## What's Different from the Notebook?

The main.py integration:
1. Uses the **exact same** retriever setup as your notebook
2. Wraps it in FastAPI endpoints for web access
3. Adds a fallback mock response if retrievers fail
4. Caches the agent globally (initialized once per server start)
5. Works with your React frontend

## Next Steps

1. **Test the API**: Run `python test_api.py`
2. **Try the Frontend**: Start the React app and chat with the bot
3. **Customize**: Modify routing rules in `book_agent.py:100-140`
4. **Expand Data**: Add more PDFs or CSV books to `book_research/data/`
5. **Deploy**: Use Docker or a cloud platform (see deployment guides)

## Development Tips

### Adding More Books to CSV Catalog

Edit `book_research/data/space_exploration_books.csv` and add rows with:
- title
- author
- work_key
- description
- subjects

### Adding More PDFs

1. Put PDFs in `book_research/data/`
2. Update `setup_retrievers()` in main.py to load multiple PDFs
3. Update metadata in the agent

### Changing the AI Model

Edit `.env` or configuration:
```python
CHAT_MODEL=gpt-4  # or gpt-3.5-turbo, etc.
```

### Debugging

The agent prints detailed routing information:
```
🧭 Routing: The query asks for catalog → csv_metadata
📚 CSV Search: Found 3 books in catalog
✅ Search complete: Results sufficient
```

Watch the console to see what's happening!

## Support

If you encounter issues:
1. Check the console output for detailed error messages
2. Verify all dependencies are installed
3. Ensure API keys are valid
4. Test with the simple test script first
5. Check that data files exist and are readable

Enjoy your AI-powered book recommendation system! 📚✨
