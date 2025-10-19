# Testing Guide - LangGraph Integration

## Overview

This guide shows you **4 different ways** to test the integrated LangGraph agent in your FastAPI application.

---

## Prerequisites

Before testing, ensure:

1. **Dependencies are installed**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment variables are set** (already in `.env`):
   - `OPENAI_API_KEY` ✓
   - `TAVILY_API_KEY` ✓ (optional)
   - `COHERE_API_KEY` ✓ (optional)

3. **Data files exist**:
   - `book_research/data/space_exploration_books.csv` ✓
   - `book_research/data/thief_of_sorrows.pdf` ✓

---

## Method 1: Quick Start - Run the Test Script 🚀

**Easiest way to test everything at once!**

### Step 1: Start the Server
In **Terminal 1**:
```bash
cd C:\Users\nickm\Documents\book_customer_support
python main.py
```

You should see:
```
[OK] Book agent components loaded successfully
🚀 Starting BookStore API...
📚 Loaded 4 books in inventory
🤖 AI Agent available: True
🌐 Server starting at http://localhost:8000
```

### Step 2: Run the Tests
In **Terminal 2** (keep server running):
```bash
cd C:\Users\nickm\Documents\book_customer_support
python test_api.py
```

This will test:
- ✅ Welcome endpoint
- ✅ Books inventory
- ✅ Catalog query (CSV routing)
- ✅ PDF content query (PDF routing)
- ✅ Social media post generation
- ✅ Combined queries

---

## Method 2: Interactive API Documentation (Swagger UI) 📚

**Visual, browser-based testing!**

### Step 1: Start the Server
```bash
python main.py
```

### Step 2: Open Swagger UI
Open your browser and go to:
```
http://localhost:8000/library
```

### Step 3: Test the `/ai-agent` Endpoint

1. Find **"AI Agent"** section
2. Click on **POST /ai-agent**
3. Click **"Try it out"**
4. Enter test data:
   ```json
   {
     "query": "Tell me about Thief of Sorrows",
     "request_type": "specific_book"
   }
   ```
5. Click **"Execute"**
6. See the response below!

**More test queries to try**:
```json
# Catalog search
{
  "query": "What books about space do you have?",
  "request_type": "recommendation"
}

# Social media post
{
  "query": "Write a post about Thief of Sorrows",
  "request_type": "recommendation"
}

# Similar books
{
  "query": "Books similar to Thief of Sorrows",
  "request_type": "recommendation"
}
```

---

## Method 3: Command Line with curl 💻

**Quick testing from terminal!**

### Step 1: Start the Server
```bash
python main.py
```

### Step 2: Test with curl
In a new terminal:

**Test 1: PDF Content Query**
```bash
curl -X POST http://localhost:8000/ai-agent ^
  -H "Content-Type: application/json" ^
  -d "{\"query\": \"Tell me about Thief of Sorrows\", \"request_type\": \"specific_book\"}"
```

**Test 2: Catalog Search**
```bash
curl -X POST http://localhost:8000/ai-agent ^
  -H "Content-Type: application/json" ^
  -d "{\"query\": \"What books about space do you have?\", \"request_type\": \"recommendation\"}"
```

**Test 3: Social Media Post**
```bash
curl -X POST http://localhost:8000/ai-agent ^
  -H "Content-Type: application/json" ^
  -d "{\"query\": \"Write a post about Thief of Sorrows\", \"request_type\": \"recommendation\"}"
```

> **Note**: On Linux/Mac, use single quotes and `\` instead of `^`:
> ```bash
> curl -X POST http://localhost:8000/ai-agent \
>   -H "Content-Type: application/json" \
>   -d '{"query": "Tell me about Thief of Sorrows"}'
> ```

---

## Method 4: Python Script (Custom Testing) 🐍

**For programmatic testing!**

Create a file `my_test.py`:

```python
import asyncio
import httpx

async def test_agent():
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "http://localhost:8000/ai-agent",
            json={
                "query": "Tell me about Thief of Sorrows",
                "request_type": "specific_book"
            }
        )

        result = response.json()
        print(f"Status: {response.status_code}")
        print(f"Mode: {result['search_results']['mode']}")
        print(f"Route: {result['search_results']['search_target']}")
        print(f"\nResponse:\n{result['final_response']}")

if __name__ == "__main__":
    asyncio.run(test_agent())
```

Run it:
```bash
python my_test.py
```

---

## Understanding the Response

The `/ai-agent` endpoint returns a JSON response like this:

```json
{
  "response": "LangGraph agent processed your request successfully",
  "search_results": {
    "mode": "langgraph_agent",
    "query": "Tell me about Thief of Sorrows",
    "agent_available": true,
    "search_target": "pdf_fulltext",
    "routing_reasoning": "Query asks about plot...",
    "timestamp": "2025-10-18T..."
  },
  "final_response": "Thief of Sorrows by Kristen Long is a dark fantasy..."
}
```

**Key fields**:
- `mode`: How the query was processed
  - `langgraph_agent`: Full LangGraph agent used ✓
  - `intelligent_response_fallback`: Fallback mode
  - `fallback`: Error occurred

- `search_target`: Which data source was used
  - `csv_metadata`: Book catalog
  - `pdf_fulltext`: Thief of Sorrows content
  - `both`: Combined search
  - `write_post`: Social media post

- `final_response`: The actual response text to show users

---

## Expected Routing Behavior

| Query Type | Example | Expected Route |
|------------|---------|----------------|
| Catalog search | "What books do you have?" | `csv_metadata` |
| Book info | "Tell me about Thief of Sorrows" | `pdf_fulltext` |
| Similar books | "Books like Thief of Sorrows" | `both` |
| Social post | "Write a post about Thief of Sorrows" | `write_post` |

---

## Troubleshooting

### Problem: Server won't start

**Check for import errors**:
```bash
python main.py
```

If you see:
```
[WARNING] Advanced retrieval not available: No module named 'langchain_openai'
```

**Fix**: Install missing packages:
```bash
pip install langchain-openai langchain-cohere qdrant-client pymupdf pandas
```

---

### Problem: Agent returns fallback responses

**Check server logs** for:
```
[WARNING] Retrieval dependencies not available
```

**Fix**: Ensure all dependencies are installed:
```bash
pip install -r requirements.txt --upgrade
```

---

### Problem: "No module named 'dotenv'"

**Fix**:
```bash
pip install python-dotenv
```

---

### Problem: OPENAI_API_KEY not found

**Check** if `.env` file is loaded:
```python
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print(os.getenv('OPENAI_API_KEY'))"
```

Should print your API key. If it prints `None`:
1. Verify `.env` file exists
2. Check format: `OPENAI_API_KEY=sk-...` (no quotes, no export)

---

### Problem: Test timeout

If tests timeout:
1. First request is slow (initializing retrievers) - this is normal!
2. Increase timeout in test script (already set to 90s)
3. Check if OpenAI API is responding

---

## What to Expect on First Run

### First Request (Slow - 30-60 seconds):
```
🔧 Initializing retrievers...
📚 Loaded 5 books from CSV
📖 Loaded 250 pages from PDF
🔧 Building CSV retriever...
✅ CSV retriever created
🔧 Building PDF retriever...
✅ PDF retriever created
✅ Advanced AI agent initialized successfully!
```

### Subsequent Requests (Fast - 2-5 seconds):
```
📤 Processing query with LangGraph agent: Tell me about...
```

---

## Quick Verification Checklist

After starting the server, verify:

- [ ] Server starts without errors
- [ ] You see "Book agent components loaded successfully"
- [ ] Opening http://localhost:8000 shows welcome message
- [ ] Opening http://localhost:8000/library shows API docs
- [ ] `/ai-agent` endpoint appears in "AI Agent" section
- [ ] First test query takes ~60s (retriever initialization)
- [ ] Subsequent queries take ~5s
- [ ] Responses include routing information
- [ ] Different query types route to different sources

---

## Performance Benchmarks

Expected timings:
- **First request**: 30-60 seconds (one-time initialization)
- **Catalog queries**: 3-8 seconds
- **PDF queries**: 5-10 seconds
- **Post generation**: 8-15 seconds
- **Combined queries**: 10-20 seconds

---

## Next Steps

Once testing works:

1. **Test with your frontend**:
   - Update `frontend/src/services/api.ts`
   - Change endpoint from `/chat` to `/ai-agent`

2. **Deploy to production**:
   - Use persistent vector storage (not in-memory)
   - Load full PDF (not just 10 pages)
   - Add caching for retrievers
   - Add rate limiting

3. **Monitor performance**:
   - Add logging for query routing
   - Track response times
   - Monitor API costs

---

## Need Help?

If tests fail:
1. Check server logs for detailed errors
2. Verify all dependencies are installed
3. Check `.env` file format
4. Ensure data files exist
5. Try the Swagger UI for visual testing

For dependency issues:
```bash
pip install --upgrade langchain langchain-core langchain-community langgraph langchain-openai
```
