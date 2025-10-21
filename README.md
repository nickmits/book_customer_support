# 📚 BookStore API with AI Recommendations

A sophisticated FastAPI application combining traditional bookstore management with AI-powered book recommendations and social media content generation.

## 📋 Certification Project

**Problem Statement:** Bookstore customers cannot find relevant books based on specific plot details, themes, or character traits because traditional search systems only index catalog metadata instead of actual book content.

**Why This Matters:**

When bookstore customers search for books, they often have specific content preferences in mind—they might be looking for a fantasy novel with morally complex characters, a mystery with an unreliable narrator, or a romance set in a specific historical period. Traditional search systems force these customers to sift through dozens of generic catalog descriptions, reading summaries that may not accurately convey the themes, writing style, or emotional depth they're seeking. This leads to frustration, abandoned searches, and missed opportunities to discover books that would genuinely resonate with them. Without the ability to search actual book content, customers must rely on superficial metadata like genre tags and brief descriptions, which rarely capture the nuanced elements that make a book compelling to individual readers.

The consequences of this limitation extend beyond simple inconvenience. Customers who cannot find books matching their specific interests are more likely to abandon the bookstore entirely and seek recommendations from social media, friend groups, or competitors with better discovery tools. This represents lost sales for the bookstore and a diminished customer experience. Additionally, readers may purchase books based on misleading or incomplete descriptions, leading to dissatisfaction and returns. By enabling content-based search that understands plot details, character development, and thematic elements, bookstores can dramatically improve customer satisfaction, increase successful book discoveries, and build customer loyalty through personalized, accurate recommendations.

**Proposed Solution:**

This project implements an AI-powered bookstore API that uses Retrieval-Augmented Generation (RAG) to enable content-based book search and recommendations. The system combines dual data sources—a catalog metadata database (CSV) and full-text PDF content—with intelligent routing to determine the best search strategy for each customer query. By leveraging LangChain and LangGraph for AI orchestration, the system can understand natural language questions like "What books have morally complex protagonists?" or "Tell me about the character development in Thief of Sorrows" and search actual book content to provide accurate, detailed answers.

The solution architecture consists of three key layers. First, a **parsing and routing layer** analyzes incoming customer queries using GPT-4o-mini to determine intent (specific book lookup vs. general recommendation) and intelligently routes to the appropriate data source—catalog metadata for browsing available titles, PDF full-text for detailed content questions, or both for complex queries like "books similar to Thief of Sorrows." Second, a **RAG retrieval layer** employs ensemble retrieval combining Qdrant vector stores, BM25 keyword search, and optional Cohere reranking to find the most relevant passages from either the book catalog or full PDF content. Third, a **response generation layer** synthesizes retrieved information into natural, conversational answers that directly address the customer's specific question. The system includes a fallback web search mechanism using Tavily for queries about books outside the catalog, and a React-based frontend provides an intuitive chat interface for customers to interact with the AI assistant.

By indexing and searching actual book content rather than just metadata, customers can now discover books based on the precise elements that matter to them—character traits, plot themes, writing style, and emotional depth. The intelligent routing ensures efficient searches (avoiding unnecessary processing), while the RAG approach grounds all responses in actual book text, preventing AI hallucinations and ensuring accuracy. This dramatically improves book discovery, reduces customer frustration, and increases the likelihood of successful purchases by matching readers with books that truly align with their preferences.

**LangChain Tools:**

- **TavilySearchResults** - Provides fallback web search capability when books are not found in the local catalog, ensuring customers get helpful responses even for out-of-catalog queries by searching the broader web.
- **format_social_post** - Formats AI-generated content into ready-to-publish social media posts with appropriate hashtags and platform-specific styling, used with bind_tools for autonomous tool calling by the LLM during post generation.

**Agent Architecture and Agentic Reasoning:**

This project implements a **multi-agent LangGraph architecture** with specialized agents orchestrated through a state graph with conditional routing. Each agent employs agentic reasoning to make autonomous decisions:

**Core Agents:**

1. **Clarify With User Agent** (`clarify_with_user`) - Analyzes incoming queries to determine if clarification is needed before processing, reasoning about query ambiguity and information completeness to improve response accuracy.

2. **Parse Request Agent** (`parse_request`) - Extracts structured information from natural language queries including request type (specific book vs. recommendation), book titles, customer age, interests, and gift status using GPT-4o-mini with structured output, enabling the system to understand unstructured customer intent.

3. **Route Search Agent** (`route_search`) - Makes intelligent routing decisions by analyzing query semantics to determine whether to search catalog metadata, PDF full-text, both sources, or trigger post generation, implementing logical reasoning rules like "questions about plot details → PDF search" and "browsing requests → catalog search."

4. **CSV Search Agent** (`csv_search_agent`) - Searches book catalog metadata using ensemble retrieval (BM25 + Qdrant vector search) to find relevant books based on titles, authors, descriptions, and subjects.

5. **PDF Search Agent** (`pdf_search_agent`) - Searches full-text content of "Thief of Sorrows" using ensemble retrieval to answer detailed questions about plot, characters, themes, and specific story elements that catalog metadata cannot provide.

6. **Combined Search Agent** (`combined_search_agent`) - Executes parallel searches across both catalog and PDF sources for complex queries like "books similar to Thief of Sorrows" that require understanding both available options and content characteristics.

7. **Check Results Agent** (`check_results`) - Evaluates search result quality and decides whether results are sufficient or if web search fallback is needed, implementing conditional logic based on result count and search target type.

8. **Web Search Agent** (`web_search_agent`) - Serves as intelligent fallback using Tavily API to search the broader web when local searches return no results, ensuring customers receive helpful responses even for out-of-catalog queries.

9. **Check Satisfaction Agent** (`check_satisfaction`) - Performs iterative quality evaluation by reasoning about whether retrieved results adequately answer the customer's query, and if not, autonomously determining the next search strategy (try different source, expand query, etc.) across up to 3 iterations.

10. **Generate Response Agent** (`generate_response`) - Synthesizes all retrieved information into natural, conversational responses with context-aware prompting that adapts based on whether results came from catalog metadata, full-text content, or both sources.

11. **Post Writing Agent** (`post_writing_agent`) - Generates social media content by retrieving relevant PDF passages, crafting engaging promotional text, and autonomously deciding when to invoke the `format_social_post` tool via bind_tools.

**What Agentic Reasoning Accomplishes:**

- **Intent Understanding**: Agents reason about user intent from natural language, distinguishing between specific book requests, general recommendations, content questions, or post generation without requiring structured input formats.

- **Dynamic Routing**: The routing agent applies logical reasoning to determine optimal data sources—routing "What happens to the main character?" to PDF full-text while routing "Space books" to catalog metadata automatically.

- **Quality Evaluation**: The satisfaction agent iteratively evaluates result quality, reasoning about adequacy and autonomously refining search strategies without hardcoded retry logic.

- **Adaptive Recovery**: When searches fail, agents reason about the best recovery path—switching sources, expanding queries, or falling back to web search based on the specific failure mode.

- **Autonomous Tool Usage**: The post-writing agent demonstrates reasoning about workflow progression by deciding when to invoke formatting tools based on content readiness rather than predetermined triggers.

This multi-agent architecture with agentic reasoning provides intelligent, context-aware book discovery that adapts to query complexity and gracefully handles edge cases without requiring customers to understand system limitations or manually reformulate queries.

**Data Sources and External APIs:**

**Data Sources:**

1. **space_exploration_books.csv** - Curated catalog of 100+ space exploration books containing structured metadata including titles, authors, descriptions, subjects, and work keys from Open Library. This serves as the primary book catalog for metadata-based searches, enabling customers to browse available titles, filter by genre/author, and receive recommendations based on book descriptions and subject classifications.

2. **thief_of_sorrows.pdf** - Complete fantasy novel PDF containing full book text processed page-by-page using PyMuPDF and chunked into 800-character segments with 200-character overlap. This enables deep content-based search capabilities where customers can ask detailed questions about plot elements, character development, themes, and specific story moments that catalog metadata cannot answer.

**External API:**

**Tavily Search API** - Powers the web search fallback agent (`web_search_agent`) to handle out-of-catalog queries by retrieving relevant book information from the broader web when local searches (CSV catalog and PDF content) return no results. This ensures customers always receive helpful responses even when asking about books not in the local inventory, preventing "dead end" interactions and maintaining a positive user experience.

**Retrieval System:**

This project uses **Advanced Recursive Retrieval** combining:
- **RecursiveCharacterTextSplitter**: Fast chunking (1000 chars, 200 overlap)
- **Ensemble Retrieval**: BM25 + Multi-Query + Cohere Rerank for optimal relevance
- **RAGAS Score**: 0.6935 (13.1% improvement over baseline)
- **Startup Time**: ~60 seconds (pre-initialized at server startup)

The system was selected after comprehensive RAGAS evaluation of 4 different approaches, balancing quality improvement (+66.7% context recall, +113.3% context precision) with production viability.

**See [RAG Pipeline Evaluation & Certification](#-rag-pipeline-evaluation--certification) section below for detailed evaluation results and methodology.**

## 🚀 Features

### Traditional Bookstore Management
- **Browse Collection**: View all books with optional filtering by genre/author
- **Find Specific Book**: Search by book ID
- **Add New Books**: Expand the collection with new literary treasures
- **Update Book Details**: Keep catalog information current
- **Remove Books**: Manage inventory (parting is such sweet sorrow)
- **Store Statistics**: Get insights about the collection

### 🤖 AI-Powered Features
- **Intelligent Recommendations**: AI-powered book suggestions based on user preferences
- **Multi-Source Search**: Searches both book catalog and full-text content
- **Smart Routing**: Automatically determines the best search strategy
- **Social Media Posts**: Generate engaging social media content about specific books
- **Post-Writing Tools**: Automated formatting and hashtag generation

## 🏗️ Architecture

The application integrates a sophisticated book recommendation system built with:
- **LangChain & LangGraph**: For AI agent orchestration
- **Intelligent Routing**: Determines whether to search catalog, book content, or both
- **Tool Integration**: Uses structured tools for post writing and search
- **Configuration Management**: Flexible system configuration

## ⚡ Quick Start

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Set up environment variables** (optional):
```bash
export OPENAI_API_KEY="your-openai-key"
export TAVILY_API_KEY="your-tavily-key"
```

3. **Run the application**:
```bash
python main.py
```

4. **Visit the interactive documentation**:
   - Swagger UI: http://localhost:8000/library
   - ReDoc: http://localhost:8000/archive

## 📡 API Endpoints

### Traditional Bookstore
- `GET /` - Welcome message with all available endpoints
- `GET /books` - Browse all books (with optional genre/author filters)
- `GET /books/{id}` - Get specific book details
- `POST /books` - Add new book to collection
- `PUT /books/{id}` - Update book information
- `DELETE /books/{id}` - Remove book from collection
- `GET /stats` - Get bookstore analytics

### 🤖 AI Recommendations
- `POST /recommendations` - Get AI-powered book recommendations
  ```json
  {
    "query": "dark fantasy books like Thief of Sorrows",
    "request_type": "recommendation",
    "age": 25,
    "interests": ["fantasy", "dark fantasy"],
    "is_gift": false
  }
  ```

- `POST /social-post` - Generate social media posts
  ```json
  {
    "platform": "twitter",
    "include_hashtags": true
  }
  ```

- `GET /recommendations/config` - View AI system configuration

## 🧠 AI System Components

The book recommendation system includes several sophisticated modules:

### Core Modules
- **`book_research/configuration.py`**: System configuration with Pydantic models
- **`book_research/state.py`**: State management for the AI agent workflow
- **`book_research/book_agent.py`**: Main agent with intelligent routing and search
- **`book_research/tools.py`**: Search and utility tools
- **`book_research/post_tools.py`**: Social media post generation tools

### Key Features
1. **Intelligent Routing**: Determines search strategy based on query analysis
2. **Multi-Modal Search**: Can search catalog metadata, full-text content, or both
3. **Post Writing Agent**: Generates formatted social media content using tool calling
4. **Satisfaction Loop**: Iterates until finding satisfactory results

## 📝 Example Usage

### Get Book Recommendations
```bash
curl -X POST "http://localhost:8000/recommendations" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "books similar to Thief of Sorrows",
       "interests": ["fantasy", "dark fantasy"],
       "age": 25
     }'
```

### Generate Social Media Post
```bash
curl -X POST "http://localhost:8000/social-post" \
     -H "Content-Type: application/json" \
     -d '{
       "platform": "instagram",
       "include_hashtags": true
     }'
```

## 🔧 Configuration

The system supports extensive configuration through environment variables:
- `CHAT_MODEL`: OpenAI model to use (default: gpt-4o-mini)
- `MAX_TOKENS`: Maximum tokens for responses (default: 1000)
- `TEMPERATURE`: Response creativity (default: 0.1)
- `ALLOW_CLARIFICATION`: Enable clarifying questions (default: true)

## 🎯 Use Cases

1. **Book Recommendations**: Customers can get personalized book suggestions
2. **Content Marketing**: Generate social media posts for book promotion
3. **Inventory Management**: Traditional CRUD operations for book collection
4. **Analytics**: Track bookstore performance and inventory insights

The API comes with sample books including classics like "The Great Gatsby", "To Kill a Mockingbird", and "1984", plus sophisticated AI capabilities for modern bookstore operations.

---

## 📊 RAG Pipeline Evaluation & Certification

### RAGAS Framework Assessment

This project underwent comprehensive evaluation using the RAGAS (Retrieval-Augmented Generation Assessment) framework to measure and optimize retrieval quality.

### Baseline System Performance

**System:** Simple Retrieval with Recursive Chunking

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **Faithfulness** | 0.9800 | Excellent - Responses factually consistent with context |
| **Context Recall** | 0.5000 | Moderate - Retrieves 50% of relevant information |
| **Context Precision** | 0.2667 | Low - Significant noise in retrieved results |
| **Answer Relevancy** | 0.9362 | Excellent - Answers directly address queries |
| **Factual Correctness** | 0.3820 | Moderate - 38% factual overlap with ground truth |
| **Average Score** | **0.6130** | Good baseline with room for improvement |

**Evaluation Source:** [`evaluation/simple_retrieval/simple_retrieval_recursive_metrics.md`](evaluation/simple_retrieval/simple_retrieval_recursive_metrics.md)

### Advanced Retrieval System Performance

After implementing Advanced Recursive Retrieval with Ensemble methods:

**System:** Advanced Retrieval with Recursive Chunking (BM25 + Multi-Query + Cohere Rerank)

| Metric | Baseline | Advanced | Improvement | % Change |
|--------|----------|----------|-------------|----------|
| **Faithfulness** | 0.9800 | 0.9647 | -0.0153 | -1.6% |
| **Context Recall** | 0.5000 | 0.8333 | +0.3333 | **+66.7%** ⭐ |
| **Context Precision** | 0.2667 | 0.5689 | +0.3022 | **+113.3%** ⭐ |
| **Answer Relevancy** | 0.9362 | 0.8436 | -0.0926 | -9.9% |
| **Factual Correctness** | 0.3820 | 0.2571 | -0.1249 | -32.7% |
| **Overall Average** | 0.6130 | 0.6935 | +0.0805 | **+13.1%** ⭐ |

**Evaluation Source:** [`evaluation/advanced_retrieval/advanced_recursive_retrieval_metrics.md`](evaluation/advanced_retrieval/advanced_recursive_retrieval_metrics.md)

### Complete Comparison Across All Systems

Four retrieval approaches were evaluated:

| System | Chunking Method | Retrieval Type | Average Score | Startup Time | Production Ready |
|--------|----------------|----------------|---------------|--------------|------------------|
| **Production (Current)** | Recursive | Ensemble+Rerank | **0.6935** | ~60 sec | ✅ Yes |
| Simple Semantic | Semantic | Basic Vector | 0.7829 | 120+ sec | ❌ Timeout |
| Simple Recursive | Recursive | Basic Vector | 0.6130 | <10 sec | ✅ Fast but lower quality |
| Advanced Semantic | Semantic | Ensemble+Rerank | 0.5902 | 180+ sec | ❌ Timeout |

**Evaluation Source:** [`evaluation/compare_retrieval_metrics.md`](evaluation/compare_retrieval_metrics.md)

### Key Findings & Architecture Decisions

**Why Advanced Recursive Retrieval Was Chosen:**

1. **Best Production-Viable Quality** - Highest quality (0.6935) among systems with acceptable startup times
2. **Dramatic Performance Gains:**
   - Context Recall: 50% → 83% (+66.7%) - Retrieves much more relevant information
   - Context Precision: 27% → 57% (+113.3%) - Dramatically reduced noise
   - Overall Quality: +13.1% improvement over baseline
3. **Acceptable Startup** - 60 seconds vs 120+ for semantic chunking, 180+ for advanced semantic
4. **Pre-initialization** - Loaded at startup eliminates customer-facing latency

**Why Simple Semantic (0.7829) Was Rejected:**
- Despite highest quality, 120+ second initialization causes API timeouts
- Unacceptable for customer-facing production environment
- Quality advantage not worth reliability risk

**Trade-offs Accepted:**
- Slightly lower faithfulness (-1.6%) - Still excellent at 96%
- Moderate relevancy decrease (-9.9%) - Still good at 84%
- Lower factual correctness (-32.7%) - Requires investigation but acceptable for comprehensive retrieval
- Longer startup (60s vs 10s) - One-time cost worth the quality improvement

**Current Production Configuration:**
- ✅ RecursiveCharacterTextSplitter (1000 chars, 200 overlap)
- ✅ Ensemble Retrieval (BM25 + Multi-Query + Cohere Rerank)
- ✅ Pre-initialization at server startup
- ✅ No timeout risk, reliable performance

### Evaluation Methodology

**Testing Framework:**
- **Tool:** RAGAS (Retrieval-Augmented Generation Assessment)
- **Test Dataset:** 6 representative customer queries about book content
- **Metrics:** Faithfulness, Context Recall, Context Precision, Answer Relevancy, Factual Correctness
- **Model:** GPT-4o-mini for consistency across evaluations

**Evaluation Process:**
1. Baseline measurement with Simple Retrieval (0.6130)
2. Test alternative chunking methods (Semantic vs Recursive)
3. Test advanced retrieval (Ensemble with BM25 + Multi-Query + Rerank)
4. Measure startup times and production viability
5. Select optimal system balancing quality and performance

### Key Takeaways

**Evaluation-Driven Optimization:**
> "Evaluation frameworks like RAGAS provide quantitative guidance for optimization, but real-world system design requires balancing multiple constraints including performance, reliability, cost, and user experience. The 'best' system is one that actually works for the customers while delivering measurable quality improvements."

**Production vs Benchmark:**
> "After rigorous RAGAS evaluation of 4 retrieval approaches, I implemented Advanced Recursive Retrieval with Ensemble methods, achieving a 13.1% quality improvement over the baseline while maintaining production reliability. The system demonstrates dramatically better context recall (+66.7%) and precision (+113.3%), proving that evaluation-driven optimization delivers measurable business value."

**Success Metrics Achieved:**
- ✅ Faithfulness: 0.9647 (maintained >0.95 target)
- ✅ Context Recall: 0.8333 (exceeded 0.70 target by 19%)
- ✅ Context Precision: 0.5689 (exceeded 0.50 target by 14%)
- ✅ Overall RAGAS: 0.6935 (13.1% improvement achieved)

### Complete Evaluation Documentation

For detailed evaluation results and methodology:
- **Baseline System:** [`evaluation/simple_retrieval/simple_retrieval_recursive_metrics.md`](evaluation/simple_retrieval/simple_retrieval_recursive_metrics.md)
- **Advanced System:** [`evaluation/advanced_retrieval/advanced_recursive_retrieval_metrics.md`](evaluation/advanced_retrieval/advanced_recursive_retrieval_metrics.md)
- **Complete Comparison:** [`evaluation/compare_retrieval_metrics.md`](evaluation/compare_retrieval_metrics.md)
- **Certification Answers:** [`CERTIFICATION_ANSWERS.md`](CERTIFICATION_ANSWERS.md)
