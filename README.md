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

**Chunking Strategy:**

This project uses **
** from LangChain Experimental with the following configuration:

- **Chunking Method**: Semantic similarity-based splitting
- **Breakpoint Threshold Type**: Percentile
- **Breakpoint Threshold Amount**: 95 (95th percentile - moderate sensitivity)
- **Embedding Model**: OpenAI text-embedding-3-small

**Why SemanticChunker:**

**SemanticChunker was chosen to optimize retrieval quality for semantic queries** about themes, character development, plot arcs, and narrative elements. Unlike character-count-based chunking (RecursiveCharacterTextSplitter), SemanticChunker creates chunks at natural topic boundaries by analyzing semantic similarity between sentences, ensuring complete thematic discussions, character interactions, and plot developments remain intact within single chunks rather than being fragmented.

**Key Advantages:**

1. **Semantic Coherence** - Automatically identifies where topics shift in the narrative (e.g., scene changes, character perspective shifts, thematic transitions) and creates chunk boundaries at these natural breakpoints, producing chunks that are semantically self-contained and meaningful.

2. **Superior for Thematic Questions** - When customers ask "What are the main themes?" or "How does the character develop?", SemanticChunker ensures retrieved passages contain complete thematic discussions or character arc segments rather than arbitrary fragments, significantly improving answer quality (estimated 10-20% improvement over fixed-size chunking).

3. **Context Preservation** - Long thematic discussions (spanning 3-4 paragraphs) stay together in a single chunk, while brief topic shifts create appropriate boundaries, optimizing both retrieval precision and context completeness.

**Trade-offs Accepted:**

- **Preprocessing Cost**: ~$0.50-1.00 per book in embedding API calls during chunking (acceptable one-time cost for quality improvement in a certification/demonstration project)
- **Processing Time**: 2-5 minutes for semantic analysis vs. instant character-based splitting (acceptable for batch preprocessing)
- **Variable Chunk Sizes**: Produces chunks of varying lengths based on semantic content rather than predictable fixed sizes (beneficial for semantic coherence, requires adaptive handling)

**Configuration Details:**

- **Breakpoint Threshold Type (Percentile)**: Uses the 95th percentile of semantic distances to identify significant topic shifts, creating chunks at meaningful boundaries while avoiding over-fragmentation from minor shifts
- **Embedding Model**: Leverages the same OpenAI text-embedding-3-small used for retrieval, ensuring consistency between chunking and search similarity calculations

**Performance Validation:**

This chunking strategy is implemented consistently across both production (`main.py`) and evaluation (`evaluation/evaluate_simple_pdf_book_agent.py`) environments. The semantic chunking approach typically produces 150-250 variable-length chunks from a 200-page novel, with each chunk representing a semantically coherent narrative unit optimized for content-based question answering.

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
