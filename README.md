# 📚 BookStore API with AI Recommendations

A sophisticated FastAPI application combining traditional bookstore management with AI-powered book recommendations and social media content generation.

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
