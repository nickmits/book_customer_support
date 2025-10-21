"""Book recommendation agent with intelligent search routing, post writing, and satisfaction loop."""

from typing import Literal
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field

from book_research.state import (
    AgentState,
    ClarifyWithUser,
    BookRequest,
    SearchRouting,
    SatisfactionCheck,
)
from book_research.configuration import Configuration
from book_research.tools import get_tavily_search_tool, format_social_post


configurable_model = init_chat_model(
    configurable_fields=("model", "max_tokens", "temperature", "api_key"),
)


async def route_search_query(
    query: str,
    request_type: str,
    config: RunnableConfig
) -> SearchRouting:
    """
    Intelligent routing tool that determines which data source to search.
    NOW INCLUDES POST WRITING DETECTION.
    """
    
    configurable = Configuration.from_runnable_config(config)
   
    model_config = {
        "model": f"openai:{configurable.chat_model}",
        "max_tokens": 500,
        "temperature": 0.1,
        "api_key": config.get("configurable", {}).get("openai_api_key"),
    }
    
    routing_model = (
        configurable_model.with_structured_output(SearchRouting)
        .with_config(model_config)
    )
    
    prompt = f"""Analyze this book search query and determine which action to take.

Query: "{query}"
Request Type: {request_type}

Available Actions:
1. CSV_METADATA: Book catalog with titles, authors, descriptions
2. PDF_FULLTEXT: Full text of "Thief of Sorrows" 
3. BOTH: Search both sources
4. WRITE_POST: Create social media post about Thief of Sorrows

CRITICAL ROUTING RULES (Check in this order):

Rule 0: If query asks to WRITE/CREATE a POST:
   → Route to: write_post
   Keywords: "write", "post", "create post", "social media"
   Examples: "Write a post about Thief of Sorrows", "Create a social media post"

Rule 1: If query contains "Thief of Sorrows" AND asks for content/plot/characters:
   → Route to: pdf_fulltext

Rule 2: If query asks for books SIMILAR/LIKE "Thief of Sorrows":
   → Route to: both
   WHY: Need PDF to understand book AND catalog to find similar books

Rule 3: If query asks about catalog/available books:
   → Route to: csv_metadata

Rule 4: If query asks for general recommendations:
   → Route to: csv_metadata

IMPORTANT: Check for post-writing keywords FIRST!
Does it ask to write/create a post? {"Yes" if any(word in query.lower() for word in ["write post", "create post", "social media", "write about", "write a post"]) else "No"}

If YES → MUST route to: write_post

Your query: "{query}"
Does it mention "Thief of Sorrows"? {"Yes" if "thief of sorrows" in query.lower() else "No"}
Does it ask for similar/like books? {"Yes" if any(word in query.lower() for word in ["similar", "like", "comparable"]) else "No"}

Determine the best action."""

    routing = await routing_model.ainvoke([HumanMessage(content=prompt)])
    return routing



async def post_writing_agent(
    state: AgentState,
    config: RunnableConfig,
    pdf_retriever
) -> Command[Literal["__end__"]]:
    """
    Write social media post about Thief of Sorrows using bind_tools.
    This agent uses the format_social_post tool.
    """
    
    configurable = Configuration.from_runnable_config(config)
    search_results = state.get("search_results", {})
    
    try:
        results = pdf_retriever.get_relevant_documents("plot themes characters quotes")
        passages = []
        for i, doc in enumerate(results[:3], 1):
            page = doc.metadata.get("page_number", "Unknown")
            content = doc.page_content[:350]
            passages.append(f"Passage {i} (Page {page}):\n{content}")
        
        book_content = "\n\n".join(passages) if passages else "No content available."
    except Exception as e:
        book_content = f"Limited content available."
    
    tools = [format_social_post]
    
    model_config = {
        "model": f"openai:{configurable.chat_model}",
        "max_tokens": 2000,
        "temperature": 0.7, 
        "api_key": config.get("configurable", {}).get("openai_api_key"),
    }
    
    llm = init_chat_model(
        configurable_fields=("model", "max_tokens", "temperature", "api_key"),
    ).with_config(model_config)
    
    llm_with_tools = llm.bind_tools(tools)
    
    prompt = f"""You are a creative social media content writer for book promotions.

Write an engaging social media post about "Thief of Sorrows" by Kristen Long.

Book Content from PDF:
{book_content}

Your post should:
- Hook readers with an intriguing opening line
- Highlight compelling plot points or character moments from the passages above
- Be 150-250 words
- Include a call-to-action ("Available now!", "Add to your TBR!", etc.)
- Use 2-3 emojis naturally
- Be conversational and exciting

When you're done writing, use the format_social_post tool to format it properly.

Write the post now:"""
    
    response = await llm_with_tools.ainvoke([HumanMessage(content=prompt)])
    
    final_content = response.content
    
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        if tool_call["name"] == "format_social_post":
            formatted = format_social_post.invoke(tool_call["args"])
            final_content = formatted
    else:
        # LLM didn't use tool, format manually
        final_content = format_social_post.invoke({
            "content": response.content,
            "include_hashtags": True
        })
    
    return Command(
        goto=END,
        update={
            "final_response": final_content,
            "messages": [AIMessage(content=final_content)]
        }
    )

async def clarify_with_user(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["parse_request", "__end__"]]:
    """Analyze user request and ask clarifying questions if needed."""

    configurable = Configuration.from_runnable_config(config)

    if not configurable.allow_clarification:
        return Command(goto="parse_request")

    messages = state.get("messages", [])

    def get_content(msg):
        if isinstance(msg, dict):
            return msg.get("content", "")
        return getattr(msg, "content", "")

    model_config = {
        "model": f"openai:{configurable.chat_model}",
        "max_tokens": configurable.max_tokens,
        "temperature": configurable.temperature,
        "api_key": config.get("configurable", {}).get("openai_api_key"),
    }

    clarification_model = (
        configurable_model.with_structured_output(ClarifyWithUser)
        .with_retry(stop_after_attempt=configurable.max_retries)
        .with_config(model_config)
    )

    user_content = get_content(messages[-1]) if messages else ""
    prompt = f"""Analyze the user's book request and determine if you need clarification.

User message: {user_content}

Consider:
1. Are they asking for a SPECIFIC book by title?
2. Are they asking for RECOMMENDATIONS based on preferences?
3. Do we have enough information (age, interests, genre)?

If asking for a specific book by title, NO clarification needed - proceed directly.
If asking for recommendations but missing key info, ask for clarification.
"""

    response = await clarification_model.ainvoke([HumanMessage(content=prompt)])

    if response.need_clarification:
        return Command(goto=END, update={"messages": [AIMessage(content=response.question)]})
    else:
        return Command(goto="parse_request")

async def parse_request(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["search_subgraph"]]:
    """Parse the user request into structured format."""

    configurable = Configuration.from_runnable_config(config)
    messages = state.get("messages", [])

    def get_content(msg):
        if isinstance(msg, dict):
            return msg.get("content", "")
        return getattr(msg, "content", "")

    def is_user_message(msg):
        if isinstance(msg, dict):
            return msg.get("role") == "user"
        return isinstance(msg, HumanMessage)

    model_config = {
        "model": f"openai:{configurable.chat_model}",
        "max_tokens": configurable.max_tokens,
        "temperature": configurable.temperature,
        "api_key": config.get("configurable", {}).get("openai_api_key"),
    }

    parsing_model = (
        configurable_model.with_structured_output(BookRequest)
        .with_retry(stop_after_attempt=configurable.max_retries)
        .with_config(model_config)
    )

    user_messages = [get_content(msg) for msg in messages if is_user_message(msg)]
    all_context = "\n".join(user_messages)

    prompt = f"""Parse this book request into structured format:

{all_context}

Determine:
- request_type: "specific_book" (user wants a specific title) or "recommendation" (user wants suggestions)
- specific_book_title: Extract the exact book title if mentioned
- age: Age of the reader (0 if not specified)
- interests: List of interests/genres mentioned
- is_gift: Is this book for a gift?
"""

    book_request = await parsing_model.ainvoke([HumanMessage(content=prompt)])

    return Command(goto="search_subgraph", update={"book_request": book_request})
async def generate_response(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["__end__"]]:
    """Generate final response to user."""

    configurable = Configuration.from_runnable_config(config)
    book_request = state.get("book_request")
    search_results = state.get("search_results", {})

    model_config = {
        "model": f"openai:{configurable.chat_model}",
        "max_tokens": configurable.max_tokens,
        "temperature": configurable.temperature,
        "api_key": config.get("configurable", {}).get("openai_api_key"),
    }

    chat_model = configurable_model.with_config(model_config)

    request_type = book_request.request_type if book_request else "unclear"
    books = search_results.get("books", [])
    books_count = search_results.get("books_count", 0)
    query = search_results.get("query", "")
    search_target = search_results.get("search_target", "unknown")
    web_search_info = search_results.get("web_search_info", "")

    if search_target == "pdf_fulltext":
        prompt = f"""The user asked about: '{query}'

📖 We searched the FULL TEXT of "Thief of Sorrows" by Kristen Long.

Search Results ({books_count} relevant passages found):
{books}

Generate a response:
1. Answer their question using the book's content
2. Reference specific passages or details found
3. Be informative and engaging
4. If they asked about plot/characters, provide rich details from the text"""

    elif search_target == "csv_metadata":
        if request_type == "specific_book":
            if books_count > 0:
                book = books[0]
                prompt = f"""The user asked for: '{query}'

Search returned this book from our catalog:
- Title: {book.get('title')}
- Author: {book.get('author')}
- Description: {book.get('description', 'No description available')}
- Subjects: {book.get('subjects', 'N/A')}

CRITICAL: First, check if this book actually matches what the user asked for.
- Does the title match their query? (Consider variations, synonyms, partial matches)
- If the book is CLEARLY DIFFERENT from what they asked for, say we don't have it.

If it matches:
1. Confirm we have "{book.get('title')}" by {book.get('author')}
2. Share a compelling summary
3. Highlight interesting themes

If it doesn't match:
Say: "I searched our catalog, but we don't currently have '{query}' in stock."

{f'''
IMPORTANT - WEB SEARCH RESULTS FOUND:
{web_search_info}

YOU MUST include information from these web search results in your response to help the user learn about the book they asked for. Provide a brief summary based on the search results, including key details like the plot, characters, or themes. Include a link to one of the sources if relevant.''' if web_search_info else "However, we do carry other titles you might enjoy like {book.get('title')}..."}"""
            else:
                prompt = f"""The user asked for: '{query}'

❌ Not found in our catalog.

{f"Web search results: {web_search_info}" if web_search_info else ""}

Generate a helpful response about not having this book."""
        else:
            if books_count > 0:
                prompt = f"""The user wants book recommendations.

Preferences: {book_request.interests if book_request.interests else 'general'}

{f'''IMPORTANT: We DO NOT have books matching "{book_request.interests if book_request.interests else 'their preferences'}" in our catalog.

WEB SEARCH RESULTS:
{web_search_info}

YOU MUST:
1. Start by clearly stating: "We don't currently have {book_request.interests if book_request.interests else 'books matching your preferences'} books in our catalog."
2. Then offer to help: "However, here are some popular {book_request.interests if book_request.interests else ''} books you might be interested in:"
3. List the books from the web search results with titles and brief descriptions.
4. DO NOT suggest the books from our catalog (they don't match what the user wants).''' if web_search_info else f'''📚 Found {books_count} books in our CATALOG:
{books}

Generate recommendations with titles, authors, and descriptions.'''}"""
            else:
                prompt = f"""The user wants book recommendations.

Preferences: {book_request.interests if book_request.interests else 'general'}

❌ Not found in our catalog.

{f'''WEB SEARCH RESULTS:
{web_search_info}

YOU MUST:
1. Start by clearly stating: "We don't currently have {book_request.interests if book_request.interests else 'books matching your preferences'} books in our catalog."
2. Then offer: "However, here are some popular {book_request.interests if book_request.interests else ''} books you might be interested in:"
3. List the books from the web search results with titles and brief descriptions.''' if web_search_info else "Apologize that we don't have books matching their preferences in our catalog."}"""

    elif search_target == "both":
        prompt = f"""The user asked: '{query}'

We searched BOTH our catalog and full book content.

Results: {books_count} items found
{books}

Generate a comprehensive response using information from both sources."""

    else:
        prompt = f"""Generate a response for: '{query}'
Results: {search_results}"""

    response = await chat_model.ainvoke([HumanMessage(content=prompt)])

    return Command(goto=END, update={"final_response": response.content, "messages": [response]})

async def route_search(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["csv_search_agent", "pdf_search_agent", "combined_search_agent", "post_writing_agent"]]:
    """Route the search to appropriate data source based on query analysis."""
    
    book_request = state.get("book_request")
    messages = state.get("messages", [])
    
    original_query = ""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            original_query = msg.get("content", "")
            break
        elif hasattr(msg, "__class__") and msg.__class__.__name__ == "HumanMessage":
            original_query = msg.content
            break
    
    if book_request.request_type == "specific_book":
        search_query = book_request.specific_book_title
    else:
        query_parts = []
        if book_request.interests:
            query_parts.append(" ".join(book_request.interests))
        search_query = " ".join(query_parts) if query_parts else "books"
    
    routing = await route_search_query(
        original_query if original_query else search_query,
        book_request.request_type, 
        config
    )
    
    search_results = {
        "query": search_query,
        "original_query": original_query,
        "request_type": book_request.request_type,
        "search_target": routing.search_target,
        "routing_reasoning": routing.reasoning
    }
    
    if routing.search_target == "write_post":
        next_node = "post_writing_agent"
    elif routing.search_target == "pdf_fulltext":
        next_node = "pdf_search_agent"
    elif routing.search_target == "csv_metadata":
        next_node = "csv_search_agent"
    else:  
        next_node = "combined_search_agent"
    
    return Command(
        goto=next_node,
        update={
            "search_results": search_results,
            "messages": [AIMessage(content=f"🧭 Routing: {routing.reasoning} → {routing.search_target}")]
        }
    )


async def csv_search_agent(
    state: AgentState, config: RunnableConfig, csv_retriever
) -> Command[Literal["check_results"]]:
    """Search CSV metadata (book catalog)."""
    
    search_results = state.get("search_results", {})
    query = search_results.get("query", "")
    
    try:
        results = csv_retriever.get_relevant_documents(query)
        
        books = []
        for doc in results[:5]:
            books.append({
                "title": doc.metadata.get("title", "Unknown"),
                "author": doc.metadata.get("author", "Unknown"),
                "description": doc.page_content[:500],
                "subjects": doc.metadata.get("subjects", ""),
                "source": "csv_catalog"
            })
        
        search_results["books"] = books
        search_results["books_count"] = len(books)
        
        return Command(
            goto="check_results",
            update={
                "search_results": search_results,
                "messages": [AIMessage(content=f"📚 CSV Search: Found {len(books)} books in catalog")]
            }
        )
    
    except Exception as e:
        search_results["books"] = []
        search_results["books_count"] = 0
        search_results["error"] = str(e)
        return Command(goto="check_results", update={"search_results": search_results})


async def pdf_search_agent(
    state: AgentState, config: RunnableConfig, pdf_retriever
) -> Command[Literal["check_results"]]:
    """Search PDF full text (Thief of Sorrows)."""
    
    search_results = state.get("search_results", {})
    query = search_results.get("query", "")
    
    try:
        results = pdf_retriever.get_relevant_documents(query)
        
        passages = []
        for doc in results[:5]:
            passages.append({
                "content": doc.page_content[:500],
                "page": doc.metadata.get("page_number", "Unknown"),
                "book_title": doc.metadata.get("book_title", "Thief of Sorrows"),
                "author": doc.metadata.get("author", "Kristen Long"),
                "source": "pdf_fulltext"
            })
        
        search_results["books"] = passages
        search_results["books_count"] = len(passages)
        
        return Command(
            goto="check_results",
            update={
                "search_results": search_results,
                "messages": [AIMessage(content=f"📖 PDF Search: Found {len(passages)} relevant passages")]
            }
        )
    
    except Exception as e:
        search_results["books"] = []
        search_results["books_count"] = 0
        search_results["error"] = str(e)
        return Command(goto="check_results", update={"search_results": search_results})

async def combined_search_agent(
    state: AgentState, config: RunnableConfig, csv_retriever, pdf_retriever
) -> Command[Literal["check_results"]]:
    """Search both CSV and PDF sources."""
    
    search_results = state.get("search_results", {})
    query = search_results.get("query", "")
    
    try:
        csv_results = csv_retriever.get_relevant_documents(query)
        csv_books = []
        for doc in csv_results[:3]:
            csv_books.append({
                "title": doc.metadata.get("title", "Unknown"),
                "author": doc.metadata.get("author", "Unknown"),
                "description": doc.page_content[:300],
                "source": "csv_catalog"
            })
        
        pdf_results = pdf_retriever.get_relevant_documents(query)
        pdf_passages = []
        for doc in pdf_results[:3]:
            pdf_passages.append({
                "content": doc.page_content[:300],
                "page": doc.metadata.get("page_number", "Unknown"),
                "source": "pdf_fulltext"
            })
        
        all_results = csv_books + pdf_passages
        
        search_results["books"] = all_results
        search_results["books_count"] = len(all_results)
        
        return Command(
            goto="check_results",
            update={
                "search_results": search_results,
                "messages": [AIMessage(content=f"🔍 Combined Search: Found {len(csv_books)} catalog + {len(pdf_passages)} passages")]
            }
        )
    
    except Exception as e:
        search_results["books"] = []
        search_results["books_count"] = 0
        search_results["error"] = str(e)
        return Command(goto="check_results", update={"search_results": search_results})


async def check_results(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["web_search_agent", "__end__"]]:
    """Check if we found good results or need web search."""

    search_results = state.get("search_results", {})
    books_count = search_results.get("books_count", 0)
    search_target = search_results.get("search_target", "")
    book_request = state.get("book_request")
    request_type = book_request.request_type if book_request else "unclear"

    # For PDF full text searches, always accept the results
    if search_target == "pdf_fulltext":
        return Command(goto=END, update={"search_results": search_results})

    if books_count > 0:
        if request_type == "specific_book":
            query = search_results.get("query", "").lower()
            original_query = search_results.get("original_query", "").lower()
            books = search_results.get("books", [])

            has_relevant_match = False
            for book in books:
                book_title = book.get("title", "").lower()
        
                query_words = [w for w in query.split() if len(w) > 3] 
                if any(word in book_title for word in query_words):
                    has_relevant_match = True
                    break

            if has_relevant_match:
                # Found a relevant book, proceed with results
                return Command(goto=END, update={"search_results": search_results})
            else:
                # Books found but none match the specific query - try web search
                return Command(
                    goto="web_search_agent",
                    update={"messages": [AIMessage(content="No exact match found in catalog. Searching the web...")]}
                )
        else:
            # For recommendation requests, validate books match the interests/genre
            if request_type == "recommendation":
                interests = book_request.interests if book_request else []
                if interests:
                    # Check if books actually match the requested interests/genre
                    books = search_results.get("books", [])
                    query = search_results.get("query", "").lower()

                    # Extract key genre/interest words (interests is already a list)
                    interest_words = [w.lower() for interest in interests for w in interest.split() if len(w) > 3]
                    query_words = [w for w in query.split() if len(w) > 3]
                    all_keywords = set(interest_words + query_words)

                    # Check if any book subjects/description match the interests
                    has_relevant_match = False
                    for book in books:
                        subjects = str(book.get("subjects", "")).lower()
                        description = str(book.get("description", "")).lower()
                        title = str(book.get("title", "")).lower()

                        # Check if any keyword appears in subjects, description, or title
                        if any(keyword in subjects or keyword in description or keyword in title
                               for keyword in all_keywords):
                            has_relevant_match = True
                            break

                    if not has_relevant_match:
                        # Books don't match the requested genre/interests - try web search
                        return Command(
                            goto="web_search_agent",
                            update={"messages": [AIMessage(content="No matching books found for your interests. Searching the web...")]}
                        )

            # Books are relevant or no specific interests to validate against
            return Command(goto=END, update={"search_results": search_results})
    else:
        # No books found at all - definitely try web search
        return Command(
            goto="web_search_agent",
            update={"messages": [AIMessage(content="No results found in catalog. Searching the web...")]}
        )


async def web_search_agent(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["__end__"]]:
    """Search the web for book information."""

    search_results = state.get("search_results", {})
    query = search_results.get("query", "")
    configurable = Configuration.from_runnable_config(config)

    tavily_tool = get_tavily_search_tool(max_results=configurable.max_web_search_results)

    try:
        results = await tavily_tool.ainvoke({"query": f"{query} book"})

        web_info = f"Web search results for '{query}':\n\n{results}"
        search_results["web_search_info"] = web_info

        return Command(
            goto=END,
            update={
                "search_results": search_results,
                "messages": [AIMessage(content=f"Web search completed for '{query}'")]
            }
        )

    except Exception as e:
        search_results["web_search_info"] = f"Could not search web: {str(e)}"
        return Command(goto=END, update={"search_results": search_results})




# ============================================================================
# SATISFACTION CHECK NODE (New - implements the loop logic)
# ============================================================================

async def check_satisfaction(
    state: AgentState,
    config: RunnableConfig
) -> Command[Literal["search_subgraph", "generate_response"]]:
    """
    Check if search results are satisfactory and decide whether to continue.
    Similar to supervisor_tools checking if research is complete.
    """

    configurable = Configuration.from_runnable_config(config)
    search_results = state.get("search_results", {})
    search_iterations = state.get("search_iterations", 0)
    book_request = state.get("book_request")

    if search_iterations >= 3:
        return Command(
            goto="generate_response",
            update={
                "messages": [AIMessage(content="Maximum search attempts reached. Proceeding with available results.")]
            }
        )

    books_count = search_results.get("books_count", 0)
    search_target = search_results.get("search_target", "")
    request_type = book_request.request_type if book_request else "unclear"

    if search_target == "write_post":
        return Command(goto="generate_response")

    if request_type == "recommendation" and books_count > 0:
        return Command(goto="generate_response")
    
    model_config = {
        "model": f"openai:{configurable.chat_model}",
        "max_tokens": 500,
        "temperature": 0.1,
        "api_key": config.get("configurable", {}).get("openai_api_key"),
    }
    
    satisfaction_model = (
        configurable_model
        .with_structured_output(SatisfactionCheck)
        .with_config(model_config)
    )
    
    original_query = search_results.get("original_query", "")
    current_findings = search_results.get("books", [])
    
    prompt = f"""Evaluate if the search results are sufficient.

Original Query: "{original_query}"
Search Target: {search_target}
Results Found: {books_count} items
Iteration: {search_iterations + 1}/3

Current Findings Summary:
{str(current_findings)[:500]}...

Determine:
1. Are these results sufficient to answer the user's query?
2. If not, what search strategy should we try next?

Consider:
- Do we have the specific book they asked for?
- Do we have enough variety for recommendations?
- Is critical information missing?"""

    evaluation = await satisfaction_model.ainvoke([HumanMessage(content=prompt)])
    
    if evaluation.is_satisfied:
        return Command(
            goto="generate_response",
            update={
                "messages": [AIMessage(content=f"✅ Search complete: {evaluation.reasoning}")]
            }
        )
    else:
        if evaluation.next_strategy == "try_different_source":
            new_target = "pdf_fulltext" if search_target == "csv_metadata" else "csv_metadata"
            search_results["search_target"] = new_target
        elif evaluation.next_strategy == "expand_search":
            search_results["query"] = f"{search_results.get('query', '')} similar related"
        
        return Command(
            goto="search_subgraph",
            update={
                "search_results": search_results,
                "search_iterations": search_iterations + 1,
                "messages": [AIMessage(content=f"🔄 Refining search (attempt {search_iterations + 2}/3): {evaluation.reasoning}")]
            }
        )




def build_search_subgraph_with_loop(csv_retriever, pdf_retriever):
    """Build search subgraph with satisfaction checking loop."""
    
    subgraph = StateGraph(AgentState, config_schema=Configuration)
    
    async def csv_search_with_retriever(state, config):
        return await csv_search_agent(state, config, csv_retriever)
    
    async def pdf_search_with_retriever(state, config):
        return await pdf_search_agent(state, config, pdf_retriever)
    
    async def combined_search_with_retriever(state, config):
        return await combined_search_agent(state, config, csv_retriever, pdf_retriever)
    
    async def post_writing_with_retriever(state, config):
        return await post_writing_agent(state, config, pdf_retriever)

    subgraph.add_node("route_search", route_search)
    subgraph.add_node("csv_search_agent", csv_search_with_retriever)
    subgraph.add_node("pdf_search_agent", pdf_search_with_retriever)
    subgraph.add_node("combined_search_agent", combined_search_with_retriever)
    subgraph.add_node("web_search_agent", web_search_agent)
    subgraph.add_node("post_writing_agent", post_writing_with_retriever)
    subgraph.add_node("check_results", check_results)

    subgraph.add_edge(START, "route_search")

    subgraph.add_edge("csv_search_agent", "check_results")
    subgraph.add_edge("pdf_search_agent", "check_results")
    subgraph.add_edge("combined_search_agent", "check_results")
    # check_results conditionally routes to web_search_agent or END

    subgraph.add_edge("web_search_agent", END)

    subgraph.add_edge("post_writing_agent", END)

    return subgraph.compile()


def build_book_agent(csv_retriever, pdf_retriever):
    """Build the main book agent with search refinement loop."""
    
    search_subgraph = build_search_subgraph_with_loop(csv_retriever, pdf_retriever)
    
    builder = StateGraph(AgentState, config_schema=Configuration)
    
    builder.add_node("clarify_with_user", clarify_with_user)
    builder.add_node("parse_request", parse_request)
    builder.add_node("search_subgraph", search_subgraph)
    builder.add_node("check_satisfaction", check_satisfaction)  
    builder.add_node("generate_response", generate_response)
    
    builder.add_edge(START, "clarify_with_user")
    builder.add_edge("parse_request", "search_subgraph")
    builder.add_edge("search_subgraph", "check_satisfaction")  
    builder.add_edge("generate_response", END)
    
    return builder.compile()