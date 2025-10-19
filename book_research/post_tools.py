"""
Post Writing Agent Using bind_tools - CORRECT IMPLEMENTATION
This version properly uses LangChain's bind_tools for tool calling
"""

from typing import Literal, List
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field

from book_research.state import AgentState
from book_research.configuration import Configuration


# ============================================================================
# TOOL DEFINITIONS
# ============================================================================

@tool
def retrieve_book_passages(
    query: str,
    num_passages: int = 3
) -> str:
    """
    Retrieve relevant passages from Thief of Sorrows PDF.
    
    Args:
        query: What to search for (e.g., "plot", "characters", "themes")
        num_passages: Number of passages to retrieve (default 3)
        
    Returns:
        String containing relevant passages with page numbers
    """
    # This is a placeholder - will be replaced with actual retriever at runtime
    return "Placeholder for book passages"


@tool
def format_social_post(
    content: str,
    platform: str = "general",
    include_hashtags: bool = True
) -> str:
    """
    Format content as a social media post.
    
    Args:
        content: The main content to format
        platform: Platform type (general, twitter, instagram, facebook)
        include_hashtags: Whether to include hashtags
        
    Returns:
        Formatted post ready for social media
    """
    hashtags = "\n\n#ThiefOfSorrows #Fantasy #DarkFantasy #BookRecommendation" if include_hashtags else ""
    
    post = f"""
📱 SOCIAL MEDIA POST
{'='*70}

{content}
{hashtags}

{'='*70}
✅ Ready to copy and paste!
"""
    return post


# ============================================================================
# POST WRITING AGENT WITH bind_tools
# ============================================================================

async def post_writing_agent_with_tools(
    state: AgentState,
    config: RunnableConfig,
    pdf_retriever
) -> Command[Literal["__end__"]]:
    """
    Post writing agent that uses bind_tools for tool calling.
    The LLM decides when to use tools autonomously.
    """
    
    configurable = Configuration.from_runnable_config(config)
    search_results = state.get("search_results", {})
    messages = state.get("messages", [])
    
    # Create tool that captures the retriever
    def create_retrieval_tool(retriever):
        @tool
        def retrieve_book_passages_impl(query: str, num_passages: int = 3) -> str:
            """Retrieve relevant passages from Thief of Sorrows PDF."""
            try:
                results = retriever.get_relevant_documents(query)
                passages = []
                for i, doc in enumerate(results[:num_passages], 1):
                    page = doc.metadata.get("page_number", "Unknown")
                    content = doc.page_content[:300]
                    passages.append(f"Passage {i} (Page {page}):\n{content}\n")
                
                return "\n".join(passages) if passages else "No relevant passages found."
            except Exception as e:
                return f"Error retrieving passages: {str(e)}"
        
        return retrieve_book_passages_impl
    
    # Create the actual tool with retriever
    retrieval_tool = create_retrieval_tool(pdf_retriever)
    
    # Define available tools
    tools = [retrieval_tool, format_social_post]
    
    # Create model with tools bound
    model_config = {
        "model": f"openai:{configurable.chat_model}",
        "max_tokens": 2000,
        "temperature": 0.7,  # Creative for post writing
        "api_key": config.get("configurable", {}).get("openai_api_key"),
    }
    
    llm = init_chat_model(
        configurable_fields=("model", "max_tokens", "temperature", "api_key"),
    ).with_config(model_config)
    
    # BIND TOOLS to the model
    llm_with_tools = llm.bind_tools(tools)
    
    # Create the prompt
    system_prompt = """You are a creative social media content writer specializing in book promotions.

Your task: Write an engaging social media post about "Thief of Sorrows" by Kristen Long.

Available tools:
1. retrieve_book_passages: Get actual content from the book (plot, characters, themes, quotes)
2. format_social_post: Format your content for social media

Process:
1. FIRST: Use retrieve_book_passages to get interesting content from the book
2. THEN: Write an engaging post using that content
3. FINALLY: Use format_social_post to format it properly

The post should:
- Hook readers immediately
- Highlight compelling aspects from the book
- Be 150-250 words
- Include a call-to-action
- Use 2-3 emojis appropriately
- Be ready to copy-paste

Start by retrieving relevant passages from the book."""
    
    # Build conversation history
    conversation = [
        {"role": "system", "content": system_prompt},
    ]
    
    # Add user messages
    for msg in messages:
        if isinstance(msg, HumanMessage):
            conversation.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            conversation.append({"role": "assistant", "content": msg.content})
    
    # If no explicit user message, add the task
    if not any(isinstance(m, HumanMessage) for m in messages):
        conversation.append({
            "role": "user", 
            "content": "Write an engaging social media post about Thief of Sorrows"
        })
    
    # Agentic loop - let LLM use tools autonomously
    max_iterations = 5
    iteration = 0
    
    agent_messages = []
    
    while iteration < max_iterations:
        iteration += 1
        
        # Get LLM response with tool calling
        response = await llm_with_tools.ainvoke(conversation)
        agent_messages.append(response)
        
        # Check if LLM wants to call tools
        if response.tool_calls:
            # Execute each tool call
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]
                
                # Find and execute the tool
                if tool_name == "retrieve_book_passages_impl":
                    result = retrieval_tool.invoke(tool_args)
                elif tool_name == "format_social_post":
                    result = format_social_post.invoke(tool_args)
                else:
                    result = f"Unknown tool: {tool_name}"
                
                # Add tool result to conversation
                conversation.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [tool_call]
                })
                conversation.append({
                    "role": "tool",
                    "content": result,
                    "tool_call_id": tool_id
                })
        else:
            # No more tool calls - LLM has finished
            final_response = response.content
            break
    else:
        # Max iterations reached
        final_response = "Post generation incomplete - please try again."
    
    return Command(
        goto=END,
        update={
            "final_response": final_response,
            "messages": [AIMessage(content=final_response)]
        }
    )


# ============================================================================
# SIMPLER VERSION: Single-shot with bind_tools
# ============================================================================

async def simple_post_agent_with_tools(
    state: AgentState,
    config: RunnableConfig,
    pdf_retriever
) -> Command[Literal["__end__"]]:
    """
    Simpler version: Get passages first, then let LLM write with tools available.
    """
    
    configurable = Configuration.from_runnable_config(config)
    search_results = state.get("search_results", {})
    
    # Step 1: Get book passages
    try:
        results = pdf_retriever.get_relevant_documents("plot themes characters")
        passages = []
        for i, doc in enumerate(results[:3], 1):
            page = doc.metadata.get("page_number", "Unknown")
            content = doc.page_content[:300]
            passages.append(f"Passage {i} (Page {page}):\n{content}")
        
        book_content = "\n\n".join(passages) if passages else "No content available."
    except Exception as e:
        book_content = f"Error: {str(e)}"
    
    # Step 2: Create model with formatting tool bound
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
    
    # Bind the formatting tool
    llm_with_tools = llm.bind_tools(tools)
    
    # Create prompt with book content
    prompt = f"""Write an engaging social media post about "Thief of Sorrows" by Kristen Long.

Book Content Available:
{book_content}

Create a post that:
- Hooks readers immediately
- Highlights compelling plot points or character moments  
- Is 150-250 words
- Includes a call-to-action
- Uses 2-3 emojis
- Ends with hashtags

Use the format_social_post tool to format your final post when ready."""
    
    # Get response - LLM may or may not use the tool
    response = await llm_with_tools.ainvoke([HumanMessage(content=prompt)])
    
    # Check if LLM called the formatting tool
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        if tool_call["name"] == "format_social_post":
            formatted = format_social_post.invoke(tool_call["args"])
            final_content = formatted
        else:
            final_content = response.content
    else:
        # LLM didn't use tool, format it ourselves
        final_content = format_social_post.invoke({
            "content": response.content,
            "platform": "general",
            "include_hashtags": True
        })
    
    return Command(
        goto=END,
        update={
            "final_response": final_content,
            "messages": [AIMessage(content=final_content)]
        }
    )


# ============================================================================
# INTEGRATION - Replace in your book_agent.py
# ============================================================================

"""
In your build_search_subgraph function:

# Use the SIMPLE version for easier integration:
async def write_post_with_tools(state, config):
    return await simple_post_agent_with_tools(state, config, pdf_retriever)

subgraph.add_node("write_post_agent", write_post_with_tools)

# Or use the AGENTIC version for autonomous tool calling:
async def write_post_agentic(state, config):
    return await post_writing_agent_with_tools(state, config, pdf_retriever)

subgraph.add_node("write_post_agent", write_post_agentic)
"""