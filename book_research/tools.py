"""Tools for the book recommendation system."""

import os
from typing import Optional
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from pydantic import Field


@tool
def think_tool(reflection: str) -> str:
    """Use this tool to reflect on your current progress and plan next steps.

    Args:
        reflection: Your thoughts about the current state and what to do next

    Returns:
        Confirmation that reflection was recorded
    """
    return f"Reflection recorded: {reflection}"


# Tavily search tool
def get_tavily_search_tool(max_results: int = 3):
    """Create a Tavily search tool for web search.

    Args:
        max_results: Maximum number of search results to return

    Returns:
        TavilySearchResults tool
    """
    return TavilySearchResults(
        max_results=max_results,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=False,
        include_images=False,
    )


# Vector store search will be initialized in the graph
# since it depends on the ensemble retriever
def create_vector_search_tool(ensemble_retriever):
    """Create a tool for searching the vector store.

    Args:
        ensemble_retriever: The ensemble retriever with vector store

    Returns:
        A search tool function
    """
    @tool
    def vector_search_books(query: str) -> str:
        """Search for books in the vector database.

        Args:
            query: The search query for finding books

        Returns:
            Information about relevant books found
        """
        try:
            results = ensemble_retriever.get_relevant_documents(query)

            if not results:
                return "No books found matching your query."

            output = []
            for i, doc in enumerate(results[:5], 1):  # Top 5 results
                title = doc.metadata.get('title', 'Unknown')
                author = doc.metadata.get('author', 'Unknown')
                subjects = doc.metadata.get('subjects', '')[:200]  # Truncate subjects

                # Extract description from page_content
                description = ""
                if 'Description:' in doc.page_content:
                    desc_part = doc.page_content.split('Description:')[1]
                    if 'Subjects:' in desc_part:
                        description = desc_part.split('Subjects:')[0].strip()[:200]

                output.append(
                    f"{i}. {title} by {author}\n"
                    f"   Description: {description}\n"
                    f"   Subjects: {subjects}\n"
                )

            return "\n".join(output)

        except Exception as e:
            return f"Error searching vector store: {str(e)}"

    return vector_search_books


def get_all_tools(ensemble_retriever, config):
    """Get all tools for the agent.

    Args:
        ensemble_retriever: The ensemble retriever for vector search
        config: Configuration object

    Returns:
        List of tools
    """
    tools = [
        think_tool,
        create_vector_search_tool(ensemble_retriever),
        get_tavily_search_tool(max_results=config.max_web_search_results),
    ]

    return tools
