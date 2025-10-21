"""Tools for the book recommendation system."""

import os
from typing import Optional
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults


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


@tool
def format_social_post(
    content: str,
    include_hashtags: bool = True
) -> str:
    """
    Format content as a social media post.

    Args:
        content: The main content to format
        include_hashtags: Whether to include hashtags (default True)

    Returns:
        Formatted post ready for social media
    """
    hashtags = "\n\n#ThiefOfSorrows #Fantasy #DarkFantasy #BookRecommendation" if include_hashtags else ""

    post = f"""
📱 SOCIAL MEDIA POST - Thief of Sorrows
{'='*70}

{content}
{hashtags}

{'='*70}
✅ Ready to copy and paste!
"""
    return post
