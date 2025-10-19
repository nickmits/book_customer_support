"""State definitions for the book recommendation system."""

from typing import TypedDict, Annotated, Literal
from operator import add
from pydantic import BaseModel, Field
from typing import TypedDict, Optional, Annotated
from langgraph.graph import add_messages

def merge_dicts(left: dict | None, right: dict | None) -> dict:
    """Merge two dictionaries, with right taking precedence."""
    if left is None:
        return right or {}
    if right is None:
        return left
    return {**left, **right}


class ClarifyWithUser(BaseModel):
    """Structured output for clarifying user needs."""

    need_clarification: bool = Field(
        description="Whether we need to ask clarifying questions"
    )
    question: str = Field(
        description="The clarifying question to ask the user"
    )
    verification: str = Field(
        description="Verification message if no clarification needed"
    )


class BookRequest(BaseModel):
    """Structured output for parsed book request."""

    request_type: Literal["specific_book", "recommendation", "unclear"] = Field(
        description="Type of request: specific book, general recommendation, or unclear"
    )
    specific_book_title: str = Field(
        default="",
        description="Title of specific book if user wants one"
    )
    age: int = Field(
        default=0,
        description="Age of reader (0 if not specified)"
    )
    interests: list[str] = Field(
        default_factory=list,
        description="List of interests/genres"
    )
    is_gift: bool = Field(
        default=False,
        description="Whether book is a gift"
    )
    additional_context: str = Field(
        default="",
        description="Any additional context about the request"
    )


class SearchComplete(BaseModel):
    """Tool to signal search is complete."""

    summary: str = Field(
        description="Summary of search results"
    )


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    book_request: Optional[BookRequest]
    search_results: dict
    final_response: str
    
    search_iterations: int 
    post_revisions: int   
    satisfaction_status: Optional[dict] 

class SearchState(TypedDict):
    """State for the book search subgraph."""

    book_query: str
    request_type: str
    search_results: Annotated[dict, merge_dicts]
    found_in_vectorstore: bool
    web_search_results: str
    similar_books: list
