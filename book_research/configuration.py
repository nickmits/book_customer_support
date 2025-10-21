"""Simple configuration for the book recommendation system."""

import os
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig


class Configuration(BaseModel):
    """Configuration for the book recommendation agent."""

    # Model configuration
    chat_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model for chat and reasoning"
    )

    max_tokens: int = Field(
        default=1000,
        description="Maximum tokens for model responses"
    )

    temperature: float = Field(
        default=0.1,
        description="Temperature for model responses (0-1)"
    )

    # Retrieval configuration
    vector_search_k: int = Field(
        default=10,
        description="Number of results from vector search"
    )

    bm25_k: int = Field(
        default=5,
        description="Number of results from BM25 search"
    )

    ensemble_weights: list[float] = Field(
        default=[0.2, 0.3, 0.5],
        description="Weights for ensemble retriever [bm25, multi_query, compression]"
    )

    max_web_search_results: int = Field(
        default=3,
        description="Maximum number of web search results to consider"
    )

    allow_clarification: bool = Field(
        default=True,
        description="Whether to ask clarifying questions"
    )

    max_retries: int = Field(
        default=3,
        description="Maximum retries for API calls"
    )

    @classmethod
    def from_runnable_config(cls, config: Optional[RunnableConfig] = None) -> "Configuration":
        """Create Configuration from RunnableConfig."""
        if not config:
            return cls()

        configurable = config.get("configurable", {})
        field_names = list(cls.model_fields.keys())

        values = {
            field_name: os.environ.get(field_name.upper(), configurable.get(field_name))
            for field_name in field_names
        }

        return cls(**{k: v for k, v in values.items() if v is not None})
