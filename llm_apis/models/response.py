"""
Response models for LLM API interactions.
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime


class Usage(BaseModel):
    """Token usage information."""
    prompt_tokens: int = Field(..., description="Number of tokens in the prompt")
    completion_tokens: Optional[int] = Field(None, description="Number of tokens in the completion")
    total_tokens: int = Field(..., description="Total number of tokens used")


class ChatChoice(BaseModel):
    """Individual chat completion choice."""
    index: int = Field(..., description="Choice index")
    message: Dict[str, Any] = Field(..., description="Generated message")
    finish_reason: Optional[str] = Field(None, description="Reason for completion finish")


class ChatResponse(BaseModel):
    """Response model for chat completions."""
    id: str = Field(..., description="Unique response identifier")
    object: str = Field("chat.completion", description="Object type")
    created: int = Field(..., description="Creation timestamp")
    model: str = Field(..., description="Model used for generation")
    choices: List[ChatChoice] = Field(..., description="List of completion choices")
    usage: Optional[Usage] = Field(None, description="Token usage information")
    provider: str = Field(..., description="Provider name")


class EmbeddingData(BaseModel):
    """Individual embedding data."""
    object: str = Field("embedding", description="Object type")
    embedding: List[float] = Field(..., description="Embedding vector")
    index: int = Field(..., description="Embedding index")


class EmbeddingResponse(BaseModel):
    """Response model for embeddings."""
    object: str = Field("list", description="Object type")
    data: List[EmbeddingData] = Field(..., description="List of embeddings")
    model: str = Field(..., description="Model used for embeddings")
    usage: Optional[Usage] = Field(None, description="Token usage information")
    provider: str = Field(..., description="Provider name")


class CompletionChoice(BaseModel):
    """Individual completion choice."""
    text: str = Field(..., description="Generated text")
    index: int = Field(..., description="Choice index")
    finish_reason: Optional[str] = Field(None, description="Reason for completion finish")


class CompletionResponse(BaseModel):
    """Response model for text completions."""
    id: str = Field(..., description="Unique response identifier")
    object: str = Field("text_completion", description="Object type")
    created: int = Field(..., description="Creation timestamp")
    model: str = Field(..., description="Model used for generation")
    choices: List[CompletionChoice] = Field(..., description="List of completion choices")
    usage: Optional[Usage] = Field(None, description="Token usage information")
    provider: str = Field(..., description="Provider name")