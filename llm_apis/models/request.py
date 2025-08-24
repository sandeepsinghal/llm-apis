"""
Request models for LLM API interactions.
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Individual chat message."""
    role: str = Field(..., description="Role of the message sender (user, assistant, system)")
    content: str = Field(..., description="Content of the message")
    name: Optional[str] = Field(None, description="Name of the message sender")


class ChatRequest(BaseModel):
    """Request model for chat completions."""
    messages: List[ChatMessage] = Field(..., description="List of chat messages")
    model: str = Field(..., description="Model identifier")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(None, ge=1, description="Maximum tokens to generate")
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    frequency_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0, description="Frequency penalty")
    presence_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0, description="Presence penalty")
    stop: Optional[Union[str, List[str]]] = Field(None, description="Stop sequences")
    stream: Optional[bool] = Field(False, description="Whether to stream responses")


class EmbeddingRequest(BaseModel):
    """Request model for embeddings."""
    texts: List[str] = Field(..., description="List of texts to embed")
    model: str = Field(..., description="Model identifier")
    encoding_format: Optional[str] = Field("float", description="Encoding format")
    dimensions: Optional[int] = Field(None, description="Number of dimensions")


class CompletionRequest(BaseModel):
    """Request model for text completions."""
    prompt: str = Field(..., description="Text prompt")
    model: str = Field(..., description="Model identifier")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(None, ge=1, description="Maximum tokens to generate")
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    frequency_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0, description="Frequency penalty")
    presence_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0, description="Presence penalty")
    stop: Optional[Union[str, List[str]]] = Field(None, description="Stop sequences")