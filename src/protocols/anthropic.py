"""
Anthropic Protocol Handler
==========================
Handles Anthropic Claude API protocol format.
Supports both Messages API and legacy Completions API.
"""

import json
import uuid
import time
import logging
from typing import List, Dict, Any, Optional, AsyncIterator

from pydantic import BaseModel, Field, field_validator

from .base import BaseProtocolHandler, ProtocolRequest

logger = logging.getLogger(__name__)


class AnthropicMessage(BaseModel):
    """Anthropic message format."""
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class AnthropicContentBlock(BaseModel):
    """Anthropic content block."""
    type: str = Field(default="text")
    text: str


class AnthropicRequest(BaseModel):
    """Anthropic Messages API request format."""
    
    model: str = Field(..., description="Model identifier")
    max_tokens: int = Field(
        ...,
        ge=1,
        le=8192,
        description="Maximum tokens to generate"
    )
    messages: List[AnthropicMessage] = Field(
        ...,
        min_length=1,
        description="Conversation messages"
    )
    system: Optional[str] = Field(
        default=None,
        description="System prompt"
    )
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Sampling temperature"
    )
    top_p: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Top-p sampling"
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=0,
        description="Top-k sampling"
    )
    stream: Optional[bool] = Field(
        default=False,
        description="Enable streaming response"
    )
    stop_sequences: Optional[List[str]] = Field(
        default=None,
        description="Custom stop sequences"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata"
    )
    
    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v):
        """Ensure messages are not empty."""
        if not v:
            raise ValueError("messages array cannot be empty")
        return v


class AnthropicResponse(BaseModel):
    """Anthropic Messages API response format."""
    
    id: str = Field(..., description="Unique message ID")
    type: str = Field(default="message", description="Response type")
    role: str = Field(default="assistant", description="Message role")
    content: List[AnthropicContentBlock] = Field(
        ...,
        description="Response content blocks"
    )
    model: str = Field(..., description="Model identifier")
    stop_reason: Optional[str] = Field(
        default=None,
        description="Reason for stopping"
    )
    stop_sequence: Optional[str] = Field(
        default=None,
        description="Stop sequence triggered"
    )
    usage: Dict[str, int] = Field(
        ...,
        description="Token usage statistics"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "msg_01abc123",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "Hello! How can I help?"}],
                "model": "qwen3-0.6b",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 15}
            }
        }


class AnthropicProtocolHandler(BaseProtocolHandler):
    """
    Handler for Anthropic Claude API protocol.
    
    Implements the BaseProtocolHandler interface for Anthropic format.
    Supports Messages API (/v1/messages) and legacy Completions API.
    """
    
    def __init__(self, model_engine):
        super().__init__(model_engine)
        self.protocol_name = "anthropic"
    
    @property
    def protocol_name(self) -> str:
        return "anthropic"
    
    def validate_request(self, request: AnthropicRequest) -> bool:
        """Validate Anthropic API request."""
        if not request.messages:
            raise ValueError("messages array cannot be empty")
        
        if request.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        
        # Check for user message
        has_user_message = any(
            msg.role == "user" 
            for msg in request.messages
        )
        if not has_user_message:
            raise ValueError("At least one user message is required")
        
        return True
    
    def format_prompt(self, request: AnthropicRequest) -> str:
        """
        Convert Anthropic messages to model prompt.
        
        Anthropic format uses "Human:" and "Assistant:" prefixes.
        System messages are prepended at the beginning.
        """
        prompt_parts = []
        
        # Add system message if present
        if request.system:
            prompt_parts.append(f"System: {request.system}\n")
        
        # Add conversation history
        for msg in request.messages:
            if msg.role == "user":
                prompt_parts.append(f"Human: {msg.content}\n")
            elif msg.role == "assistant":
                prompt_parts.append(f"Assistant: {msg.content}\n")
        
        # Add assistant prompt for generation
        prompt_parts.append("Assistant:")
        
        return "\n".join(prompt_parts)
    
    def format_response(
        self,
        generated_text: str,
        request: AnthropicRequest,
        input_tokens: int,
        output_tokens: int,
        stop_reason: str = "end_turn"
    ) -> Dict[str, Any]:
        """
        Format response in Anthropic API format.
        
        Returns a dictionary matching Anthropic's Messages API response.
        """
        # Clean up the generated text
        content = generated_text.strip()
        
        # Remove any trailing prompt artifacts
        if content.startswith("Human:") or content.startswith("System:"):
            content = ""
        
        return {
            "id": f"msg_{uuid.uuid4().hex[:24]}",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": content
                }
            ],
            "model": request.model,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens
            }
        }
    
    async def format_streaming_response(
        self,
        tokens: AsyncIterator[str],
        request: AnthropicRequest,
        input_tokens: int
    ) -> AsyncIterator[str]:
        """
        Format streaming response in Anthropic SSE format.
        
        Yields SSE events matching Anthropic's streaming protocol:
        - message_start
        - content_block_start
        - content_block_delta
        - content_block_stop
        - message_delta
        - message_stop
        """
        message_id = f"msg_{uuid.uuid4().hex[:24]}"
        output_tokens = 0
        
        # Event: message_start
        message_start = {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "model": request.model
            }
        }
        yield f"event: message_start\ndata: {json.dumps(message_start)}\n\n"
        
        # Event: content_block_start
        content_block_start = {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "text",
                "text": ""
            }
        }
        yield f"event: content_block_start\ndata: {json.dumps(content_block_start)}\n\n"
        
        # Stream tokens as content_block_delta events
        async for token in tokens:
            output_tokens += 1
            
            # Escape special characters for JSON
            escaped_token = (
                token
                .replace("\\", "\\\\")
                .replace('"', '\\"')
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t")
            )
            
            delta_event = {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "text_delta",
                    "text": escaped_token
                }
            }
            yield f"event: content_block_delta\ndata: {json.dumps(delta_event)}\n\n"
        
        # Event: content_block_stop
        yield f"event: content_block_stop\ndata: {{\"type\": \"content_block_stop\", \"index\": 0}}\n\n"
        
        # Event: message_delta (usage info)
        message_delta = {
            "type": "message_delta",
            "delta": {
                "stop_reason": "end_turn",
                "stop_sequence": None
            },
            "usage": {
                "output_tokens": output_tokens
            }
        }
        yield f"event: message_delta\ndata: {json.dumps(message_delta)}\n\n"
        
        # Event: message_stop
        yield f"event: message_stop\ndata: {{\"type\": \"message_stop\"}}\n\n"
    
    def detect_protocol(self, request_body: Dict[str, Any]) -> Optional[str]:
        """
        Detect if request body matches Anthropic protocol.
        
        Checks for:
        - Required max_tokens field
        - Optional system field
        - Anthropic-specific field names
        """
        # Check for required max_tokens (Anthropic uses max_tokens, OpenAI uses max_tokens too)
        if "max_tokens" not in request_body:
            return None
        
        # Check for Anthropic-specific fields
        anthropic_indicators = [
            "stop_sequences",  # Anthropic uses this, OpenAI uses stop
            "system",  # Anthropic has separate system field
            "max_tokens_to_sample",  # Legacy Anthropic field
            "anthropic_version",  # Anthropic API version header
        ]
        
        for indicator in anthropic_indicators:
            if indicator in request_body:
                return self.protocol_name
        
        # Check for Anthropic-style structure
        # Anthropic typically has messages with max_tokens as required
        if (
            isinstance(request_body.get("messages"), list) and
            request_body.get("messages") and
            isinstance(request_body.get("max_tokens"), int)
        ):
            # Could be either, but lean toward Anthropic if system is present
            if "system" in request_body:
                return self.protocol_name
        
        return None


def convert_to_anthropic_request(request: ProtocolRequest) -> AnthropicRequest:
    """
    Convert generic protocol request to Anthropic-specific request.
    
    Args:
        request: Generic protocol request
    
    Returns:
        Anthropic-specific request object
    """
    # Extract system message from messages if present
    system = request.system
    anthropic_messages = []
    
    for msg in request.messages:
        if msg.role == "system":
            system = msg.content
        else:
            anthropic_messages.append(
                AnthropicMessage(role=msg.role, content=msg.content)
            )
    
    return AnthropicRequest(
        model=request.model,
        max_tokens=request.max_tokens or 1024,
        messages=anthropic_messages,
        system=system,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        stream=request.stream,
        stop_sequences=request.stop
    )
