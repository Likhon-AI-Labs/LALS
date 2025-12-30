"""
OpenAI Protocol Handler
=======================
Handles OpenAI API protocol format.
Supports Chat Completions, Legacy Completions, and Models endpoints.
"""

import json
import uuid
import logging
from typing import List, Dict, Any, Optional, AsyncIterator

from pydantic import BaseModel, Field, field_validator

from .base import BaseProtocolHandler, ProtocolRequest

logger = logging.getLogger(__name__)


class OpenAIMessage(BaseModel):
    """OpenAI message format."""
    role: str = Field(..., pattern="^(system|user|assistant|function)$")
    content: str
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None


class OpenAIChatRequest(BaseModel):
    """OpenAI Chat Completions API request format."""
    
    model: str = Field(..., description="Model identifier")
    messages: List[OpenAIMessage] = Field(
        ...,
        min_length=1,
        description="Conversation messages"
    )
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature"
    )
    top_p: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Top-p sampling"
    )
    n: Optional[int] = Field(
        default=1,
        ge=1,
        le=10,
        description="Number of completions"
    )
    stream: Optional[bool] = Field(
        default=False,
        description="Enable streaming response"
    )
    stop: Optional[List[str]] = Field(
        default=None,
        description="Stop sequences"
    )
    max_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum tokens to generate"
    )
    presence_penalty: Optional[float] = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Presence penalty"
    )
    frequency_penalty: Optional[float] = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Frequency penalty"
    )
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    
    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v):
        """Ensure messages are not empty."""
        if not v:
            raise ValueError("messages array cannot be empty")
        return v


class OpenAICompletionRequest(BaseModel):
    """OpenAI Legacy Completions API request format."""
    
    model: str = Field(..., description="Model identifier")
    prompt: Optional[str] = Field(
        default=None,
        description="Prompt text"
    )
    suffix: Optional[str] = Field(
        default=None,
        description="Suffix to append"
    )
    temperature: Optional[float] = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature"
    )
    top_p: Optional[float] = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Top-p sampling"
    )
    n: Optional[int] = Field(
        default=1,
        ge=1,
        le=10,
        description="Number of completions"
    )
    stream: Optional[bool] = Field(
        default=False,
        description="Enable streaming"
    )
    stop: Optional[List[str]] = Field(
        default=None,
        description="Stop sequences"
    )
    max_tokens: Optional[int] = Field(
        default=16,
        ge=1,
        description="Max tokens to generate"
    )
    presence_penalty: Optional[float] = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Presence penalty"
    )
    frequency_penalty: Optional[float] = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Frequency penalty"
    )
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None


class OpenAIChatResponse(BaseModel):
    """OpenAI Chat Completions response format."""
    
    id: str = Field(..., description="Completion ID")
    object: str = Field(default="chat.completion", description="Object type")
    created: int = Field(..., description="Creation timestamp")
    model: str = Field(..., description="Model identifier")
    choices: List[Dict[str, Any]] = Field(
        ...,
        description="Completion choices"
    )
    usage: Dict[str, int] = Field(
        ...,
        description="Token usage"
    )
    system_fingerprint: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "chatcmpl-abc123",
                "object": "chat.completion",
                "created": 1677858242,
                "model": "qwen3-0.6b",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello!"
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 15,
                    "total_tokens": 25
                }
            }
        }


class OpenAICompletionResponse(BaseModel):
    """OpenAI Legacy Completions response format."""
    
    id: str = Field(..., description="Completion ID")
    object: str = Field(default="text_completion", description="Object type")
    created: int = Field(..., description="Creation timestamp")
    model: str = Field(..., description="Model identifier")
    choices: List[Dict[str, Any]] = Field(
        ...,
        description="Completion choices"
    )
    usage: Dict[str, int] = Field(
        ...,
        description="Token usage"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "cmpl-abc123",
                "object": "text_completion",
                "created": 1677858242,
                "model": "qwen3-0.6b",
                "choices": [{
                    "text": "Hello!",
                    "index": 0,
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 15,
                    "total_tokens": 25
                }
            }
        }


class OpenAIModelsResponse(BaseModel):
    """OpenAI Models list response format."""
    
    object: str = Field(default="list", description="Object type")
    data: List[Dict[str, Any]] = Field(
        ...,
        description="Model list"
    )


class OpenAIProtocolHandler(BaseProtocolHandler):
    """
    Handler for OpenAI API protocol.
    
    Implements the BaseProtocolHandler interface for OpenAI format.
    Supports Chat Completions (/v1/chat/completions) and
    Legacy Completions (/v1/completions).
    """
    
    def __init__(self, model_engine):
        super().__init__(model_engine)
        self.protocol_name = "openai"
        
        # Available models
        self._models = [
            {
                "id": "qwen3-0.6b",
                "object": "model",
                "created": 1677610602,
                "owned_by": "likhon-ai-labs",
                "permission": [],
                "root": "qwen3-0.6b",
                "parent": None,
            }
        ]
    
    @property
    def protocol_name(self) -> str:
        return "openai"
    
    def validate_request(self, request: OpenAIChatRequest) -> bool:
        """Validate OpenAI API request."""
        if not request.messages:
            raise ValueError("messages array cannot be empty")
        
        # Check for at least one user message
        has_user_message = any(
            msg.role == "user" 
            for msg in request.messages
        )
        if not has_user_message:
            raise ValueError("At least one user message is required")
        
        return True
    
    def format_prompt(self, request: OpenAIChatRequest) -> str:
        """
        Convert OpenAI messages to model prompt.
        
        Uses Qwen3 chat template format.
        """
        prompt_parts = []
        
        for msg in request.messages:
            if msg.role == "system":
                prompt_parts.append(f"<|im_start|>system\n{msg.content}<|im_end|>")
            elif msg.role == "user":
                prompt_parts.append(f"<|im_start|>user\n{msg.content}<|im_end|>")
            elif msg.role == "assistant":
                prompt_parts.append(f"<|im_start|>assistant\n{msg.content}<|im_end|>")
        
        # Add assistant prompt for generation
        prompt_parts.append("<|im_start|>assistant\n")
        
        return "\n".join(prompt_parts)
    
    def format_response(
        self,
        generated_text: str,
        request: OpenAIChatRequest,
        input_tokens: int,
        output_tokens: int,
        stop_reason: str = "stop"
    ) -> Dict[str, Any]:
        """
        Format response in OpenAI Chat Completions format.
        """
        # Clean up the generated text
        content = generated_text.strip()
        
        # Remove any trailing prompt artifacts
        content = content.replace("<|im_end|>", "").strip()
        
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(__import__("time").time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content
                    },
                    "finish_reason": stop_reason
                }
            ],
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            }
        }
    
    def format_legacy_completion_response(
        self,
        generated_text: str,
        request: OpenAICompletionRequest,
        input_tokens: int,
        output_tokens: int,
        stop_reason: str = "stop"
    ) -> Dict[str, Any]:
        """Format response in OpenAI Legacy Completions format."""
        return {
            "id": f"cmpl-{uuid.uuid4().hex[:24]}",
            "object": "text_completion",
            "created": int(__import__("time").time()),
            "model": request.model,
            "choices": [
                {
                    "text": generated_text.strip(),
                    "index": 0,
                    "finish_reason": stop_reason
                }
            ],
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            }
        }
    
    async def format_streaming_response(
        self,
        tokens: AsyncIterator[str],
        request: OpenAIChatRequest,
        input_tokens: int
    ) -> AsyncIterator[str]:
        """
        Format streaming response in OpenAI SSE format.
        
        Yields SSE chunks matching OpenAI's streaming protocol.
        """
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(__import__("time").time())
        output_tokens = 0
        
        # Stream tokens as delta updates
        async for token in tokens:
            output_tokens += 1
            
            # Escape special characters for SSE
            escaped_token = (
                token
                .replace("\\", "\\\\")
                .replace('"', '\\"')
                .replace("\n", "\\n")
                .replace("\r", "\\r")
            )
            
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": escaped_token
                        },
                        "finish_reason": None
                    }
                ]
            }
            
            yield f"data: {json.dumps(chunk)}\n\n"
        
        # Final chunk with usage info
        final_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            }
        }
        
        yield f"data: {json.dumps(final_chunk)}\n\n"
        
        # Signal end of stream
        yield "data: [DONE]\n\n"
    
    def format_models_response(self) -> Dict[str, Any]:
        """Format response for models list endpoint."""
        return {
            "object": "list",
            "data": self._models
        }
    
    def detect_protocol(self, request_body: Dict[str, Any]) -> Optional[str]:
        """
        Detect if request body matches OpenAI protocol.
        
        Checks for:
        - messages array (OpenAI chat format)
        - prompt field (OpenAI legacy format)
        - No Anthropic-specific fields
        """
        # Check for messages array (OpenAI chat)
        if "messages" in request_body and isinstance(request_body["messages"], list):
            # Exclude Anthropic-style
            if "system" in request_body and "max_tokens" in request_body:
                # Could be either, but check for other Anthropic indicators
                if "stop_sequences" in request_body:
                    return None  # Likely Anthropic
            return self.protocol_name
        
        # Check for prompt field (OpenAI legacy)
        if "prompt" in request_body:
            return self.protocol_name
        
        return None
    
    def convert_chat_to_completion(self, request: OpenAIChatRequest) -> OpenAICompletionRequest:
        """Convert chat request to legacy completion format."""
        # Extract text from messages
        prompt_text = "\n".join(
            f"{msg.role}: {msg.content}"
            for msg in request.messages
        )
        
        return OpenAICompletionRequest(
            model=request.model,
            prompt=prompt_text,
            temperature=request.temperature,
            top_p=request.top_p,
            n=request.n,
            stream=request.stream,
            stop=request.stop,
            max_tokens=request.max_tokens,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
            logit_bias=request.logit_bias,
            user=request.user
        )


def convert_to_openai_request(request: ProtocolRequest) -> OpenAIChatRequest:
    """
    Convert generic protocol request to OpenAI-specific request.
    
    Args:
        request: Generic protocol request
    
    Returns:
        OpenAI-specific chat request object
    """
    # Convert messages to OpenAI format
    messages = [
        OpenAIMessage(role=msg.role, content=msg.content)
        for msg in request.messages
    ]
    
    return OpenAIChatRequest(
        model=request.model,
        messages=messages,
        temperature=request.temperature,
        top_p=request.top_p,
        stream=request.stream,
        stop=request.stop,
        max_tokens=request.max_tokens
    )
