"""
Universal Routes Module
=======================
FastAPI routes for auto-detecting and routing requests.
Implements /v1/inference endpoint with protocol auto-detection.
"""

import logging
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..core.model import get_model_engine
from ..core.config import get_config
from ..protocols.openai import OpenAIProtocolHandler, OpenAIChatRequest
from ..protocols.anthropic import AnthropicProtocolHandler, AnthropicRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["universal"])


class ProtocolDetector:
    """Detects protocol from request headers, path, and body."""
    
    @staticmethod
    def from_headers(headers: Dict[str, str]) -> Optional[str]:
        """Detect protocol from request headers."""
        provider = headers.get("x-lals-provider") or headers.get("x-provider")
        if provider:
            return provider.lower()
        return None
    
    @staticmethod
    def from_path(path: str) -> Optional[str]:
        """Detect protocol from request path."""
        if path == "/v1/messages" or path == "/v1/complete":
            return "anthropic"
        if path == "/v1/chat/completions":
            return "openai"
        return None
    
    @staticmethod
    def from_body(body: Dict[str, Any]) -> str:
        """
        Auto-detect protocol from request body structure.
        
        Detection logic:
        1. Anthropic: has 'max_tokens' (required) and 'system' field
        2. OpenAI: has 'messages' array
        
        Args:
            body: Request body dictionary
        
        Returns:
            Detected protocol name
        """
        # Anthropic detection
        if "max_tokens" in body:
            # Anthropic uses max_tokens as required field
            # OpenAI uses it as optional
            if "system" in body:
                return "anthropic"
            
            # Check for Anthropic-specific fields
            anthropic_indicators = ["stop_sequences", "max_tokens_to_sample"]
            for indicator in anthropic_indicators:
                if indicator in body:
                    return "anthropic"
        
        # OpenAI detection - messages array
        if "messages" in body and isinstance(body["messages"], list):
            # Could be OpenAI (system in messages) or Anthropic (system as field)
            if isinstance(body["messages"], list) and len(body["messages"]) > 0:
                first_msg = body["messages"][0]
                if isinstance(first_msg, dict) and "role" in first_msg:
                    # Check if system message is separate (Anthropic style)
                    if "system" not in body:
                        return "openai"
                    # System is in body, could be either
                    # Default to OpenAI for messages array
                    return "openai"
        
        # Default fallback to OpenAI
        return "openai"


@router.post("/inference")
async def universal_inference(request: Request):
    """
    Universal inference endpoint with auto-detection.
    
    Automatically detects whether the request is in OpenAI or Anthropic
    format and routes to the appropriate handler.
    
    Headers for manual override:
    - x-lals-provider: "openai" | "anthropic"
    - x-provider: "openai" | "anthropic"
    """
    try:
        body = await request.json()
        headers = dict(request.headers)
        
        config = get_config()
        
        # Detect protocol
        detector = ProtocolDetector()
        
        # Priority 1: Header override
        protocol = detector.from_headers(headers)
        
        # Priority 2: Auto-detect from body
        if protocol is None:
            protocol = detector.from_body(body)
        
        logger.info(f"Detected protocol: {protocol}")
        
        # Route to appropriate handler
        if protocol == "anthropic":
            return await handle_anthropic(body)
        else:
            return await handle_openai(body)
    
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in universal inference: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def handle_openai(body: Dict[str, Any]) -> Dict[str, Any]:
    """Handle OpenAI-format request."""
    engine = get_model_engine()
    handler = OpenAIProtocolHandler(engine)
    
    # Parse request
    chat_request = OpenAIChatRequest(**body)
    
    # Validate
    handler.validate_request(chat_request)
    
    # Format prompt
    prompt = handler.format_prompt(chat_request)
    
    # Generate
    if chat_request.stream:
        async def generate_stream():
            tokens = engine.generate_streaming(
                prompt=prompt,
                max_tokens=chat_request.max_tokens,
                temperature=chat_request.temperature,
                top_p=chat_request.top_p,
                stop=chat_request.stop
            )
            input_tokens = len(prompt.split())
            async for chunk in handler.format_streaming_response(
                tokens, chat_request, input_tokens
            ):
                yield chunk
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )
    else:
        output = engine.generate(
            prompt=prompt,
            max_tokens=chat_request.max_tokens,
            temperature=chat_request.temperature,
            top_p=chat_request.top_p,
            stop=chat_request.stop
        )
        
        return handler.format_response(
            generated_text=output["text"],
            request=chat_request,
            input_tokens=output["prompt_tokens"],
            output_tokens=output["completion_tokens"],
            stop_reason=output.get("stop_reason", "stop")
        )


async def handle_anthropic(body: Dict[str, Any]) -> Dict[str, Any]:
    """Handle Anthropic-format request."""
    engine = get_model_engine()
    handler = AnthropicProtocolHandler(engine)
    
    # Parse request
    anthropic_request = AnthropicRequest(**body)
    
    # Validate
    handler.validate_request(anthropic_request)
    
    # Format prompt
    prompt = handler.format_prompt(anthropic_request)
    
    # Generate
    if anthropic_request.stream:
        async def generate_stream():
            tokens = engine.generate_streaming(
                prompt=prompt,
                max_tokens=anthropic_request.max_tokens,
                temperature=anthropic_request.temperature,
                top_p=anthropic_request.top_p,
                stop=anthropic_request.stop_sequences
            )
            input_tokens = len(prompt.split())
            async for chunk in handler.format_streaming_response(
                tokens, anthropic_request, input_tokens
            ):
                yield chunk
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )
    else:
        output = engine.generate(
            prompt=prompt,
            max_tokens=anthropic_request.max_tokens,
            temperature=anthropic_request.temperature,
            top_p=anthropic_request.top_p,
            stop=anthropic_request.stop_sequences
        )
        
        return handler.format_response(
            generated_text=output["text"],
            request=anthropic_request,
            input_tokens=output["prompt_tokens"],
            output_tokens=output["completion_tokens"],
            stop_reason=output.get("stop_reason", "end_turn")
        )
