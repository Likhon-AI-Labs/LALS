"""
Anthropic Routes Module
=======================
FastAPI routes for Anthropic-compatible endpoints.
Implements /v1/messages and /v1/complete endpoints.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, Query
from fastapi.responses import StreamingResponse

from ..core.model import get_model_engine
from ..protocols.anthropic import (
    AnthropicRequest,
    AnthropicProtocolHandler
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="", tags=["anthropic"])


def get_handler():
    """Get Anthropic protocol handler with model engine."""
    engine = get_model_engine()
    return AnthropicProtocolHandler(engine)


@router.post("/v1/messages")
async def create_message(
    request: Request,
    model: Optional[str] = Query(default="qwen3-0.6b", description="Model ID")
):
    """
    Anthropic Messages API endpoint.
    
    Creates a message completion for the provided conversation.
    This is the primary endpoint for Claude API compatibility.
    Supports both synchronous and streaming responses.
    """
    try:
        body = await request.json()
        
        # Add model to body if not present
        if "model" not in body:
            body["model"] = model
        
        # Parse request
        anthropic_request = AnthropicRequest(**body)
        handler = get_handler()
        
        # Validate request
        handler.validate_request(anthropic_request)
        
        # Format prompt
        prompt = handler.format_prompt(anthropic_request)
        
        # Generate response
        if anthropic_request.stream:
            async def generate_stream():
                tokens = get_model_engine().generate_streaming(
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
            output = get_model_engine().generate(
                prompt=prompt,
                max_tokens=anthropic_request.max_tokens,
                temperature=anthropic_request.temperature,
                top_p=anthropic_request.top_p,
                stop=anthropic_request.stop_sequences
            )
            
            response = handler.format_response(
                generated_text=output["text"],
                request=anthropic_request,
                input_tokens=output["prompt_tokens"],
                output_tokens=output["completion_tokens"],
                stop_reason=output.get("stop_reason", "end_turn")
            )
            
            return response
    
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in message creation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/complete")
async def create_completion(
    request: Request,
    model: Optional[str] = Query(default="qwen3-0.6b", description="Model ID")
):
    """
    Anthropic Legacy Completions API endpoint.
    
    Creates a completion for the provided prompt.
    This is the legacy endpoint for backwards compatibility.
    """
    try:
        body = await request.json()
        
        # Add model to body if not present
        if "model" not in body:
            body["model"] = model
        
        # Convert to Anthropic request format
        # Anthropic uses "max_tokens_to_sample" in legacy format
        if "max_tokens_to_sample" in body:
            body["max_tokens"] = body.pop("max_tokens_to_sample")
        
        # Parse request
        anthropic_request = AnthropicRequest(**body)
        handler = get_handler()
        
        # Validate request
        handler.validate_request(anthropic_request)
        
        # Format prompt
        prompt = handler.format_prompt(anthropic_request)
        
        # Generate response
        if anthropic_request.stream:
            async def generate_stream():
                tokens = get_model_engine().generate_streaming(
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
            output = get_model_engine().generate(
                prompt=prompt,
                max_tokens=anthropic_request.max_tokens,
                temperature=anthropic_request.temperature,
                top_p=anthropic_request.top_p,
                stop=anthropic_request.stop_sequences
            )
            
            response = handler.format_response(
                generated_text=output["text"],
                request=anthropic_request,
                input_tokens=output["prompt_tokens"],
                output_tokens=output["completion_tokens"],
                stop_reason=output.get("stop_reason", "end_turn")
            )
            
            return response
    
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))
