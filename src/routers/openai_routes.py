"""
OpenAI Routes Module
====================
FastAPI routes for OpenAI-compatible endpoints.
Implements /v1/chat/completions, /v1/completions, and /v1/models.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, Query
from fastapi.responses import StreamingResponse

from ..core.model import get_model_engine
from ..protocols.openai import (
    OpenAIChatRequest,
    OpenAICompletionRequest,
    OpenAIProtocolHandler,
    convert_to_openai_request
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="", tags=["openai"])


def get_handler():
    """Get OpenAI protocol handler with model engine."""
    engine = get_model_engine()
    return OpenAIProtocolHandler(engine)


@router.post("/v1/chat/completions")
async def create_chat_completion(
    request: Request,
    model: Optional[str] = Query(default="qwen3-0.6b", description="Model ID")
):
    """
    OpenAI Chat Completions API endpoint.
    
    Creates a chat completion for the provided messages.
    Supports both synchronous and streaming responses.
    """
    try:
        body = await request.json()
        
        # Add model to body if not present
        if "model" not in body:
            body["model"] = model
        
        # Parse request
        chat_request = OpenAIChatRequest(**body)
        handler = get_handler()
        
        # Validate request
        handler.validate_request(chat_request)
        
        # Format prompt
        prompt = handler.format_prompt(chat_request)
        
        # Generate response
        if chat_request.stream:
            async def generate_stream():
                tokens = get_model_engine().generate_streaming(
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
            output = get_model_engine().generate(
                prompt=prompt,
                max_tokens=chat_request.max_tokens,
                temperature=chat_request.temperature,
                top_p=chat_request.top_p,
                stop=chat_request.stop
            )
            
            response = handler.format_response(
                generated_text=output["text"],
                request=chat_request,
                input_tokens=output["prompt_tokens"],
                output_tokens=output["completion_tokens"],
                stop_reason=output.get("stop_reason", "stop")
            )
            
            return response
    
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in chat completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/completions")
async def create_completion(
    request: Request,
    model: Optional[str] = Query(default="qwen3-0.6b", description="Model ID")
):
    """
    OpenAI Legacy Completions API endpoint.
    
    Creates a text completion for the provided prompt.
    Supports both synchronous and streaming responses.
    """
    try:
        body = await request.json()
        
        # Add model to body if not present
        if "model" not in body:
            body["model"] = model
        
        # Parse request
        completion_request = OpenAICompletionRequest(**body)
        handler = get_handler()
        
        # Format prompt
        prompt = completion_request.prompt or ""
        
        # Generate response
        if completion_request.stream:
            async def generate_stream():
                tokens = get_model_engine().generate_streaming(
                    prompt=prompt,
                    max_tokens=completion_request.max_tokens,
                    temperature=completion_request.temperature,
                    top_p=completion_request.top_p,
                    stop=completion_request.stop
                )
                input_tokens = len(prompt.split())
                async for chunk in handler.format_streaming_response(
                    tokens, completion_request, input_tokens
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
                max_tokens=completion_request.max_tokens,
                temperature=completion_request.temperature,
                top_p=completion_request.top_p,
                stop=completion_request.stop
            )
            
            # Convert to completion format
            completion_output = handler.format_legacy_completion_response(
                generated_text=output["text"],
                request=completion_request,
                input_tokens=output["prompt_tokens"],
                output_tokens=output["completion_tokens"],
                stop_reason=output.get("stop_reason", "stop")
            )
            
            return completion_output
    
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/v1/models")
async def list_models():
    """
    OpenAI Models API endpoint.
    
    Returns a list of available models.
    """
    try:
        handler = get_handler()
        return handler.format_models_response()
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    """
    OpenAI Model retrieval endpoint.
    
    Returns information about a specific model.
    """
    handler = get_handler()
    
    model = next(
        (m for m in handler._models if m["id"] == model_id),
        None
    )
    
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return model
