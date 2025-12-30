"""
LALS (Likhon Advanced Language System) - Qwen3-0.6B Endpoint
============================================================
This module implements a FastAPI-based inference endpoint for the Qwen3-0.6B model
in GGUF format, providing OpenAI-compatible chat completion API.

The LALS architecture is built on transformer-based frameworks optimized for
lower latency and efficient token processing, designed for versatile applications
including advanced natural language understanding, automated content generation,
and code assistance.
"""

import os
import asyncio
import logging
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from llama_cpp import Llama

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance
model: Optional[Llama] = None

# Model configuration
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "models", "qwen3-0.6b-q4_k_m.gguf")
)
MODEL_N_CTX = int(os.environ.get("MODEL_N_CTX", 2048))
MODEL_N_THREADS = int(os.environ.get("MODEL_N_THREADS", 4))
MODEL_TEMP = float(os.environ.get("MODEL_TEMP", 0.7))
MODEL_TOP_P = float(os.environ.get("MODEL_TOP_P", 0.95))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - load model on startup, unload on shutdown."""
    global model
    
    logger.info("Initializing LALS - Qwen3-0.6B Inference Engine...")
    logger.info(f"Model path: {MODEL_PATH}")
    logger.info(f"Context size: {MODEL_N_CTX}")
    logger.info(f"Threads: {MODEL_N_THREADS}")
    
    try:
        model = Llama(
            model_path=MODEL_PATH,
            n_ctx=MODEL_N_CTX,
            n_threads=MODEL_N_THREADS,
            n_gpu_layers=0,  # CPU inference
            verbose=False
        )
        logger.info("Model loaded successfully!")
        logger.info(f"LALS v1.0.0 - Qwen3-0.6B-Q4_K_M ready at http://0.0.0.0:8000")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise e
    
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down LALS...")
    del model
    logger.info("Model unloaded.")


# Create FastAPI application
app = FastAPI(
    title="LALS - Likhon Advanced Language System",
    description="A transformer-based language model optimized for lower latency and efficient token processing. "
                "Provides advanced NLU, automated content generation, and code assistance capabilities.",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for API
class Message(BaseModel):
    """Chat message model."""
    role: str = Field(..., description="Role of the message sender (user, assistant, system)")
    content: str = Field(..., description="Content of the message")


class ChatCompletionRequest(BaseModel):
    """Chat completion request model compatible with OpenAI API."""
    model: str = Field(default="qwen3-0.6b", description="Model identifier")
    messages: List[Message] = Field(..., description="List of chat messages")
    temperature: Optional[float] = Field(default=MODEL_TEMP, ge=0.0, le=2.0, description="Temperature for sampling")
    max_tokens: Optional[int] = Field(default=1024, ge=1, le=4096, description="Maximum tokens to generate")
    top_p: Optional[float] = Field(default=MODEL_TOP_P, ge=0.0, le=1.0, description="Top-p sampling parameter")
    stream: Optional[bool] = Field(default=False, description="Enable streaming response")


class Choice(BaseModel):
    """Chat completion choice."""
    index: int
    message: Message
    finish_reason: str


class Usage(BaseModel):
    """Token usage statistics."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """Chat completion response model compatible with OpenAI API."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage


def format_prompt(messages: List[Dict[str, str]]) -> str:
    """Format messages into Qwen3 chat template."""
    if not messages:
        return ""
    
    prompt_parts = []
    
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        
        if role == "system":
            prompt_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
        elif role == "user":
            prompt_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role == "assistant":
            prompt_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
    
    # Add assistant start token for generation
    prompt_parts.append("<|im_start|>assistant\n")
    
    return "\n".join(prompt_parts)


def extract_assistant_response(text: str) -> str:
    """Extract assistant response from generated text."""
    # Remove the prompt and any trailing special tokens
    if "<|im_start|>" in text:
        # Find the assistant section
        parts = text.split("<|im_start|>assistant")
        if len(parts) > 1:
            response = parts[1].strip()
            # Remove any trailing im_end tokens
            response = response.replace("<|im_end|>", "").strip()
            return response
    
    # If no special tokens, return as is
    return text.strip()


@app.get("/")
async def root():
    """Root endpoint - returns service information."""
    return {
        "service": "LALS - Likhon Advanced Language System",
        "version": "1.0.0",
        "model": "Qwen3-0.6B-Q4_K_M",
        "architecture": "Transformer-based, optimized for lower latency",
        "capabilities": ["NLU", "Content Generation", "Code Assistance"],
        "endpoints": {
            "GET /": "Service information",
            "POST /v1/chat/completions": "Chat completion (OpenAI-compatible)",
            "GET /health": "Health check"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model": "qwen3-0.6b-q4_k_m.gguf"}


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    """
    Create chat completion.
    
    This endpoint is compatible with OpenAI's chat completions API,
    allowing easy integration with existing applications and tools.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    try:
        # Convert messages to dict format
        messages_data = [msg.model_dump() for msg in request.messages]
        
        # Format prompt using Qwen3 chat template
        prompt = format_prompt(messages_data)
        
        # Generate response
        output = model(
            prompt=prompt,
            max_tokens=request.max_tokens or 1024,
            temperature=request.temperature or MODEL_TEMP,
            top_p=request.top_p or MODEL_TOP_P,
            echo=False,
            stream=False
        )
        
        # Extract generated text
        generated_text = output["choices"][0]["text"]
        
        # Extract assistant response
        assistant_content = extract_assistant_response(generated_text)
        
        # Calculate token usage
        prompt_tokens = output["usage"]["prompt_tokens"]
        completion_tokens = output["usage"]["completion_tokens"]
        total_tokens = prompt_tokens + completion_tokens
        
        # Create response
        response = ChatCompletionResponse(
            id=f"cmpl-{os.urandom(8).hex()}",
            created=int(asyncio.get_event_loop().time()),
            model=request.model,
            choices=[
                Choice(
                    index=0,
                    message=Message(role="assistant", content=assistant_content),
                    finish_reason="stop"
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens
            )
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )
