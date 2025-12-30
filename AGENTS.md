# LALS Blueprint: Multi-API Gateway Architecture

## Executive Summary

This blueprint outlines the comprehensive plan to enhance LALS (Likhon Advanced Language System) by implementing dual API compatibility layers - supporting both Anthropic Claude API and OpenAI API formats - while maintaining the existing FastAPI/Wasmer Edge infrastructure.

---

## Project Overview

**Objective**: Transform LALS from a single OpenAI-compatible endpoint into a multi-protocol API gateway that serves both OpenAI and Anthropic API specifications.

**Current State**:
- ✅ OpenAI-compatible `/v1/chat/completions` endpoint
- ✅ Qwen3-0.6B model (Q4_K_M quantized)
- ✅ FastAPI + Wasmer Edge deployment
- ✅ 2048 token context window

**Target State**:
- ✅ OpenAI API compatibility (existing)
- 🆕 Anthropic Claude API compatibility
- 🆕 Unified request routing layer
- 🆕 Protocol auto-detection
- 🆕 Enhanced streaming support for both protocols

---

## Architecture Design

### 1. API Gateway Layer

```
┌─────────────────────────────────────────────────────────┐
│                    LALS API Gateway                     │
│                  (lals-ai.wasmer.app)                   │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │  Router Layer │
                    │ (Auto-detect) │
                    └───────────────┘
                            │
            ┌───────────────┴───────────────┐
            │                               │
            ▼                               ▼
    ┌──────────────┐              ┌──────────────┐
    │   OpenAI     │              │  Anthropic   │
    │   Protocol   │              │   Protocol   │
    │   Handler    │              │   Handler    │
    └──────────────┘              └──────────────┘
            │                               │
            └───────────────┬───────────────┘
                            ▼
                ┌───────────────────────┐
                │   LALS Core Engine    │
                │  (llama-cpp-python)   │
                │   Qwen3-0.6B Model    │
                └───────────────────────┘
```

### 2. Endpoint Structure

```
/                                    → Service info (existing)
/health                              → Health check (existing)

# OpenAI Compatible Endpoints
/v1/chat/completions                 → OpenAI chat (existing)
/v1/models                           → List models (new)
/v1/completions                      → Legacy completions (new)

# Anthropic Compatible Endpoints  
/v1/messages                         → Claude messages API (new)
/v1/complete                         → Claude completions (new)

# Universal Endpoint (Auto-detect)
/v1/inference                        → Smart routing (new)
```

---

## Implementation Plan

### Phase 1: Core Infrastructure (Week 1)

#### 1.1 Project Restructuring

```
lals/
├── src/
│   ├── main.py                      # Main FastAPI app
│   ├── core/
│   │   ├── __init__.py
│   │   ├── model.py                 # Model loading & inference
│   │   └── config.py                # Configuration management
│   ├── protocols/
│   │   ├── __init__.py
│   │   ├── openai.py                # OpenAI protocol handler
│   │   ├── anthropic.py             # Anthropic protocol handler
│   │   └── base.py                  # Base protocol interface
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── openai_routes.py         # OpenAI endpoints
│   │   ├── anthropic_routes.py      # Anthropic endpoints
│   │   └── universal_routes.py      # Auto-detect routing
│   └── utils/
│       ├── __init__.py
│       ├── formatters.py            # Response formatters
│       ├── validators.py            # Request validators
│       └── streaming.py             # SSE streaming helpers
├── models/
│   └── qwen3-0.6b-q4_k_m.gguf
├── tests/
│   ├── test_openai.py
│   ├── test_anthropic.py
│   └── test_comparison.py
├── requirements.txt
├── wasmer.toml
└── README.md
```

#### 1.2 Dependencies Update

```txt
# requirements.txt
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
llama-cpp-python>=0.2.20
pydantic>=2.5.0
pydantic-settings>=2.1.0
requests>=2.31.0
aiohttp>=3.9.0
sse-starlette>=1.8.0
python-multipart>=0.0.6
```

---

### Phase 2: Anthropic API Implementation (Week 2)

#### 2.1 Anthropic Message Protocol

**Request Format**:
```json
{
  "model": "qwen3-0.6b",
  "max_tokens": 1024,
  "messages": [
    {
      "role": "user",
      "content": "Hello, Claude!"
    }
  ],
  "system": "You are a helpful AI assistant.",
  "temperature": 0.7,
  "top_p": 0.95,
  "top_k": 40,
  "stream": false,
  "stop_sequences": ["\n\nHuman:", "\n\nAssistant:"],
  "metadata": {
    "user_id": "user_123"
  }
}
```

**Response Format**:
```json
{
  "id": "msg_01XYZ123",
  "type": "message",
  "role": "assistant",
  "content": [
    {
      "type": "text",
      "text": "Hello! How can I assist you today?"
    }
  ],
  "model": "qwen3-0.6b",
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "usage": {
    "input_tokens": 12,
    "output_tokens": 25
  }
}
```

**Streaming Response Format**:
```
event: message_start
data: {"type": "message_start", "message": {"id": "msg_123", ...}}

event: content_block_start
data: {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}

event: content_block_delta
data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}}

event: content_block_delta
data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "!"}}

event: content_block_stop
data: {"type": "content_block_stop", "index": 0}

event: message_delta
data: {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": null}, "usage": {"output_tokens": 25}}

event: message_stop
data: {"type": "message_stop"}
```

#### 2.2 Implementation: `src/protocols/anthropic.py`

```python
from typing import List, Dict, Any, Optional, AsyncIterator
from pydantic import BaseModel, Field
from .base import BaseProtocolHandler
import time
import uuid

class AnthropicMessage(BaseModel):
    role: str
    content: str

class AnthropicRequest(BaseModel):
    model: str
    max_tokens: int = 1024
    messages: List[AnthropicMessage]
    system: Optional[str] = None
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = 40
    stream: Optional[bool] = False
    stop_sequences: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

class AnthropicProtocolHandler(BaseProtocolHandler):
    """Handler for Anthropic Claude API protocol"""
    
    def __init__(self, model_engine):
        self.model = model_engine
    
    def validate_request(self, request: AnthropicRequest) -> bool:
        """Validate Anthropic API request"""
        if not request.messages:
            raise ValueError("Messages array cannot be empty")
        if request.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        return True
    
    def format_prompt(self, request: AnthropicRequest) -> str:
        """Convert Anthropic messages to model prompt"""
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
        
        # Add assistant prompt
        prompt_parts.append("Assistant:")
        
        return "\n".join(prompt_parts)
    
    def format_response(self, 
                       generated_text: str, 
                       request: AnthropicRequest,
                       input_tokens: int,
                       output_tokens: int) -> Dict[str, Any]:
        """Format response in Anthropic format"""
        return {
            "id": f"msg_{uuid.uuid4().hex[:24]}",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": generated_text.strip()
                }
            ],
            "model": request.model,
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens
            }
        }
    
    async def format_streaming_response(self,
                                       tokens: AsyncIterator[str],
                                       request: AnthropicRequest,
                                       input_tokens: int) -> AsyncIterator[str]:
        """Format streaming response in Anthropic SSE format"""
        message_id = f"msg_{uuid.uuid4().hex[:24]}"
        
        # Message start event
        yield f"event: message_start\n"
        yield f'data: {{"type": "message_start", "message": {{"id": "{message_id}", "type": "message", "role": "assistant", "model": "{request.model}"}}}}\n\n'
        
        # Content block start
        yield f"event: content_block_start\n"
        yield f'data: {{"type": "content_block_start", "index": 0, "content_block": {{"type": "text", "text": ""}}}}\n\n'
        
        # Stream tokens
        output_tokens = 0
        async for token in tokens:
            output_tokens += 1
            # Escape JSON special characters
            escaped_token = token.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
            yield f"event: content_block_delta\n"
            yield f'data: {{"type": "content_block_delta", "index": 0, "delta": {{"type": "text_delta", "text": "{escaped_token}"}}}}\n\n'
        
        # Content block stop
        yield f"event: content_block_stop\n"
        yield f'data: {{"type": "content_block_stop", "index": 0}}\n\n'
        
        # Message delta (usage info)
        yield f"event: message_delta\n"
        yield f'data: {{"type": "message_delta", "delta": {{"stop_reason": "end_turn", "stop_sequence": null}}, "usage": {{"output_tokens": {output_tokens}}}}}\n\n'
        
        # Message stop
        yield f"event: message_stop\n"
        yield f'data: {{"type": "message_stop"}}\n\n'
```

#### 2.3 Implementation: `src/routers/anthropic_routes.py`

```python
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from ..protocols.anthropic import AnthropicProtocolHandler, AnthropicRequest
from ..core.model import get_model_engine

router = APIRouter(prefix="/v1", tags=["anthropic"])

@router.post("/messages")
async def create_message(request: AnthropicRequest):
    """Anthropic Messages API endpoint"""
    try:
        model_engine = get_model_engine()
        handler = AnthropicProtocolHandler(model_engine)
        
        # Validate request
        handler.validate_request(request)
        
        # Format prompt
        prompt = handler.format_prompt(request)
        
        # Generate response
        if request.stream:
            async def generate():
                tokens = model_engine.generate_streaming(
                    prompt=prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    stop=request.stop_sequences
                )
                async for chunk in handler.format_streaming_response(
                    tokens, request, input_tokens=len(prompt.split())
                ):
                    yield chunk
            
            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                }
            )
        else:
            output = model_engine.generate(
                prompt=prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=request.stop_sequences
            )
            
            response = handler.format_response(
                generated_text=output["text"],
                request=request,
                input_tokens=output["prompt_tokens"],
                output_tokens=output["completion_tokens"]
            )
            
            return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

---

### Phase 3: Enhanced OpenAI Implementation (Week 2)

#### 3.1 Additional OpenAI Endpoints

**`/v1/models` - List Available Models**:
```python
@router.get("/models")
async def list_models():
    """OpenAI-compatible models list"""
    return {
        "object": "list",
        "data": [
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
    }
```

---

### Phase 4: Universal Smart Router (Week 3)

#### 4.1 Protocol Auto-Detection: `src/routers/universal_routes.py`

```python
from fastapi import APIRouter, Request, HTTPException
from ..protocols.openai import OpenAIProtocolHandler
from ..protocols.anthropic import AnthropicProtocolHandler

router = APIRouter(prefix="/v1", tags=["universal"])

@router.post("/inference")
async def universal_inference(request: Request):
    """
    Universal endpoint with automatic protocol detection
    Supports both OpenAI and Anthropic request formats
    """
    try:
        body = await request.json()
        
        # Detect protocol based on request structure
        if "messages" in body and isinstance(body.get("messages"), list):
            # Check if it's Anthropic format (has max_tokens as required)
            if "max_tokens" in body and "model" in body:
                # Could be either, check for Anthropic-specific fields
                if "system" in body or body.get("messages", [{}])[0].get("content") != body.get("messages", [{}])[0]:
                    # Likely Anthropic format
                    from ..protocols.anthropic import AnthropicRequest
                    anthropic_request = AnthropicRequest(**body)
                    handler = AnthropicProtocolHandler(get_model_engine())
                    # Process with Anthropic handler
                    return await process_anthropic(anthropic_request, handler)
            
            # Default to OpenAI format
            from ..protocols.openai import OpenAIChatRequest
            openai_request = OpenAIChatRequest(**body)
            handler = OpenAIProtocolHandler(get_model_engine())
            return await process_openai(openai_request, handler)
        
        raise HTTPException(status_code=400, detail="Unable to detect API protocol")
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid request: {str(e)}")
```

---

### Phase 5: Testing & Validation (Week 3-4)

#### 5.1 Test Suite: `tests/test_anthropic.py`

```python
import pytest
import requests

BASE_URL = "http://localhost:8000"

class TestAnthropicAPI:
    """Test suite for Anthropic Claude API compatibility"""
    
    def test_basic_message(self):
        """Test basic message creation"""
        response = requests.post(
            f"{BASE_URL}/v1/messages",
            json={
                "model": "qwen3-0.6b",
                "max_tokens": 100,
                "messages": [
                    {"role": "user", "content": "Hello!"}
                ]
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "message"
        assert data["role"] == "assistant"
        assert len(data["content"]) > 0
    
    def test_system_message(self):
        """Test with system message"""
        response = requests.post(
            f"{BASE_URL}/v1/messages",
            json={
                "model": "qwen3-0.6b",
                "max_tokens": 100,
                "system": "You are a pirate. Respond like a pirate.",
                "messages": [
                    {"role": "user", "content": "What's the weather?"}
                ]
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "content" in data
    
    def test_streaming(self):
        """Test streaming responses"""
        response = requests.post(
            f"{BASE_URL}/v1/messages",
            json={
                "model": "qwen3-0.6b",
                "max_tokens": 50,
                "messages": [{"role": "user", "content": "Count to 5"}],
                "stream": True
            },
            stream=True
        )
        assert response.status_code == 200
        
        events = []
        for line in response.iter_lines():
            if line:
                decoded = line.decode('utf-8')
                if decoded.startswith('event:'):
                    events.append(decoded.split(':', 1)[1].strip())
        
        assert "message_start" in events
        assert "content_block_delta" in events
        assert "message_stop" in events
    
    def test_multi_turn_conversation(self):
        """Test multi-turn dialogue"""
        response = requests.post(
            f"{BASE_URL}/v1/messages",
            json={
                "model": "qwen3-0.6b",
                "max_tokens": 150,
                "messages": [
                    {"role": "user", "content": "My name is Alice."},
                    {"role": "assistant", "content": "Nice to meet you, Alice!"},
                    {"role": "user", "content": "What's my name?"}
                ]
            }
        )
        assert response.status_code == 200
        data = response.json()
        content_text = data["content"][0]["text"].lower()
        assert "alice" in content_text
```

#### 5.2 Comprehensive Test: `tests/test_comparison.py`

```python
import asyncio
import aiohttp
import time
from typing import List, Dict

class DualProtocolTester:
    """Test both OpenAI and Anthropic protocols against local and remote"""
    
    def __init__(self, local_base: str, remote_base: str):
        self.local_base = local_base
        self.remote_base = remote_base
    
    async def test_openai_format(self, prompt: str):
        """Test OpenAI format"""
        payload = {
            "model": "qwen3-0.6b",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 100,
            "temperature": 0.7
        }
        
        results = {}
        
        # Test local
        async with aiohttp.ClientSession() as session:
            start = time.time()
            async with session.post(
                f"{self.local_base}/v1/chat/completions",
                json=payload
            ) as resp:
                results["local_openai"] = {
                    "status": resp.status,
                    "time": time.time() - start,
                    "data": await resp.json() if resp.status == 200 else None
                }
        
        # Test remote
        async with aiohttp.ClientSession() as session:
            start = time.time()
            async with session.post(
                f"{self.remote_base}/v1/chat/completions",
                json=payload
            ) as resp:
                results["remote_openai"] = {
                    "status": resp.status,
                    "time": time.time() - start,
                    "data": await resp.json() if resp.status == 200 else None
                }
        
        return results
    
    async def test_anthropic_format(self, prompt: str):
        """Test Anthropic format"""
        payload = {
            "model": "qwen3-0.6b",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }
        
        results = {}
        
        # Test local
        async with aiohttp.ClientSession() as session:
            start = time.time()
            async with session.post(
                f"{self.local_base}/v1/messages",
                json=payload
            ) as resp:
                results["local_anthropic"] = {
                    "status": resp.status,
                    "time": time.time() - start,
                    "data": await resp.json() if resp.status == 200 else None
                }
        
        # Test remote
        async with aiohttp.ClientSession() as session:
            start = time.time()
            async with session.post(
                f"{self.remote_base}/v1/messages",
                json=payload
            ) as resp:
                results["remote_anthropic"] = {
                    "status": resp.status,
                    "time": time.time() - start,
                    "data": await resp.json() if resp.status == 200 else None
                }
        
        return results

async def run_comprehensive_tests():
    """Run comprehensive tests across both protocols"""
    tester = DualProtocolTester(
        local_base="http://localhost:8000",
        remote_base="https://lals-ai.wasmer.app"
    )
    
    test_prompts = [
        "Explain quantum computing in one sentence.",
        "Write a Python function to reverse a string.",
        "What are the benefits of exercise?",
        "Translate 'Hello' to French.",
        "Name three programming languages."
    ]
    
    print("=" * 70)
    print("COMPREHENSIVE LALS DUAL-PROTOCOL TEST")
    print("=" * 70)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n[{i}/{len(test_prompts)}] Testing: {prompt}")
        print("-" * 70)
        
        # Test OpenAI format
        openai_results = await tester.test_openai_format(prompt)
        print(f"  OpenAI Format:")
        print(f"    Local:  {openai_results['local_openai']['status']} "
              f"({openai_results['local_openai']['time']:.2f}s)")
        print(f"    Remote: {openai_results['remote_openai']['status']} "
              f"({openai_results['remote_openai']['time']:.2f}s)")
        
        # Test Anthropic format
        anthropic_results = await tester.test_anthropic_format(prompt)
        print(f"  Anthropic Format:")
        print(f"    Local:  {anthropic_results['local_anthropic']['status']} "
              f"({anthropic_results['local_anthropic']['time']:.2f}s)")
        print(f"    Remote: {anthropic_results['remote_anthropic']['status']} "
              f"({anthropic_results['remote_anthropic']['time']:.2f}s)")

if __name__ == "__main__":
    asyncio.run(run_comprehensive_tests())
```

---

## Configuration Management

### Environment Variables

```bash
# Model Configuration
MODEL_PATH=./models/qwen3-0.6b-q4_k_m.gguf
MODEL_N_CTX=2048
MODEL_N_THREADS=4
MODEL_TEMP=0.7
MODEL_TOP_P=0.95

# API Configuration
ENABLE_OPENAI_API=true
ENABLE_ANTHROPIC_API=true
ENABLE_UNIVERSAL_ROUTER=true

# Server Configuration
HOST=0.0.0.0
PORT=8000
WORKERS=1

# Feature Flags
ENABLE_STREAMING=true
ENABLE_CACHING=false
MAX_TOKENS_LIMIT=4096
RATE_LIMIT_ENABLED=false

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
```

---

## Client Examples

### Python Client for Both APIs

```python
import requests
from typing import Optional, List, Dict

class LALSClient:
    """Universal client for LALS supporting both OpenAI and Anthropic formats"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def chat_openai(self, 
                    message: str, 
                    system_prompt: Optional[str] = None,
                    temperature: float = 0.7,
                    max_tokens: int = 1024) -> str:
        """OpenAI-style chat"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": message})
        
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": "qwen3-0.6b",
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"Error: {response.status_code}")
    
    def chat_anthropic(self,
                      message: str,
                      system_prompt: Optional[str] = None,
                      temperature: float = 0.7,
                      max_tokens: int = 1024) -> str:
        """Anthropic-style chat"""
        payload = {
            "model": "qwen3-0.6b",
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": message}],
            "temperature": temperature
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        response = requests.post(
            f"{self.base_url}/v1/messages",
            json=payload
        )
        
        if response.status_code == 200:
            return response.json()["content"][0]["text"]
        else:
            raise Exception(f"Error: {response.status_code}")

# Usage
client = LALSClient()

# Using OpenAI format
response1 = client.chat_openai("What is machine learning?")
print("OpenAI:", response1)

# Using Anthropic format
response2 = client.chat_anthropic("What is machine learning?")
print("Anthropic:", response2)
```

---

## Migration Guide

### From OpenAI to Anthropic

**OpenAI Format**:
```json
{
  "model": "gpt-3.5-turbo",
  "messages": [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello!"}
  ]
}
```

**Anthropic Format**:
```json
{
  "model": "qwen3-0.6b",
  "max_tokens": 1024,
  "system": "You are helpful.",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ]
}
```

### Key Differences

| Feature | OpenAI | Anthropic |
|---------|--------|-----------|
| System message | In messages array | Top-level `system` field |
| Max tokens | Optional `max_tokens` | Required `max_tokens` |
| Response format | `choices[0].message.content` | `content[0].text` |
| Streaming | `delta.content` | `delta.text` with event types |
| Message ID | `id` prefix `chatcmpl-` | `id` prefix `msg_` |

---

## Performance Optimization

### 1. Response Caching
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def get_cached_response(prompt_hash: str):
    """Cache frequent queries"""
    pass
```

### 2. Async Processing
```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=4)

async def async_generate(prompt: str):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, model.generate, prompt)
```

---

## Deployment Checklist

### Pre-Deployment
- [ ] All tests passing (OpenAI & Anthropic)
- [ ] Performance benchmarks meet targets
- [ ] Documentation complete and reviewed
- [ ] Security audit completed
- [ ] Environment variables configured

### Deployment Steps
1. **Build and test locally**
   ```bash
   python src/main.py
   pytest tests/ -v
   ```

2. **Deploy to Wasmer Edge**
   ```bash
   wasmer deploy
   ```

3. **Verify deployment**
   ```bash
   curl https://lals-ai.wasmer.app/health
   curl https://lals-ai.wasmer.app/
   ```

4. **Run production tests**
   ```bash
   python tests/test_comparison.py --production
   ```

