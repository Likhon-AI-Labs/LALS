# LALS Multi-API Gateway

**Version**: 2.0.0  
**Status**: In Development  
**Last Updated**: 2024

## Overview

LALS Multi-API Gateway is an enhanced version of the Likhon Advanced Language System that supports both OpenAI and Anthropic Claude API protocols through a unified gateway architecture. This enables developers to use a single endpoint with automatic protocol detection, or choose specific API formats for their applications.

## Architecture

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

## Features

### Core Features
- **Dual Protocol Support**: OpenAI and Anthropic API compatibility
- **Auto-Detection**: Intelligent protocol detection based on request structure
- **Unified Response Format**: Consistent response handling across protocols
- **Streaming Support**: Full streaming capability for both protocols

### Endpoints

#### OpenAI Compatible
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completions |
| `/v1/models` | GET | List available models |
| `/v1/completions` | POST | Legacy completions |

#### Anthropic Compatible
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/messages` | POST | Claude Messages API |
| `/v1/complete` | POST | Legacy completions |

#### Universal
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/inference` | POST | Auto-detected routing |

## Quick Start

### Installation

```bash
git clone https://github.com/likhon-ai-labs/lals.git
cd lals
pip install -r requirements.txt
```

### Configuration

```bash
# Model Configuration
export MODEL_PATH="./models/qwen3-0.6b-q4_k_m.gguf"
export MODEL_N_CTX=2048
export MODEL_N_THREADS=4

# API Configuration
export ENABLE_OPENAI_API=true
export ENABLE_ANTHROPIC_API=true
export ENABLE_UNIVERSAL_ROUTER=true

# Server Configuration
export HOST="0.0.0.0"
export PORT=8000
```

### Running the Server

```bash
python -m src.main
```

## API Usage

### OpenAI Format

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "qwen3-0.6b",
        "messages": [
            {"role": "user", "content": "Hello!"}
        ],
        "temperature": 0.7,
        "max_tokens": 100
    }
)

print(response.json())
```

### Anthropic Format

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/messages",
    json={
        "model": "qwen3-0.6b",
        "max_tokens": 100,
        "messages": [
            {"role": "user", "content": "Hello!"}
        ],
        "temperature": 0.7
    }
)

print(response.json())
```

### Universal Endpoint

```python
import requests

# Auto-detects protocol based on request structure
response = requests.post(
    "http://localhost:8000/v1/inference",
    json={
        "model": "qwen3-0.6b",
        "messages": [{"role": "user", "content": "Hello!"}]
    }
)
```

## Response Formats

### OpenAI Response

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1677858242,
  "model": "qwen3-0.6b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help you?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 15,
    "total_tokens": 25
  }
}
```

### Anthropic Response

```json
{
  "id": "msg_01XYZ123",
  "type": "message",
  "role": "assistant",
  "content": [
    {
      "type": "text",
      "text": "Hello! How can I help you?"
    }
  ],
  "model": "qwen3-0.6b",
  "stop_reason": "end_turn",
  "usage": {
    "input_tokens": 10,
    "output_tokens": 15
  }
}
```

## Streaming

### OpenAI Streaming

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "qwen3-0.6b",
        "messages": [{"role": "user", "content": "Count to 5"}],
        "stream": True
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        print(line)
```

### Anthropic Streaming

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/messages",
    json={
        "model": "qwen3-0.6b",
        "max_tokens": 50,
        "messages": [{"role": "user", "content": "Count to 5"}],
        "stream": True
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        print(line)
        # Events: message_start, content_block_delta, message_stop
```

## Protocol Detection

The universal endpoint automatically detects the protocol based on:

1. **Request Structure**: 
   - OpenAI: `messages` array with optional `max_tokens`
   - Anthropic: `max_tokens` required, optional `system` field

2. **Header Override**:
   - Set `x-lals-provider` header to `openai` or `anthropic`

3. **Endpoint Selection**:
   - `/v1/chat/completions` → OpenAI
   - `/v1/messages` → Anthropic
   - `/v1/inference` → Auto-detect

## Python Client

```python
class LALSClient:
    """Universal client supporting both OpenAI and Anthropic formats"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def chat_openai(self, message: str, **kwargs) -> str:
        """OpenAI-style chat"""
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": "qwen3-0.6b",
                "messages": [{"role": "user", "content": message}],
                **kwargs
            }
        )
        return response.json()["choices"][0]["message"]["content"]
    
    def chat_anthropic(self, message: str, **kwargs) -> str:
        """Anthropic-style chat"""
        response = requests.post(
            f"{self.base_url}/v1/messages",
            json={
                "model": "qwen3-0.6b",
                "max_tokens": kwargs.pop("max_tokens", 1024),
                "messages": [{"role": "user", "content": message}],
                **kwargs
            }
        )
        return response.json()["content"][0]["text"]
```

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `./models/qwen3-0.6b-q4_k_m.gguf` | Path to model file |
| `MODEL_N_CTX` | `2048` | Context window size |
| `MODEL_N_THREADS` | `4` | CPU threads for inference |
| `ENABLE_OPENAI_API` | `true` | Enable OpenAI endpoints |
| `ENABLE_ANTHROPIC_API` | `true` | Enable Anthropic endpoints |
| `ENABLE_UNIVERSAL_ROUTER` | `true` | Enable auto-detection |
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8000` | Server port |
| `LOG_LEVEL` | `INFO` | Logging level |

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
| System message | In `messages` array | Top-level `system` field |
| Max tokens | Optional | Required |
| Response | `choices[0].message.content` | `content[0].text` |
| Streaming | `delta.content` | Event-based SSE |
| Message ID | `chatcmpl-` prefix | `msg_` prefix |

## Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Protocol-Specific Tests

```bash
# OpenAI tests
pytest tests/test_openai.py -v

# Anthropic tests
pytest tests/test_anthropic.py -v

# Comparison tests
pytest tests/test_comparison.py -v
```

### Performance Benchmarks

```bash
python benchmarks/run_benchmarks.py
```

## Deployment

### Local Development

```bash
python -m src.main
```

### Wasmer Edge

```bash
wasmer deploy
```

### Docker

```bash
docker build -t lals-gateway .
docker run -p 8000:8000 lals-gateway
```

## Project Structure

```
lals/
├── src/
│   ├── main.py                      # Entry point
│   ├── core/
│   │   ├── __init__.py
│   │   ├── model.py                 # Model loading & inference
│   │   └── config.py                # Configuration
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
│   └── qwen3-0.6b-q4_k_m.gguf       # Model file (not in git)
├── tests/
│   ├── test_openai.py
│   ├── test_anthropic.py
│   └── test_comparison.py
├── benchmarks/
│   └── run_benchmarks.py
├── requirements.txt
├── wasmer.toml
└── README.md
```

## License

This project is provided by Likhon AI Labs for research and development purposes.

## References

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Anthropic Claude API](https://docs.anthropic.com/)
- [Qwen3 Model](https://huggingface.co/Qwen/Qwen3-0.6B)
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- [FastAPI](https://fastapi.tiangolo.com/)
