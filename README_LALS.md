# LALS - Likhon Advanced Language System

## Overview

LALS (Likhon Advanced Language System) is a flagship architecture developed by Likhon AI Labs. This implementation provides a local inference endpoint using the Qwen3-0.6B model in GGUF format, offering OpenAI-compatible chat completion APIs.

### Key Features

- **Transformer-based architecture** optimized for lower latency and efficient token processing
- **Versatile applications**: Advanced NLU, automated content generation, and code assistance
- **Industry specialization**: Fine-tuned for specific tasks, outperforming larger models in niche domains
- **OpenAI-compatible API**: Easy integration with existing applications and tools

## Requirements

- Python 3.10+
- 4GB+ RAM (model is ~462MB + runtime overhead)
- ~2GB disk space
- CPU (GPU not required for this model size)

## Installation

### 1. Clone and Setup

```bash
git clone https://github.com/likhon-ai-labs/lals.git
cd lals
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install --upgrade pip
```

### 3. Install Dependencies

```bash
pip install fastapi uvicorn llama-cpp-python pydantic requests
```

### 4. Download Model

The Qwen3-0.6B quantized model (Q4_K_M) has been downloaded to:
```
models/qwen3-0.6b-q4_k_m.gguf
```

If you need to download it manually:
```bash
mkdir -p models
wget -O models/qwen3-0.6b-q4_k_m.gguf \
  "https://huggingface.co/enacimie/Qwen3-0.6B-Q4_K_M-GGUF/resolve/main/qwen3-0.6b-q4_k_m.gguf"
```

## Running the Server

### Default Configuration (Port 8000)

```bash
source venv/bin/activate
python src/main.py
```

### Custom Configuration

```bash
# Custom port
MODEL_PATH=./models/qwen3-0.6b-q4_k_m.gguf \
MODEL_N_CTX=2048 \
MODEL_N_THREADS=4 \
python src/main.py
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `./models/qwen3-0.6b-q4_k_m.gguf` | Path to GGUF model file |
| `MODEL_N_CTX` | `2048` | Context window size |
| `MODEL_N_THREADS` | `4` | Number of CPU threads |
| `MODEL_TEMP` | `0.7` | Generation temperature |
| `MODEL_TOP_P` | `0.95` | Top-p sampling parameter |

## API Usage

### Base URL

```
http://localhost:8000
```

### Endpoints

#### GET / - Service Information

```bash
curl http://localhost:8000/
```

Response:
```json
{
  "service": "LALS - Likhon Advanced Language System",
  "version": "1.0.0",
  "model": "Qwen3-0.6B-Q4_K_M",
  "architecture": "Transformer-based, optimized for lower latency",
  "capabilities": ["NLU", "Content Generation", "Code Assistance"],
  "endpoints": {...}
}
```

#### GET /health - Health Check

```bash
curl http://localhost:8000/health
```

#### POST /v1/chat/completions - Chat Completion

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-0.6b",
    "messages": [
      {"role": "user", "content": "What is machine learning?"}
    ],
    "temperature": 0.7,
    "max_tokens": 200
  }'
```

Request Body:
```json
{
  "model": "string",
  "messages": [
    {"role": "user", "content": "string"},
    {"role": "assistant", "content": "string"},
    {"role": "system", "content": "string"}
  ],
  "temperature": 0.7,
  "max_tokens": 1024,
  "top_p": 0.95,
  "stream": false
}
```

Response:
```json
{
  "id": "cmpl-abc123",
  "object": "chat.completion",
  "created": 1677858242,
  "model": "qwen3-0.6b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Machine learning is a subset of artificial intelligence..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 45,
    "total_tokens": 60
  }
}
```

## Testing Against lals-ai.wasmer.app

Use the provided test script to compare your local endpoint with the remote service:

```bash
# First, start the local server in one terminal
python src/main.py

# Then, run the comparison test in another terminal
source venv/bin/activate
python test_comparison.py
```

### Example Test Results

```
==============================================================
LALS ENDPOINT COMPARISON TEST
==============================================================
Test started at: 2025-12-31 00:00:00
Local endpoint: http://localhost:8000/v1/chat/completions
Remote endpoint: https://lals-ai.wasmer.app/v1/chat/completions
Number of test prompts: 5
==============================================================

[1/5] Testing: Basic Greeting
------------------------------------------------------------
  Testing LOCAL endpoint...
    ✓ Success in 2.34s
    Response: Hello! I'm doing well, thank you for asking. How can I...
  Testing REMOTE endpoint (lals-ai.wasmer.app)...
    ✓ Success in 0.89s
    Response: Hello! I'm great, thanks for asking. How can I help you...

TEST SUMMARY
==============================================================
Local Endpoint (http://localhost:8000/v1/chat/completions):
  - Success rate: 5/5 (100%)
  - Average response time: 2.45s

Remote Endpoint (https://lals-ai.wasmer.app/v1/chat/completions):
  - Success rate: 5/5 (100%)
  - Average response time: 0.95s
```

## Python Client Example

```python
import requests

class LALSClient:
    """Client for LALS - Likhon Advanced Language System"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.chat_endpoint = f"{base_url}/v1/chat/completions"
    
    def chat(self, message: str, system_prompt: str = None, 
             temperature: float = 0.7, max_tokens: int = 1024) -> str:
        """
        Send a chat message and get a response.
        
        Args:
            message: User message
            system_prompt: Optional system prompt
            temperature: Response creativity (0.0-2.0)
            max_tokens: Maximum response length
        
        Returns:
            Assistant response text
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": message})
        
        payload = {
            "model": "qwen3-0.6b",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        response = requests.post(
            self.chat_endpoint,
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            return data["choices"][0]["message"]["content"]
        else:
            raise Exception(f"Error {response.status_code}: {response.text}")


# Usage example
client = LALSClient()

# Simple conversation
response = client.chat("Explain quantum computing in simple terms")
print(response)

# With system prompt
response = client.chat(
    "Write a haiku about the ocean",
    system_prompt="You are a poet who writes in the style of classic haiku"
)
print(response)
```

## Deployment to Wasmer Edge

This application is designed to be deployed on Wasmer Edge:

1. Ensure your project exposes `main:app` (module:variable)
2. Deploy using Wasmer CLI:
   ```bash
   wasmer deploy
   ```
3. Visit `https://<your-subdomain>.wasmer.app/` to test

## Architecture Details

### Model Specifications

- **Model**: Qwen3-0.6B
- **Quantization**: Q4_K_M (4-bit quantized)
- **Parameters**: 0.6 billion
- **Context Window**: 2048 tokens
- **Format**: GGUF (GPT-Generated Unified Format)

### Performance Characteristics

- **Memory Usage**: ~500MB (model) + ~1GB (runtime)
- **Inference Speed**: ~30 tokens/second on modern CPU
- **Latency**: ~2-3 seconds for typical queries

## Troubleshooting

### Model File Not Found

```bash
# Verify model exists
ls -lh models/qwen3-0.6b-q4_k_m.gguf

# If missing, download it
wget -O models/qwen3-0.6b-q4_k_m.gguf \
  "https://huggingface.co/enacimie/Qwen3-0.6B-Q4_K_M-GGUF/resolve/main/qwen3-0.6b-q4_k_m.gguf"
```

### Out of Memory

Reduce context size:
```bash
MODEL_N_CTX=1024 python src/main.py
```

### Slow Inference

Adjust thread count to match your CPU:
```bash
# For 8-core CPU
MODEL_N_THREADS=8 python src/main.py
```

## License

This project is provided by Likhon AI Labs for research and development purposes.

## References

- [Likhon AI Labs](https://github.com/likhon-ai-labs)
- [Qwen3 Model](https://huggingface.co/Qwen/Qwen3-0.6B)
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- [FastAPI](https://fastapi.tiangolo.com/)
