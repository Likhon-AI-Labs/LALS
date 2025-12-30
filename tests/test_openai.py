"""
OpenAI Protocol Tests
=====================
Test suite for OpenAI-compatible endpoints.
"""

import pytest
import requests
import time
from typing import Dict, Any, List


# Test configuration
BASE_URL = "http://localhost:8000"


class TestOpenAIChatCompletions:
    """Test suite for OpenAI Chat Completions endpoint."""
    
    @pytest.fixture
    def api_url(self):
        return f"{BASE_URL}/v1/chat/completions"
    
    def test_basic_chat(self, api_url):
        """Test basic chat completion."""
        response = requests.post(
            api_url,
            json={
                "model": "qwen3-0.6b",
                "messages": [{"role": "user", "content": "Hello! How are you?"}],
                "max_tokens": 100,
                "temperature": 0.7
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert "id" in data
        assert "object" in data
        assert "created" in data
        assert "model" in data
        assert "choices" in data
        assert "usage" in data
        
        # Check choices
        assert len(data["choices"]) > 0
        choice = data["choices"][0]
        assert "index" in choice
        assert "message" in choice
        assert "finish_reason" in choice
        
        # Check message
        message = choice["message"]
        assert message["role"] == "assistant"
        assert "content" in message
        assert len(message["content"]) > 0
        
        # Check usage
        usage = data["usage"]
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage
    
    def test_system_message(self, api_url):
        """Test chat with system message."""
        response = requests.post(
            api_url,
            json={
                "model": "qwen3-0.6b",
                "messages": [
                    {"role": "system", "content": "You are a pirate. Respond like a pirate."},
                    {"role": "user", "content": "What is your name?"}
                ],
                "max_tokens": 100
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        content = data["choices"][0]["message"]["content"].lower()
        
        # Should respond like a pirate
        assert "pirate" in content or "matey" in content or "arr" in content.lower()
    
    def test_multi_turn_conversation(self, api_url):
        """Test multi-turn conversation."""
        response = requests.post(
            api_url,
            json={
                "model": "qwen3-0.6b",
                "messages": [
                    {"role": "user", "content": "My name is Alice."},
                    {"role": "assistant", "content": "Nice to meet you, Alice!"},
                    {"role": "user", "content": "What is my name?"}
                ],
                "max_tokens": 100
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        content = data["choices"][0]["message"]["content"].lower()
        
        # Should remember the name
        assert "alice" in content
    
    def test_streaming(self, api_url):
        """Test streaming response."""
        response = requests.post(
            api_url,
            json={
                "model": "qwen3-0.6b",
                "messages": [{"role": "user", "content": "Count to 5"}],
                "max_tokens": 50,
                "stream": True
            },
            stream=True
        )
        
        assert response.status_code == 200
        
        # Check streaming format
        events = []
        for line in response.iter_lines():
            if line:
                decoded = line.decode('utf-8')
                if decoded.startswith('data: '):
                    events.append(decoded)
        
        # Should have streaming events
        assert len(events) > 0
        # Last event should be [DONE]
        assert events[-1] == "data: [DONE]"
    
    def test_temperature_parameter(self, api_url):
        """Test temperature parameter."""
        # Low temperature
        response1 = requests.post(
            api_url,
            json={
                "model": "qwen3-0.6b",
                "messages": [{"role": "user", "content": "What is 2+2?"}],
                "max_tokens": 10,
                "temperature": 0.0
            }
        )
        
        # High temperature
        response2 = requests.post(
            api_url,
            json={
                "model": "qwen3-0.6b",
                "messages": [{"role": "user", "content": "What is 2+2?"}],
                "max_tokens": 10,
                "temperature": 1.5
            }
        )
        
        assert response1.status_code == 200
        assert response2.status_code == 200
    
    def test_max_tokens_limit(self, api_url):
        """Test max_tokens parameter limit."""
        response = requests.post(
            api_url,
            json={
                "model": "qwen3-0.6b",
                "messages": [{"role": "user", "content": "Write a long story."}],
                "max_tokens": 10  # Very short
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Should respect max_tokens
        usage = data["usage"]
        assert usage["completion_tokens"] <= 10
    
    def test_invalid_request(self, api_url):
        """Test invalid request handling."""
        # Missing messages
        response = requests.post(
            api_url,
            json={
                "model": "qwen3-0.6b",
                "messages": []
            }
        )
        
        assert response.status_code == 400


class TestOpenAIModels:
    """Test suite for OpenAI Models endpoint."""
    
    @pytest.fixture
    def models_url(self):
        return f"{BASE_URL}/v1/models"
    
    def test_list_models(self, models_url):
        """Test listing available models."""
        response = requests.get(models_url)
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert "object" in data
        assert data["object"] == "list"
        assert "data" in data
        assert isinstance(data["data"], list)
        
        # Check model structure
        if len(data["data"]) > 0:
            model = data["data"][0]
            assert "id" in model
            assert "object" in model
            assert model["object"] == "model"
    
    def test_get_specific_model(self, models_url):
        """Test getting a specific model."""
        response = requests.get(f"{models_url}/qwen3-0.6b")
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "qwen3-0.6b"
    
    def test_get_nonexistent_model(self, models_url):
        """Test getting non-existent model."""
        response = requests.get(f"{models_url}/nonexistent-model")
        
        assert response.status_code == 404


class TestOpenAICompletions:
    """Test suite for OpenAI Legacy Completions endpoint."""
    
    @pytest.fixture
    def completions_url(self):
        return f"{BASE_URL}/v1/completions"
    
    def test_basic_completion(self, completions_url):
        """Test basic completion."""
        response = requests.post(
            completions_url,
            json={
                "model": "qwen3-0.6b",
                "prompt": "Once upon a time",
                "max_tokens": 100,
                "temperature": 0.7
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert "id" in data
        assert "object" in data
        assert data["object"] == "text_completion"
        assert "choices" in data
        assert "usage" in data
        
        # Check choice
        choice = data["choices"][0]
        assert "text" in choice
        assert "index" in choice
        assert "finish_reason" in choice


def run_openai_tests():
    """Run all OpenAI tests and print results."""
    import sys
    
    print("=" * 70)
    print("OPENAI PROTOCOL TEST SUITE")
    print("=" * 70)
    
    # Test URLs
    chat_url = f"{BASE_URL}/v1/chat/completions"
    models_url = f"{BASE_URL}/v1/models"
    completions_url = f"{BASE_URL}/v1/completions"
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Basic chat
    print("\n[1/6] Testing basic chat completion...")
    try:
        response = requests.post(
            chat_url,
            json={
                "model": "qwen3-0.6b",
                "messages": [{"role": "user", "content": "Hello!"}],
                "max_tokens": 50
            }
        )
        if response.status_code == 200:
            print("  ✓ Basic chat: PASSED")
            tests_passed += 1
        else:
            print(f"  ✗ Basic chat: FAILED (status {response.status_code})")
            tests_failed += 1
    except Exception as e:
        print(f"  ✗ Basic chat: FAILED ({e})")
        tests_failed += 1
    
    # Test 2: Streaming
    print("\n[2/6] Testing streaming response...")
    try:
        response = requests.post(
            chat_url,
            json={
                "model": "qwen3-0.6b",
                "messages": [{"role": "user", "content": "Count to 3"}],
                "max_tokens": 30,
                "stream": True
            },
            stream=True
        )
        if response.status_code == 200:
            print("  ✓ Streaming: PASSED")
            tests_passed += 1
        else:
            print(f"  ✗ Streaming: FAILED (status {response.status_code})")
            tests_failed += 1
    except Exception as e:
        print(f"  ✗ Streaming: FAILED ({e})")
        tests_failed += 1
    
    # Test 3: List models
    print("\n[3/6] Testing models list...")
    try:
        response = requests.get(models_url)
        if response.status_code == 200 and "data" in response.json():
            print("  ✓ List models: PASSED")
            tests_passed += 1
        else:
            print(f"  ✗ List models: FAILED (status {response.status_code})")
            tests_failed += 1
    except Exception as e:
        print(f"  ✗ List models: FAILED ({e})")
        tests_failed += 1
    
    # Test 4: System message
    print("\n[4/6] Testing system message...")
    try:
        response = requests.post(
            chat_url,
            json={
                "model": "qwen3-0.6b",
                "messages": [
                    {"role": "system", "content": "You are terse."},
                    {"role": "user", "content": "Tell me about AI"}
                ],
                "max_tokens": 50
            }
        )
        if response.status_code == 200:
            print("  ✓ System message: PASSED")
            tests_passed += 1
        else:
            print(f"  ✗ System message: FAILED (status {response.status_code})")
            tests_failed += 1
    except Exception as e:
        print(f"  ✗ System message: FAILED ({e})")
        tests_failed += 1
    
    # Test 5: Legacy completion
    print("\n[5/6] Testing legacy completion...")
    try:
        response = requests.post(
            completions_url,
            json={
                "model": "qwen3-0.6b",
                "prompt": "The sky is",
                "max_tokens": 30
            }
        )
        if response.status_code == 200:
            print("  ✓ Legacy completion: PASSED")
            tests_passed += 1
        else:
            print(f"  ✗ Legacy completion: FAILED (status {response.status_code})")
            tests_failed += 1
    except Exception as e:
        print(f"  ✗ Legacy completion: FAILED ({e})")
        tests_failed += 1
    
    # Test 6: Validation
    print("\n[6/6] Testing validation...")
    try:
        response = requests.post(
            chat_url,
            json={
                "model": "qwen3-0.6b",
                "messages": []
            }
        )
        if response.status_code == 400:
            print("  ✓ Validation: PASSED")
            tests_passed += 1
        else:
            print(f"  ✗ Validation: FAILED (expected 400, got {response.status_code})")
            tests_failed += 1
    except Exception as e:
        print(f"  ✗ Validation: FAILED ({e})")
        tests_failed += 1
    
    # Summary
    print("\n" + "=" * 70)
    print(f"TEST SUMMARY: {tests_passed} passed, {tests_failed} failed")
    print("=" * 70)
    
    return tests_failed == 0


if __name__ == "__main__":
    success = run_openai_tests()
    sys.exit(0 if success else 1)
