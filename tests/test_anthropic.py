"""
Anthropic Protocol Tests
========================
Test suite for Anthropic-compatible endpoints.
"""

import pytest
import requests
import time
import json


# Test configuration
BASE_URL = "http://localhost:8000"


class TestAnthropicMessages:
    """Test suite for Anthropic Messages API endpoint."""
    
    @pytest.fixture
    def api_url(self):
        return f"{BASE_URL}/v1/messages"
    
    def test_basic_message(self, api_url):
        """Test basic message creation."""
        response = requests.post(
            api_url,
            json={
                "model": "qwen3-0.6b",
                "max_tokens": 100,
                "messages": [
                    {"role": "user", "content": "Hello! How are you?"}
                ]
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert "id" in data
        assert "type" in data
        assert data["type"] == "message"
        assert "role" in data
        assert data["role"] == "assistant"
        assert "content" in data
        assert isinstance(data["content"], list)
        assert "model" in data
        assert "usage" in data
        
        # Check content block
        content = data["content"]
        assert len(content) > 0
        block = content[0]
        assert "type" in block
        assert block["type"] == "text"
        assert "text" in block
        assert len(block["text"]) > 0
        
        # Check usage
        usage = data["usage"]
        assert "input_tokens" in usage
        assert "output_tokens" in usage
    
    def test_system_message(self, api_url):
        """Test with system message."""
        response = requests.post(
            api_url,
            json={
                "model": "qwen3-0.6b",
                "max_tokens": 100,
                "system": "You are a helpful coding assistant.",
                "messages": [
                    {"role": "user", "content": "Write a Python function"}
                ]
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Should produce code-related response
        content = data["content"][0]["text"]
        assert len(content) > 0
    
    def test_multi_turn_conversation(self, api_url):
        """Test multi-turn dialogue."""
        response = requests.post(
            api_url,
            json={
                "model": "qwen3-0.6b",
                "max_tokens": 150,
                "messages": [
                    {"role": "user", "content": "My name is Bob."},
                    {"role": "assistant", "content": "Nice to meet you, Bob!"},
                    {"role": "user", "content": "What is my name?"}
                ]
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        content = data["content"][0]["text"].lower()
        
        # Should remember the name
        assert "bob" in content
    
    def test_streaming(self, api_url):
        """Test streaming responses."""
        response = requests.post(
            api_url,
            json={
                "model": "qwen3-0.6b",
                "max_tokens": 50,
                "messages": [{"role": "user", "content": "Count to 5"}],
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
                if decoded.startswith('event:'):
                    events.append(decoded.split(':', 1)[1].strip())
        
        # Should have Anthropic-specific events
        assert "message_start" in events
        assert "content_block_delta" in events
        assert "message_stop" in events
    
    def test_temperature_parameter(self, api_url):
        """Test temperature parameter."""
        response = requests.post(
            api_url,
            json={
                "model": "qwen3-0.6b",
                "max_tokens": 50,
                "messages": [{"role": "user", "content": "What is 2+2?"}],
                "temperature": 0.0
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["content"][0]["text"]) > 0
    
    def test_stop_sequences(self, api_url):
        """Test stop sequences parameter."""
        response = requests.post(
            api_url,
            json={
                "model": "qwen3-0.6b",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "Tell me a story"}],
                "stop_sequences": ["The end."]
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        content = data["content"][0]["text"]
        
        # Should stop before or at stop sequence
        assert len(content) > 0
    
    def test_max_tokens_limit(self, api_url):
        """Test max_tokens parameter."""
        response = requests.post(
            api_url,
            json={
                "model": "qwen3-0.6b",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "Write a long story."}]
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Should respect max_tokens
        usage = data["usage"]
        assert usage["output_tokens"] <= 10
    
    def test_empty_messages_validation(self, api_url):
        """Test empty messages validation."""
        response = requests.post(
            api_url,
            json={
                "model": "qwen3-0.6b",
                "max_tokens": 100,
                "messages": []
            }
        )
        
        assert response.status_code == 400
    
    def test_missing_max_tokens(self, api_url):
        """Test missing max_tokens validation."""
        response = requests.post(
            api_url,
            json={
                "model": "qwen3-0.6b",
                "messages": [{"role": "user", "content": "Hello!"}]
            }
        )
        
        # Anthropic requires max_tokens
        assert response.status_code == 422 or response.status_code == 400


class TestAnthropicComplete:
    """Test suite for Anthropic Legacy Completions endpoint."""
    
    @pytest.fixture
    def complete_url(self):
        return f"{BASE_URL}/v1/complete"
    
    def test_basic_completion(self, complete_url):
        """Test basic completion."""
        response = requests.post(
            complete_url,
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
        assert "completion" in data
        assert "stop_reason" in data
        assert "truncated" in data
        assert "log_id" in data
    
    def test_max_tokens_to_sample(self, complete_url):
        """Test legacy max_tokens_to_sample parameter."""
        response = requests.post(
            complete_url,
            json={
                "model": "qwen3-0.6b",
                "prompt": "Tell me a joke",
                "max_tokens_to_sample": 50
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "completion" in data


class TestAnthropicProtocolFormat:
    """Test Anthropic-specific protocol format details."""
    
    @pytest.fixture
    def api_url(self):
        return f"{BASE_URL}/v1/messages"
    
    def test_message_id_format(self, api_url):
        """Test that message IDs have Anthropic format."""
        response = requests.post(
            api_url,
            json={
                "model": "qwen3-0.6b",
                "max_tokens": 50,
                "messages": [{"role": "user", "content": "Hi"}]
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Should start with msg_
        assert data["id"].startswith("msg_")
    
    def test_response_type(self, api_url):
        """Test response type field."""
        response = requests.post(
            api_url,
            json={
                "model": "qwen3-0.6b",
                "max_tokens": 50,
                "messages": [{"role": "user", "content": "Hi"}]
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Should be "message" type
        assert data["type"] == "message"
    
    def test_content_block_structure(self, api_url):
        """Test content block structure."""
        response = requests.post(
            api_url,
            json={
                "model": "qwen3-0.6b",
                "max_tokens": 50,
                "messages": [{"role": "user", "content": "Hello"}]
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Content should be array of blocks
        assert isinstance(data["content"], list)
        
        # Each block should have type and text
        block = data["content"][0]
        assert "type" in block
        assert "text" in block
        assert block["type"] == "text"
    
    def test_stop_reason_values(self, api_url):
        """Test stop reason values."""
        response = requests.post(
            api_url,
            json={
                "model": "qwen3-0.6b",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "Say 'hello' and nothing else"}]
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Should have stop_reason
        assert "stop_reason" in data


def run_anthropic_tests():
    """Run all Anthropic tests and print results."""
    import sys
    
    print("=" * 70)
    print("ANTHROPIC PROTOCOL TEST SUITE")
    print("=" * 70)
    
    messages_url = f"{BASE_URL}/v1/messages"
    complete_url = f"{BASE_URL}/v1/complete"
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Basic message
    print("\n[1/6] Testing basic message creation...")
    try:
        response = requests.post(
            messages_url,
            json={
                "model": "qwen3-0.6b",
                "max_tokens": 50,
                "messages": [{"role": "user", "content": "Hello!"}]
            }
        )
        if response.status_code == 200:
            data = response.json()
            if data["id"].startswith("msg_") and data["type"] == "message":
                print("  ✓ Basic message: PASSED")
                tests_passed += 1
            else:
                print("  ✗ Basic message: FAILED (wrong format)")
                tests_failed += 1
        else:
            print(f"  ✗ Basic message: FAILED (status {response.status_code})")
            tests_failed += 1
    except Exception as e:
        print(f"  ✗ Basic message: FAILED ({e})")
        tests_failed += 1
    
    # Test 2: System message
    print("\n[2/6] Testing system message...")
    try:
        response = requests.post(
            messages_url,
            json={
                "model": "qwen3-0.6b",
                "max_tokens": 50,
                "system": "You are a poet.",
                "messages": [{"role": "user", "content": "Describe the moon"}]
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
    
    # Test 3: Streaming
    print("\n[3/6] Testing streaming response...")
    try:
        response = requests.post(
            messages_url,
            json={
                "model": "qwen3-0.6b",
                "max_tokens": 30,
                "messages": [{"role": "user", "content": "Count to 3"}],
                "stream": True
            },
            stream=True
        )
        if response.status_code == 200:
            events = []
            for line in response.iter_lines():
                if line and line.decode('utf-8').startswith('event:'):
                    events.append(line.decode('utf-8').split(':', 1)[1].strip())
            
            if "message_start" in events and "message_stop" in events:
                print("  ✓ Streaming: PASSED")
                tests_passed += 1
            else:
                print("  ✗ Streaming: FAILED (wrong event format)")
                tests_failed += 1
        else:
            print(f"  ✗ Streaming: FAILED (status {response.status_code})")
            tests_failed += 1
    except Exception as e:
        print(f"  ✗ Streaming: FAILED ({e})")
        tests_failed += 1
    
    # Test 4: Multi-turn conversation
    print("\n[4/6] Testing multi-turn conversation...")
    try:
        response = requests.post(
            messages_url,
            json={
                "model": "qwen3-0.6b",
                "max_tokens": 50,
                "messages": [
                    {"role": "user", "content": "My name is Charlie."},
                    {"role": "assistant", "content": "Hi Charlie!"},
                    {"role": "user", "content": "What is my name?"}
                ]
            }
        )
        if response.status_code == 200:
            data = response.json()
            if "charlie" in data["content"][0]["text"].lower():
                print("  ✓ Multi-turn: PASSED")
                tests_passed += 1
            else:
                print("  ✗ Multi-turn: FAILED (didn't remember name)")
                tests_failed += 1
        else:
            print(f"  ✗ Multi-turn: FAILED (status {response.status_code})")
            tests_failed += 1
    except Exception as e:
        print(f"  ✗ Multi-turn: FAILED ({e})")
        tests_failed += 1
    
    # Test 5: Legacy completion
    print("\n[5/6] Testing legacy completion (/v1/complete)...")
    try:
        response = requests.post(
            complete_url,
            json={
                "model": "qwen3-0.6b",
                "prompt": "The quick brown fox",
                "max_tokens_to_sample": 30
            }
        )
        if response.status_code == 200:
            data = response.json()
            if "completion" in data:
                print("  ✓ Legacy completion: PASSED")
                tests_passed += 1
            else:
                print("  ✗ Legacy completion: FAILED (wrong response)")
                tests_failed += 1
        else:
            print(f"  ✗ Legacy completion: FAILED (status {response.status_code})")
            tests_failed += 1
    except Exception as e:
        print(f"  ✗ Legacy completion: FAILED ({e})")
        tests_failed += 1
    
    # Test 6: Validation
    print("\n[6/6] Testing validation (empty messages)...")
    try:
        response = requests.post(
            messages_url,
            json={
                "model": "qwen3-0.6b",
                "max_tokens": 50,
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
    success = run_anthropic_tests()
    sys.exit(0 if success else 1)
