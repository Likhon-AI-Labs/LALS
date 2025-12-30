"""
Dual Protocol Comparison Tests
==============================
Test script comparing local LALS endpoint with lals-ai.wasmer.app
for both OpenAI and Anthropic protocol formats.
"""

import asyncio
import json
import time
import requests
from datetime import datetime
from typing import Dict, Any, List, Optional


# Configuration
LOCAL_BASE = "http://localhost:8000"
REMOTE_BASE = "https://lals-ai.wasmer.app"


class ProtocolTester:
    """Test both OpenAI and Anthropic protocols against local and remote endpoints."""
    
    def __init__(self, local_base: str, remote_base: str):
        self.local_base = local_base
        self.remote_base = remote_base
    
    def test_openai_chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 100,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """Test OpenAI chat format."""
        payload = {
            "model": "qwen3-0.6b",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }
        
        results = {}
        
        # Test local
        start = time.time()
        try:
            response = requests.post(
                f"{self.local_base}/v1/chat/completions",
                json=payload,
                timeout=60
            )
            local_time = time.time() - start
            
            if response.status_code == 200:
                results["local"] = {
                    "success": True,
                    "time": local_time,
                    "status": response.status_code,
                    "content": response.json()["choices"][0]["message"]["content"],
                    "tokens": response.json()["usage"]
                }
            else:
                results["local"] = {
                    "success": False,
                    "time": local_time,
                    "status": response.status_code,
                    "error": response.text[:200]
                }
        except Exception as e:
            results["local"] = {
                "success": False,
                "time": time.time() - start,
                "error": str(e)
            }
        
        # Test remote
        start = time.time()
        try:
            response = requests.post(
                f"{self.remote_base}/v1/chat/completions",
                json=payload,
                timeout=60
            )
            remote_time = time.time() - start
            
            if response.status_code == 200:
                results["remote"] = {
                    "success": True,
                    "time": remote_time,
                    "status": response.status_code,
                    "content": response.json()["choices"][0]["message"]["content"],
                    "tokens": response.json()["usage"]
                }
            else:
                results["remote"] = {
                    "success": False,
                    "time": remote_time,
                    "status": response.status_code,
                    "error": response.text[:200]
                }
        except Exception as e:
            results["remote"] = {
                "success": False,
                "time": time.time() - start,
                "error": str(e)
            }
        
        return results
    
    def test_anthropic_messages(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        max_tokens: int = 100,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """Test Anthropic Messages API format."""
        payload = {
            "model": "qwen3-0.6b",
            "max_tokens": max_tokens,
            "messages": messages,
            "temperature": temperature,
            "stream": False
        }
        
        if system:
            payload["system"] = system
        
        results = {}
        
        # Test local
        start = time.time()
        try:
            response = requests.post(
                f"{self.local_base}/v1/messages",
                json=payload,
                timeout=60
            )
            local_time = time.time() - start
            
            if response.status_code == 200:
                data = response.json()
                results["local"] = {
                    "success": True,
                    "time": local_time,
                    "status": response.status_code,
                    "content": data["content"][0]["text"],
                    "tokens": data["usage"],
                    "message_id": data["id"]
                }
            else:
                results["local"] = {
                    "success": False,
                    "time": local_time,
                    "status": response.status_code,
                    "error": response.text[:200]
                }
        except Exception as e:
            results["local"] = {
                "success": False,
                "time": time.time() - start,
                "error": str(e)
            }
        
        # Test remote
        start = time.time()
        try:
            response = requests.post(
                f"{self.remote_base}/v1/messages",
                json=payload,
                timeout=60
            )
            remote_time = time.time() - start
            
            if response.status_code == 200:
                data = response.json()
                results["remote"] = {
                    "success": True,
                    "time": remote_time,
                    "status": response.status_code,
                    "content": data["content"][0]["text"],
                    "tokens": data["usage"],
                    "message_id": data["id"]
                }
            else:
                results["remote"] = {
                    "success": False,
                    "time": remote_time,
                    "status": response.status_code,
                    "error": response.text[:200]
                }
        except Exception as e:
            results["remote"] = {
                "success": False,
                "time": time.time() - start,
                "error": str(e)
            }
        
        return results
    
    def test_universal_inference(
        self,
        payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Test universal inference endpoint with auto-detection."""
        results = {}
        
        # Test local
        start = time.time()
        try:
            response = requests.post(
                f"{self.local_base}/v1/inference",
                json=payload,
                timeout=60
            )
            local_time = time.time() - start
            
            if response.status_code == 200:
                results["local"] = {
                    "success": True,
                    "time": local_time,
                    "status": response.status_code,
                    "data": response.json()
                }
            else:
                results["local"] = {
                    "success": False,
                    "time": local_time,
                    "status": response.status_code,
                    "error": response.text[:200]
                }
        except Exception as e:
            results["local"] = {
                "success": False,
                "time": time.time() - start,
                "error": str(e)
            }
        
        return results


# Test prompts
OPENAI_TEST_CASES = [
    {
        "name": "Basic Greeting",
        "messages": [{"role": "user", "content": "Hello! How are you today?"}],
        "max_tokens": 80
    },
    {
        "name": "Knowledge Question",
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "max_tokens": 50
    },
    {
        "name": "Creative Writing",
        "messages": [{"role": "user", "content": "Write a short poem about technology."}],
        "max_tokens": 100
    },
    {
        "name": "Code Assistance",
        "messages": [{"role": "user", "content": "Write a Python function to calculate factorial."}],
        "max_tokens": 150
    },
    {
        "name": "Technical Explanation",
        "messages": [{"role": "user", "content": "Explain quantum physics in one sentence."}],
        "max_tokens": 50
    },
    {
        "name": "Multi-turn Conversation",
        "messages": [
            {"role": "user", "content": "My favorite color is blue."},
            {"role": "assistant", "content": "That's a nice color!"},
            {"role": "user", "content": "What is my favorite color?"}
        ],
        "max_tokens": 50
    }
]

ANTHROPIC_TEST_CASES = [
    {
        "name": "Basic Message",
        "messages": [{"role": "user", "content": "Hello! How are you?"}],
        "system": None,
        "max_tokens": 80
    },
    {
        "name": "With System Prompt",
        "messages": [{"role": "user", "content": "Solve this puzzle: 2+2=?"}],
        "system": "You are a math tutor who explains solutions clearly.",
        "max_tokens": 80
    },
    {
        "name": "Creative Task",
        "messages": [{"role": "user", "content": "Describe a sunset in vivid detail."}],
        "system": None,
        "max_tokens": 100
    },
    {
        "name": "Technical Question",
        "messages": [{"role": "user", "content": "What is an API?"}],
        "system": "You are a software engineer.",
        "max_tokens": 100
    }
]


def print_result(name: str, result: Dict[str, Any], protocol: str):
    """Print test result in a formatted way."""
    print(f"\n  [{protocol}] {name}")
    print("  " + "-" * 50)
    
    if result.get("local", {}).get("success"):
        local = result["local"]
        print(f"    Local:  ✓ {local['time']:.2f}s | "
              f"Tokens: {local.get('tokens', {}).get('total_tokens', 'N/A')}")
        content = local.get("content", "")[:60]
        print(f"           Response: \"{content}...\"")
    else:
        error = result.get("local", {}).get("error", "Unknown error")
        print(f"    Local:  ✗ Failed - {error}")
    
    if result.get("remote", {}).get("success"):
        remote = result["remote"]
        print(f"    Remote: ✓ {remote['time']:.2f}s | "
              f"Tokens: {remote.get('tokens', {}).get('total_tokens', 'N/A')}")
        content = remote.get("content", "")[:60]
        print(f"           Response: \"{content}...\"")
    else:
        error = result.get("remote", {}).get("error", "Unknown error")
        print(f"    Remote: ✗ Failed - {error}")


def run_comparison_tests():
    """Run comprehensive comparison tests."""
    import sys
    
    print("=" * 80)
    print("LALS DUAL-PROTOCOL COMPARISON TEST")
    print("=" * 80)
    print(f"\nTest started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Local endpoint:  {LOCAL_BASE}")
    print(f"Remote endpoint: {REMOTE_BASE}")
    print("=" * 80)
    
    tester = ProtocolTester(LOCAL_BASE, REMOTE_BASE)
    
    local_success = 0
    local_total = 0
    remote_success = 0
    remote_total = 0
    
    # OpenAI Tests
    print("\n" + "=" * 80)
    print("OPENAI PROTOCOL TESTS")
    print("=" * 80)
    
    for i, test_case in enumerate(OPENAI_TEST_CASES, 1):
        name = test_case["name"]
        messages = test_case["messages"]
        max_tokens = test_case["max_tokens"]
        
        print(f"\n[{i}/{len(OPENAI_TEST_CASES)}] {name}")
        
        result = tester.test_openai_chat(messages, max_tokens)
        
        if result.get("local", {}).get("success"):
            local_success += 1
        local_total += 1
        
        if result.get("remote", {}).get("success"):
            remote_success += 1
        remote_total += 1
        
        print_result(name, result, "OpenAI")
        
        # Brief pause between tests
        time.sleep(0.5)
    
    # Anthropic Tests
    print("\n" + "=" * 80)
    print("ANTHROPIC PROTOCOL TESTS")
    print("=" * 80)
    
    for i, test_case in enumerate(ANTHROPIC_TEST_CASES, 1):
        name = test_case["name"]
        messages = test_case["messages"]
        system = test_case["system"]
        max_tokens = test_case["max_tokens"]
        
        print(f"\n[{i}/{len(ANTHROPIC_TEST_CASES)}] {name}")
        
        result = tester.test_anthropic_messages(messages, system, max_tokens)
        
        if result.get("local", {}).get("success"):
            local_success += 1
        local_total += 1
        
        if result.get("remote", {}).get("success"):
            remote_success += 1
        remote_total += 1
        
        print_result(name, result, "Anthropic")
        
        time.sleep(0.5)
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    local_rate = (local_success / local_total * 100) if local_total > 0 else 0
    remote_rate = (remote_success / remote_total * 100) if remote_total > 0 else 0
    
    print(f"\nLocal Endpoint ({LOCAL_BASE}):")
    print(f"  - Success rate: {local_success}/{local_total} ({local_rate:.0f}%)")
    
    print(f"\nRemote Endpoint ({REMOTE_BASE}):")
    print(f"  - Success rate: {remote_success}/{remote_total} ({remote_rate:.0f}%)")
    
    print("\n" + "=" * 80)
    
    # Save results to file
    results = {
        "timestamp": datetime.now().isoformat(),
        "local_endpoint": LOCAL_BASE,
        "remote_endpoint": REMOTE_BASE,
        "summary": {
            "local_success": local_success,
            "local_total": local_total,
            "remote_success": remote_success,
            "remote_total": remote_total
        }
    }
    
    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to: test_results.json")
    
    return local_success == local_total


async def run_async_comparison():
    """Run async comparison tests."""
    # This can be extended for concurrent testing
    pass


def main():
    """Main entry point."""
    success = run_comparison_tests()
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
