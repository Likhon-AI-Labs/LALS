#!/usr/bin/env python3
"""
LALS Endpoint Comparison Test Script
=====================================
This script tests the local LALS endpoint against lals-ai.wasmer.app
to compare performance and response quality.
"""

import asyncio
import json
import time
import requests
from datetime import datetime


# Configuration
LOCAL_URL = "http://localhost:8000/v1/chat/completions"
REMOTE_URL = "https://lals-ai.wasmer.app/v1/chat/completions"

# Test prompts
TEST_PROMPTS = [
    {
        "name": "Basic Greeting",
        "messages": [{"role": "user", "content": "Hello! How are you today?"}],
        "max_tokens": 100
    },
    {
        "name": "Knowledge Question",
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "max_tokens": 100
    },
    {
        "name": "Creative Writing",
        "messages": [{"role": "user", "content": "Write a short poem about technology."}],
        "max_tokens": 200
    },
    {
        "name": "Code Assistance",
        "messages": [{"role": "user", "content": "Write a Python function to calculate factorial."}],
        "max_tokens": 200
    },
    {
        "name": "Technical Explanation",
        "messages": [{"role": "user", "content": "Explain quantum physics in one sentence."}],
        "max_tokens": 100
    }
]


def test_endpoint(url: str, prompt_config: dict, timeout: int = 60) -> dict:
    """
    Test a single endpoint with the given prompt.
    
    Args:
        url: Endpoint URL
        prompt_config: Prompt configuration dictionary
        timeout: Request timeout in seconds
    
    Returns:
        Dictionary containing response data and timing information
    """
    payload = {
        "model": "qwen3-0.6b",
        "messages": prompt_config["messages"],
        "max_tokens": prompt_config["max_tokens"],
        "temperature": 0.7,
        "top_p": 0.95,
        "stream": False
    }
    
    start_time = time.time()
    
    try:
        response = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=timeout
        )
        
        elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            return {
                "success": True,
                "status_code": response.status_code,
                "elapsed_time": elapsed_time,
                "response": data,
                "content": data["choices"][0]["message"]["content"],
                "usage": data.get("usage", {}),
                "error": None
            }
        else:
            return {
                "success": False,
                "status_code": response.status_code,
                "elapsed_time": elapsed_time,
                "response": None,
                "content": None,
                "usage": None,
                "error": f"HTTP {response.status_code}: {response.text[:200]}"
            }
            
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "status_code": None,
            "elapsed_time": timeout,
            "response": None,
            "content": None,
            "usage": None,
            "error": f"Request timed out after {timeout}s"
        }
    except Exception as e:
        return {
            "success": False,
            "status_code": None,
            "elapsed_time": time.time() - start_time,
            "response": None,
            "content": None,
            "usage": None,
            "error": str(e)
        }


async def run_comparison_test():
    """
    Run comparison tests between local and remote endpoints.
    
    This function tests both endpoints with the same prompts and
    compares response quality and performance.
    """
    print("\n" + "="*70)
    print("LALS ENDPOINT COMPARISON TEST")
    print("="*70)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Local endpoint: {LOCAL_URL}")
    print(f"Remote endpoint: {REMOTE_URL}")
    print(f"Number of test prompts: {len(TEST_PROMPTS)}")
    print("="*70 + "\n")
    
    results = []
    
    for i, prompt_config in enumerate(TEST_PROMPTS, 1):
        print(f"\n[{i}/{len(TEST_PROMPTS)}] Testing: {prompt_config['name']}")
        print("-" * 60)
        
        # Test local endpoint
        print("  Testing LOCAL endpoint...")
        local_result = test_endpoint(LOCAL_URL, prompt_config)
        
        if local_result["success"]:
            print(f"    ✓ Success in {local_result['elapsed_time']:.2f}s")
            print(f"    Response: {local_result['content'][:100]}...")
        else:
            print(f"    ✗ Failed: {local_result['error']}")
        
        # Test remote endpoint
        print("  Testing REMOTE endpoint (lals-ai.wasmer.app)...")
        remote_result = test_endpoint(REMOTE_URL, prompt_config)
        
        if remote_result["success"]:
            print(f"    ✓ Success in {remote_result['elapsed_time']:.2f}s")
            print(f"    Response: {remote_result['content'][:100]}...")
        else:
            print(f"    ✗ Failed: {remote_result['error']}")
        
        # Store results
        results.append({
            "prompt_name": prompt_config["name"],
            "local": local_result,
            "remote": remote_result
        })
        
        # Brief pause between tests
        time.sleep(1)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    local_success = sum(1 for r in results if r["local"]["success"])
    remote_success = sum(1 for r in results if r["remote"]["success"])
    
    local_avg_time = sum(r["local"]["elapsed_time"] for r in results if r["local"]["success"]) / max(local_success, 1)
    remote_avg_time = sum(r["remote"]["elapsed_time"] for r in results if r["remote"]["success"]) / max(remote_success, 1)
    
    print(f"\nLocal Endpoint ({LOCAL_URL}):")
    print(f"  - Success rate: {local_success}/{len(TEST_PROMPTS)} ({100*local_success/len(TEST_PROMPTS):.0f}%)")
    print(f"  - Average response time: {local_avg_time:.2f}s")
    
    print(f"\nRemote Endpoint ({REMOTE_URL}):")
    print(f"  - Success rate: {remote_success}/{len(TEST_PROMPTS)} ({100*remote_success/len(TEST_PROMPTS):.0f}%)")
    print(f"  - Average response time: {remote_avg_time:.2f}s")
    
    print("\n" + "="*70)
    
    return results


async def main():
    """Main entry point for the test script."""
    try:
        results = await run_comparison_test()
        
        # Save results to JSON file
        output_file = "test_results.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nResults saved to: {output_file}")
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\nError during testing: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
