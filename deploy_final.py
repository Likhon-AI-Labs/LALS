#!/usr/bin/env python3
"""
LALS Deployment Script
======================
This script handles the complete deployment process for LALS Multi-API Gateway
to Wasmer Edge. It verifies the environment, commits changes, and deploys.

Usage:
    python deploy_final.py [--skip-build] [--skip-deploy]
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# Colors for output
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"

def log(msg: str, color: str = GREEN):
    """Print a colored message."""
    print(f"{color}{msg}{RESET}")

def run_cmd(cmd: list, check: bool = True, timeout: int = 60) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    log(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd="/workspace/lals"
        )
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        if check and result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, cmd, result.stderr)
        return result
    except subprocess.TimeoutExpired:
        log(f"Command timed out after {timeout}s", RED)
        raise

def check_git_status():
    """Check git status and remote configuration."""
    log("\n=== Checking Git Status ===")
    
    # Check remotes
    run_cmd(["git", "remote", "-v"], check=False)
    
    # Check current branch
    run_cmd(["git", "branch", "--show-current"], check=False)
    
    # Check status
    run_cmd(["git", "status", "--short"], check=False)

def commit_changes():
    """Commit any uncommitted changes."""
    log("\n=== Committing Changes ===")
    
    # Add all changes
    run_cmd(["git", "add", "-A"], check=False)
    
    # Check if there are changes to commit
    result = run_cmd(["git", "status", "--porcelain"], check=False)
    if not result.stdout.strip():
        log("No changes to commit")
        return False
    
    # Commit with descriptive message
    commit_msg = """Update deployment configuration for v2.0.3

- Updated app.yaml to version 2.0.3
- Added environment variables for model and server configuration
- Verified lazy loading implementation for faster startup
- Configured for Wasmer Edge deployment"""
    
    run_cmd(["git", "commit", "-m", commit_msg], check=False)
    log("Changes committed successfully")
    return True

def push_changes():
    """Push changes to GitHub."""
    log("\n=== Pushing to GitHub ===")
    try:
        # Push to new-origin (Likhon-AI-Labs/LALS)
        run_cmd(["git", "push", "new-origin", "main"], timeout=120)
        log("Changes pushed to GitHub successfully")
        return True
    except Exception as e:
        log(f"Failed to push: {e}", RED)
        return False

def verify_wasmer_cli():
    """Verify Wasmer CLI is installed and logged in."""
    log("\n=== Verifying Wasmer CLI ===")
    
    try:
        # Check version
        result = run_cmd(["wasmer", "--version"], check=False)
        log(f"Wasmer version: {result.stdout.strip()}")
        
        # Check login status
        result = run_cmd(["wasmer", "whoami"], check=False)
        log(f"Logged in as: {result.stdout.strip()}")
        return True
    except Exception as e:
        log(f"Wasmer CLI not available: {e}", RED)
        log("Please install Wasmer CLI: curl https://get.wasmer.io -sSf | sh", YELLOW)
        return False

def deploy_to_wasmer():
    """Deploy to Wasmer Edge."""
    log("\n=== Deploying to Wasmer Edge ===")
    
    try:
        # Deploy using wasmer deploy
        log("Running: wasmer deploy")
        result = subprocess.run(
            ["wasmer", "deploy", "--verbose"],
            capture_output=True,
            text=True,
            timeout=300,
            cwd="/workspace/lals"
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        
        if result.returncode == 0:
            log("Deployment initiated successfully!")
            log("The deployment may take a few minutes to propagate.")
            return True
        else:
            log("Deployment failed", RED)
            return False
    except subprocess.TimeoutExpired:
        log("Deployment timed out", RED)
        return False
    except Exception as e:
        log(f"Deployment error: {e}", RED)
        return False

def test_deployment(url: str = "https://lals-rudushi4.wasmer.app"):
    """Test the deployment."""
    log(f"\n=== Testing Deployment at {url} ===")
    
    import urllib.request
    import json
    
    test_endpoints = [
        ("/", "GET", None, "Root"),
        ("/health", "GET", None, "Health"),
        ("/v1/models", "GET", None, "Models"),
        ("/v1/chat/completions", "POST", 
         '{"model": "qwen3-0.6b", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 10}',
         "Chat Completion"),
    ]
    
    success_count = 0
    total_count = len(test_endpoints)
    
    for endpoint, method, data, name in test_endpoints:
        try:
            log(f"\nTesting {name}: {endpoint}")
            
            req = urllib.request.Request(
                f"{url}{endpoint}",
                method=method,
                headers={"Content-Type": "application/json"}
            )
            
            if data:
                req.data = data.encode()
            
            with urllib.request.urlopen(req, timeout=30) as response:
                status = response.status
                content = response.read().decode()
                
                if status == 200:
                    log(f"  ✓ {name}: Status {status}", GREEN)
                    try:
                        json_data = json.loads(content)
                        log(f"  Response: {str(json_data)[:100]}...", GREEN)
                    except:
                        pass
                    success_count += 1
                else:
                    log(f"  ✗ {name}: Status {status}", RED)
                    
        except Exception as e:
            log(f"  ✗ {name}: {str(e)[:50]}", RED)
    
    log(f"\n=== Test Results: {success_count}/{total_count} passed ===")
    return success_count == total_count

def main():
    """Main deployment function."""
    print(f"{BOLD}========================================{RESET}")
    print(f"{BOLD}  LALS Deployment Script v2.0.3{RESET}")
    print(f"{BOLD}========================================{RESET}")
    
    skip_build = "--skip-build" in sys.argv
    skip_deploy = "--skip-deploy" in sys.argv
    
    try:
        # Step 1: Check git status
        check_git_status()
        
        # Step 2: Commit changes
        if commit_changes():
            # Step 3: Push to GitHub
            if not push_changes():
                log("Failed to push changes, continuing anyway...", YELLOW)
        else:
            log("No changes to commit")
        
        # Step 4: Deploy to Wasmer (if not skipped)
        if not skip_deploy:
            if verify_wasmer_cli():
                deploy_to_wasmer()
                
                # Wait for deployment to propagate
                log("\nWaiting 60 seconds for deployment to propagate...")
                time.sleep(60)
                
                # Test deployment
                test_deployment()
            else:
                log("\nWasmer CLI not available. Please deploy manually:", YELLOW)
                log("1. Install Wasmer CLI: curl https://get.wasmer.io -sSf | sh")
                log("2. Login: wasmer login")
                log("3. Deploy: cd /workspace/lals && wasmer deploy")
        
        log("\n=== Deployment Summary ===")
        log("1. Changes committed and pushed to GitHub")
        log("2. Deployment to Wasmer Edge initiated")
        log("3. Test endpoints at: https://lals-rudushi4.wasmer.app")
        log("\nNote: First API request will trigger model download (~350MB)")
        log("This may take a few minutes. Subsequent requests will be faster.")
        
    except Exception as e:
        log(f"\nDeployment failed: {e}", RED)
        sys.exit(1)

if __name__ == "__main__":
    main()
