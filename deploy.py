#!/usr/bin/env python3
"""
LALS Multi-API Gateway - Deployment Script
===========================================
This script handles the complete deployment process for LALS to Wasmer Edge.

Usage:
    python deploy.py --action deploy      # Deploy to Wasmer Edge
    python deploy.py --action test-local  # Test locally
    python deploy.py --action verify      # Verify deployment
    python deploy.py --action all         # Run all steps
"""

import os
import sys
import json
import time
import subprocess
import requests
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class DeploymentConfig:
    """Configuration for deployment."""
    repo_url: str = "https://github.com/Likhon-AI-Labs/LALS"
    deploy_url: str = "https://lals-ai.wasmer.app"
    local_port: int = 8000
    model_url: str = (
        "https://huggingface.co/enacimie/Qwen3-0.6B-Q4_K_M-GGUF/"
        "resolve/main/qwen3-0.6b-q4_k_m.gguf"
    )
    model_path: str = "models/qwen3-0.6b-q4_k_m.gguf"


class Colors:
    """ANSI color codes for terminal output."""
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_step(message: str) -> None:
    """Print a step message."""
    print(f"{Colors.BLUE}==>{Colors.ENDC} {message}")


def print_success(message: str) -> None:
    """Print a success message."""
    print(f"{Colors.GREEN}✓{Colors.ENDC} {message}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    print(f"{Colors.YELLOW}⚠{Colors.ENDC} {message}")


def print_error(message: str) -> None:
    """Print an error message."""
    print(f"{Colors.RED}✗{Colors.ENDC} {message}")


class DeploymentManager:
    """Manages the deployment process for LALS."""
    
    def __init__(self, config: Optional[DeploymentConfig] = None):
        """Initialize the deployment manager."""
        self.config = config or DeploymentConfig()
        self.repo_path = Path(__file__).parent
    
    def run_command(
        self,
        command: str,
        cwd: Optional[Path] = None,
        capture: bool = True
    ) -> Tuple[int, str, str]:
        """Run a shell command."""
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=str(cwd or self.repo_path),
                capture_output=capture,
                text=True,
                timeout=300
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return 1, "", "Command timed out"
        except Exception as e:
            return 1, "", str(e)
    
    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are met."""
        print_step("Checking prerequisites...")
        
        checks = [
            ("Python 3", "python3 --version"),
            ("Git", "git --version"),
        ]
        
        all_passed = True
        for name, cmd in checks:
            code, stdout, _ = self.run_command(cmd)
            if code == 0:
                version = stdout.strip().split('\n')[0] if stdout else "Unknown"
                print_success(f"{name}: {version}")
            else:
                print_error(f"{name}: Not found")
                all_passed = False
        
        # Check Wasmer
        code, stdout, _ = self.run_command("wasmer --version")
        if code == 0:
            print_success(f"Wasmer: {stdout.strip()}")
        else:
            print_warning("Wasmer: Not found (will be installed)")
        
        return all_passed
    
    def setup_repository(self, update: bool = False) -> bool:
        """Clone or update the repository."""
        print_step("Setting up repository...")
        
        repo_dir = self.repo_path / "lals"
        
        if repo_dir.exists() and update:
            print_warning("Updating existing repository...")
            code, _, stderr = self.run_command("git pull origin main", cwd=repo_dir)
            if code == 0:
                print_success("Repository updated")
                return True
            else:
                print_error(f"Update failed: {stderr}")
                return False
        elif repo_dir.exists():
            print_success(f"Repository already exists: {repo_dir}")
            return True
        else:
            print_warning("Cloning repository...")
            code, _, stderr = self.run_command(
                f"git clone {self.config.repo_url}"
            )
            if code == 0:
                print_success("Repository cloned")
                return True
            else:
                print_error(f"Clone failed: {stderr}")
                return False
    
    def install_dependencies(self) -> bool:
        """Install Python dependencies."""
        print_step("Installing dependencies...")
        
        venv_path = self.repo_path / "venv"
        
        # Create virtual environment
        if not venv_path.exists():
            code, _, stderr = self.run_command("python3 -m venv venv")
            if code != 0:
                print_error(f"Failed to create venv: {stderr}")
                return False
        
        # Activate and install
        pip_cmd = str(venv_path / "bin" / "pip")
        code, _, stderr = self.run_command(f"{pip_cmd} install --upgrade pip")
        if code != 0:
            print_warning(f"pip upgrade: {stderr}")
        
        code, _, stderr = self.run_command(
            f"{pip_cmd} install -r requirements.txt",
            cwd=self.repo_path
        )
        if code == 0:
            print_success("Dependencies installed")
            return True
        else:
            print_error(f"Failed to install: {stderr}")
            return False
    
    def download_model(self) -> bool:
        """Download the model file if needed."""
        print_step("Checking model file...")
        
        model_path = self.repo_path / self.config.model_path
        
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            print_success(f"Model found: {model_path.name} ({size_mb:.1f} MB)")
            return True
        
        print_warning("Model not found, downloading...")
        
        # Create models directory
        models_dir = self.repo_path / "models"
        models_dir.mkdir(exist_ok=True)
        
        # Download model
        print(f"Downloading from: {self.config.model_url}")
        code, stdout, stderr = self.run_command(
            f"wget -O {model_path} {self.config.model_url}"
        )
        
        if code == 0:
            size_mb = model_path.stat().st_size / (1024 * 1024)
            print_success(f"Model downloaded ({size_mb:.1f} MB)")
            return True
        else:
            print_error(f"Download failed: {stderr}")
            return False
    
    def test_locally(self) -> bool:
        """Test the application locally."""
        print_step("Testing locally...")
        
        import threading
        import uvicorn
        
        # Start server in background thread
        server_ready = threading.Event()
        
        def run_server():
            os.chdir(self.repo_path)
            sys.path.insert(0, str(self.repo_path / "src"))
            
            # Import and start app
            from main import app
            uvicorn.run(app, host="127.0.0.1", port=self.config.local_port, log_level="error")
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        # Wait for server to start
        print("Starting server...")
        time.sleep(8)
        
        # Test endpoints
        base_url = f"http://localhost:{self.config.local_port}"
        all_passed = True
        
        tests = [
            ("Root", f"{base_url}/", "LALS"),
            ("OpenAI", f"{base_url}/v1/chat/completions", "choices"),
            ("Anthropic", f"{base_url}/v1/messages", "content"),
            ("Models", f"{base_url}/v1/models", "data"),
        ]
        
        for name, url, keyword in tests:
            try:
                response = requests.post(
                    url,
                    json={
                        "model": "qwen3-0.6b",
                        "messages": [{"role": "user", "content": "Hi"}],
                        "max_tokens": 10
                    } if "completions" in url or "messages" in url else {},
                    timeout=10
                )
                
                if response.status_code == 200 and keyword in response.text:
                    print_success(f"{name}: OK")
                else:
                    print_warning(f"{name}: {response.status_code}")
                    all_passed = False
            except Exception as e:
                print_warning(f"{name}: {str(e)[:50]}")
                all_passed = False
        
        return all_passed
    
    def deploy_wasmer(self) -> bool:
        """Deploy to Wasmer Edge."""
        print_step("Deploying to Wasmer Edge...")
        
        # Check if logged in
        code, stdout, _ = self.run_command("wasmer whoami")
        if code != 0:
            print_warning("Not logged in to Wasmer")
            print("Please run: wasmer login")
            return False
        
        print_success(f"Logged in as: {stdout.strip()}")
        
        # Deploy
        code, stdout, stderr = self.run_command(
            "wasmer deploy",
            cwd=self.repo_path
        )
        
        if code == 0:
            print_success("Deployment initiated")
            print(f"Deploy URL: {self.config.deploy_url}")
            return True
        else:
            print_error(f"Deployment failed: {stderr}")
            return False
    
    def verify_deployment(self) -> bool:
        """Verify the deployment is working."""
        print_step("Verifying deployment...")
        
        base_url = self.config.deploy_url
        all_passed = True
        
        print(f"\nTesting: {base_url}\n")
        
        # Test 1: Root endpoint
        print("Testing root endpoint...")
        try:
            response = requests.get(base_url, timeout=10)
            if response.status_code == 200 and "LALS" in response.text:
                print_success("Root: OK")
            else:
                print_warning(f"Root: {response.status_code}")
                all_passed = False
        except Exception as e:
            print_error(f"Root: {str(e)[:100]}")
            all_passed = False
        
        # Test 2: OpenAI endpoint
        print("\nTesting OpenAI endpoint (/v1/chat/completions)...")
        try:
            response = requests.post(
                f"{base_url}/v1/chat/completions",
                json={
                    "model": "qwen3-0.6b",
                    "messages": [{"role": "user", "content": "Hello!"}],
                    "max_tokens": 50
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                if "choices" in data:
                    content = data["choices"][0]["message"]["content"]
                    print_success("OpenAI: OK")
                    print(f"  Response: {content[:100]}...")
                else:
                    print_warning("OpenAI: No choices in response")
                    all_passed = False
            else:
                print_error(f"OpenAI: {response.status_code}")
                print(f"  {response.text[:200]}")
                all_passed = False
        except Exception as e:
            print_error(f"OpenAI: {str(e)[:100]}")
            all_passed = False
        
        # Test 3: Anthropic endpoint
        print("\nTesting Anthropic endpoint (/v1/messages)...")
        try:
            response = requests.post(
                f"{base_url}/v1/messages",
                json={
                    "model": "qwen3-0.6b",
                    "max_tokens": 50,
                    "messages": [{"role": "user", "content": "Hello!"}]
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                if "content" in data:
                    content = data["content"][0]["text"]
                    print_success("Anthropic: OK")
                    print(f"  Response: {content[:100]}...")
                else:
                    print_warning("Anthropic: No content in response")
                    all_passed = False
            else:
                print_error(f"Anthropic: {response.status_code}")
                print(f"  {response.text[:200]}")
                all_passed = False
        except Exception as e:
            print_error(f"Anthropic: {str(e)[:100]}")
            all_passed = False
        
        # Test 4: Documentation
        print("\nTesting documentation endpoint...")
        try:
            response = requests.get(f"{base_url}/docs", timeout=10)
            if response.status_code == 200:
                print_success("Docs: OK")
            else:
                print_warning(f"Docs: {response.status_code}")
        except Exception as e:
            print_warning(f"Docs: {str(e)[:50]}")
        
        return all_passed
    
    def run_all(self) -> bool:
        """Run the complete deployment process."""
        print("\n" + "=" * 60)
        print("LALS Multi-API Gateway v2.0 - Deployment")
        print("=" * 60 + "\n")
        
        # Check prerequisites
        if not self.check_prerequisites():
            print_error("Prerequisites check failed")
            return False
        
        # Setup repository
        if not self.setup_repository():
            print_error("Repository setup failed")
            return False
        
        # Install dependencies
        if not self.install_dependencies():
            print_error("Dependency installation failed")
            return False
        
        # Download model
        if not self.download_model():
            print_warning("Model download failed, continuing anyway...")
        
        # Test locally (optional)
        print("\n")
        response = input("Test locally before deployment? (y/n): ")
        if response.lower() in ['y', 'yes']:
            if not self.test_locally():
                print_warning("Local tests had issues, but continuing...")
        
        # Deploy to Wasmer
        print("\n")
        response = input("Deploy to Wasmer Edge? (y/n): ")
        if response.lower() in ['y', 'yes']:
            if not self.deploy_wasmer():
                print_error("Deployment failed")
                return False
            
            # Wait for deployment
            print("\nWaiting for deployment to complete...")
            time.sleep(10)
            
            # Verify
            if not self.verify_deployment():
                print_warning("Some endpoints may need more time to propagate")
        
        print("\n" + "=" * 60)
        print("Deployment Complete!")
        print("=" * 60)
        print(f"\nAPI Endpoints:")
        print(f"  • OpenAI:     {self.config.deploy_url}/v1/chat/completions")
        print(f"  • Anthropic:  {self.config.deploy_url}/v1/messages")
        print(f"  • Documentation: {self.config.deploy_url}/docs")
        print()
        
        return True


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="LALS Multi-API Gateway Deployment Script"
    )
    parser.add_argument(
        "--action",
        choices=["all", "prereq", "setup", "deps", "model", "test-local", "deploy", "verify"],
        default="all",
        help="Action to perform (default: all)"
    )
    
    args = parser.parse_args()
    
    manager = DeploymentManager()
    
    if args.action == "prereq":
        manager.check_prerequisites()
    elif args.action == "setup":
        manager.setup_repository()
    elif args.action == "deps":
        manager.install_dependencies()
    elif args.action == "model":
        manager.download_model()
    elif args.action == "test-local":
        manager.test_locally()
    elif args.action == "deploy":
        manager.deploy_wasmer()
    elif args.action == "verify":
        manager.verify_deployment()
    else:
        manager.run_all()


if __name__ == "__main__":
    main()
