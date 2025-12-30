#!/usr/bin/env python3
"""
LALS Multi-API Gateway - Automated Deployment Script

This script automates the deployment process for the LALS Multi-API Gateway
to Wasmer Edge. It handles environment setup, dependency installation,
model download, and deployment execution.

Usage:
    python deploy_local.py

Requirements:
    - Python 3.10+
    - Git
    - Wasmer CLI (will be installed if not present)
    - wget or curl for model download
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_step(step_number: int, total_steps: int, message: str):
    """Print a formatted step message."""
    print(f"\n{Colors.BLUE}{Colors.BOLD}[{step_number}/{total_steps}]{Colors.END} {message}")


def print_success(message: str):
    """Print a success message."""
    print(f"{Colors.GREEN}✓{Colors.END} {message}")


def print_warning(message: str):
    """Print a warning message."""
    print(f"{Colors.YELLOW}⚠{Colors.END} {message}")


def print_error(message: str):
    """Print an error message."""
    print(f"{Colors.RED}✗{Colors.END} {message}")


def run_command(command: list, description: str, check_error: bool = True) -> bool:
    """Run a shell command and handle errors."""
    print(f"  Running: {' '.join(command)}")
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=300
        )
        if result.returncode == 0:
            if result.stdout.strip():
                print(f"    {result.stdout.strip()}")
            return True
        else:
            if check_error:
                print_error(f"{description} failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print_error(f"{description} timed out")
        return False
    except Exception as e:
        print_error(f"{description} error: {str(e)}")
        return False


def check_command_available(command: str) -> bool:
    """Check if a command is available in the system."""
    return shutil.which(command) is not None


def install_wasmer() -> bool:
    """Install Wasmer CLI if not present."""
    print_step(1, 5, "Checking Wasmer CLI installation")
    
    if check_command_available("wasmer"):
        print_success("Wasmer CLI is already installed")
        version_result = subprocess.run(
            ["wasmer", "--version"],
            capture_output=True,
            text=True
        )
        print(f"  Version: {version_result.stdout.strip()}")
        return True
    
    print_warning("Wasmer CLI not found. Installing...")
    
    # Detect OS and install accordingly
    system = platform.system().lower()
    
    if system == "linux":
        install_cmd = ["curl", "https://get.wasmer.io", "-sSf"]
    elif system == "darwin":
        install_cmd = ["curl", "https://get.wasmer.io", "-sSf"]
    elif system == "windows":
        install_cmd = ["powershell", "-c", "iwr https://get.wasmer.io -useb | iex"]
    else:
        print_error(f"Unsupported operating system: {system}")
        return False
    
    if run_command(install_cmd, "Installing Wasmer CLI", check_error=False):
        print_success("Wasmer CLI installed successfully")
        
        # Add wasmer to PATH if installed in non-standard location
        wasmer_path = Path.home() / ".wasmer" / "bin"
        if wasmer_path.exists():
            current_path = os.environ.get("PATH", "")
            if str(wasmer_path) not in current_path:
                print_warning(f"Wasmer installed at {wasmer_path}")
                print_warning("You may need to add this to your PATH")
        
        return True
    
    print_error("Failed to install Wasmer CLI")
    print("Please install manually: https://docs.wasmer.io/install")
    return False


def setup_python_environment() -> bool:
    """Set up Python virtual environment and install dependencies."""
    print_step(2, 5, "Setting up Python environment")
    
    script_dir = Path(__file__).parent.absolute()
    os.chdir(script_dir)
    
    # Check Python version
    try:
        result = subprocess.run(
            ["python3", "--version"],
            capture_output=True,
            text=True
        )
        python_version = result.stdout.strip()
        print(f"  Python version: {python_version}")
        
        # Check if version is 3.10+
        version_parts = python_version.split()[1].split(".")
        major, minor = int(version_parts[0]), int(version_parts[1])
        if major < 3 or (major == 3 and minor < 10):
            print_error("Python 3.10+ is required")
            return False
    except Exception as e:
        print_error(f"Failed to check Python version: {e}")
        return False
    
    # Create virtual environment
    venv_path = script_dir / "venv"
    
    if venv_path.exists():
        print_warning("Virtual environment already exists")
    else:
        print("  Creating virtual environment...")
        if not run_command(["python3", "-m", "venv", "venv"], "Creating virtual environment"):
            return False
        print_success("Virtual environment created")
    
    # Activate virtual environment and install dependencies
    print("  Installing dependencies...")
    
    if platform.system().lower() == "windows":
        pip_cmd = [str(venv_path / "Scripts" / "pip.exe"), "install", "-r", "requirements.txt"]
    else:
        pip_cmd = [str(venv_path / "bin" / "pip"), "install", "-r", "requirements.txt"]
    
    if run_command(pip_cmd, "Installing Python dependencies"):
        print_success("Dependencies installed successfully")
        return True
    
    return False


def download_model() -> bool:
    """Download the Qwen3-0.6B model file."""
    print_step(3, 5, "Downloading model file")
    
    script_dir = Path(__file__).parent.absolute()
    models_dir = script_dir / "models"
    
    # Create models directory if it doesn't exist
    models_dir.mkdir(exist_ok=True)
    
    model_file = models_dir / "qwen3-0.6b-q4_k_m.gguf"
    
    if model_file.exists():
        file_size = model_file.stat().st_size / (1024 * 1024)
        print_success(f"Model file already exists ({file_size:.1f} MB)")
        return True
    
    print_warning("This is a large file (~462 MB). Download may take several minutes.")
    
    model_url = "https://huggingface.co/enacimie/Qwen3-0.6B-Q4_K_M-GGUF/resolve/main/qwen3-0.6b-q4_k_m.gguf"
    
    # Check for wget or curl
    if check_command_available("wget"):
        download_cmd = ["wget", "-O", str(model_file), model_url]
    elif check_command_available("curl"):
        download_cmd = ["curl", "-L", "-o", str(model_file), model_url]
    else:
        print_error("Neither wget nor curl found. Please install one of them.")
        return False
    
    print(f"  Downloading from: {model_url}")
    
    if run_command(download_cmd, "Downloading model file", check_error=False):
        if model_file.exists():
            file_size = model_file.stat().st_size / (1024 * 1024)
            print_success(f"Model downloaded successfully ({file_size:.1f} MB)")
            return True
    
    print_error("Failed to download model file")
    print(f"Please download manually and place at: {model_file}")
    return False


def verify_repository() -> bool:
    """Verify repository status and remote configuration."""
    print_step(4, 5, "Verifying repository status")
    
    script_dir = Path(__file__).parent.absolute()
    os.chdir(script_dir)
    
    # Check if this is a git repository
    if not (script_dir / ".git").exists():
        print_error("Not a git repository")
        return False
    
    # Check remote configuration
    result = subprocess.run(
        ["git", "remote", "-v"],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0 and result.stdout.strip():
        print_success("Git remote configured")
        print(f"  {result.stdout.strip().split(chr(10))[0]}")
    else:
        print_warning("No git remote configured")
    
    # Check git status
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0 and not result.stdout.strip():
        print_success("Working directory is clean")
    else:
        print_warning("Working directory has uncommitted changes")
        print(f"  {result.stdout.strip()}")
    
    # Check if code is pushed
    result = subprocess.run(
        ["git", "log", "--oneline", "-1"],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print_success(f"Latest commit: {result.stdout.strip()}")
    
    return True


def deploy_to_wasmer() -> bool:
    """Deploy the application to Wasmer Edge."""
    print_step(5, 5, "Deploying to Wasmer Edge")
    
    # Check if wasmer is available
    if not check_command_available("wasmer"):
        print_error("Wasmer CLI not found")
        print("Please install Wasmer first")
        return False
    
    # Check wasmer login status
    result = subprocess.run(
        ["wasmer", "whoami"],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print_warning("Not logged in to Wasmer")
        print("Please run: wasmer login")
        print("Continuing with deployment command...")
    
    # Run wasmer deploy
    print("  Running: wasmer deploy")
    print("  This may take a few minutes...")
    
    try:
        result = subprocess.run(
            ["wasmer", "deploy", "--verbose"],
            capture_output=True,
            text=True,
            timeout=600
        )
        
        print(result.stdout)
        
        if result.returncode == 0:
            print_success("Deployment initiated successfully!")
            print("\nDeployment may take a few minutes to propagate.")
            print("Check the deployment status at: https://wasmer.io/deployments")
            return True
        else:
            print_error("Deployment failed")
            if result.stderr:
                print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print_error("Deployment timed out")
        return False
    except Exception as e:
        print_error(f"Deployment error: {str(e)}")
        return False


def test_deployment():
    """Test the deployed endpoints."""
    print("\n" + "=" * 60)
    print("Testing Deployed Endpoints")
    print("=" * 60)
    
    endpoints = [
        ("GET", "https://lals-ai.wasmer.app/", None, "Root endpoint"),
        ("GET", "https://lals-ai.wasmer.app/v1/models", None, "Models list"),
        ("POST", "https://lals-ai.wasmer.app/v1/chat/completions", 
         {"model": "qwen3-0.6b", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 10},
         "OpenAI chat completions"),
        ("POST", "https://lals-ai.wasmer.app/v1/messages",
         {"model": "qwen3-0.6b", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 10},
         "Anthropic messages"),
    ]
    
    import urllib.request
    import json
    
    for method, url, data, description in endpoints:
        print(f"\nTesting: {description}")
        print(f"  URL: {url}")
        
        try:
            if data:
                data_bytes = json.dumps(data).encode('utf-8')
                req = urllib.request.Request(
                    url,
                    data=data_bytes,
                    headers={'Content-Type': 'application/json'},
                    method='POST'
                )
            else:
                req = urllib.request.Request(url, method=method)
            
            with urllib.request.urlopen(req, timeout=10) as response:
                status = response.status
                body = response.read().decode('utf-8')
                print(f"  Status: {status}")
                print(f"  Response: {body[:200]}...")
                print_success("Endpoint is working!")
        except Exception as e:
            print_error(f"Endpoint test failed: {str(e)}")
    
    print("\nManual testing commands:")
    print("-" * 40)
    print("curl https://lals-ai.wasmer.app/v1/chat/completions \\")
    print("  -H 'Content-Type: application/json' \\")
    print("  -d '{\"model\": \"qwen3-0.6b\", \"messages\": [{\"role\": \"user\", \"content\": \"Hi\"}], \"max_tokens\": 10}'")


def main():
    """Main deployment function."""
    print("\n" + "=" * 60)
    print("LALS Multi-API Gateway - Deployment Script")
    print("=" * 60)
    print("\nThis script will:")
    print("  1. Check/Install Wasmer CLI")
    print("  2. Set up Python environment")
    print("  3. Download model file")
    print("  4. Verify repository status")
    print("  5. Deploy to Wasmer Edge")
    
    steps = [
        ("Checking Wasmer CLI", install_wasmer),
        ("Setting up Python environment", setup_python_environment),
        ("Downloading model file", download_model),
        ("Verifying repository", verify_repository),
        ("Deploying to Wasmer Edge", deploy_to_wasmer),
    ]
    
    for i, (name, func) in enumerate(steps, 1):
        if not func():
            print_error(f"Step '{name}' failed")
            print("\nDeployment aborted. Please resolve the error and try again.")
            print("\nFor manual deployment, see DEPLOYMENT_GUIDE.md")
            sys.exit(1)
    
    print("\n" + "=" * 60)
    print_success("Deployment completed successfully!")
    print("=" * 60)
    
    # Optionally test deployment
    print("\nWould you like to test the deployed endpoints? (y/n)")
    choice = input("> ").strip().lower()
    
    if choice == "y" or choice == "yes":
        test_deployment()
    
    print("\nNext steps:")
    print("  1. Wait a few minutes for DNS propagation")
    print("  2. Test the endpoints using the commands above")
    print("  3. Check Wasmer dashboard for deployment status")
    print("  4. Monitor logs and metrics in production")


if __name__ == "__main__":
    main()
