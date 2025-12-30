# LALS Multi-API Gateway - Deployment Guide

## Current Project Status

The LALS Multi-API Gateway v2.0 has been successfully implemented with full support for both OpenAI and Anthropic API protocols. All code has been committed to the GitHub repository and is ready for deployment to Wasmer Edge. The following components are in place:

The core architecture includes a modular design with separate modules for configuration management, model engine wrapping, protocol handling for both OpenAI and Anthropic formats, and route handlers that expose the endpoints. The verification report confirms that the remote service at `lals-ai.wasmer.app` currently returns the original "Hello World" template, indicating that the new implementation needs to be deployed to update the live service.

The repository contains all necessary files for deployment, including the `wasmer.toml` configuration file, `deploy.py` automation script, comprehensive test suites, and detailed documentation. The model file (Qwen3-0.6B-Q4_K_M-GGUF) should be downloaded separately and placed in the `models/` directory as specified in the `.gitignore` file to comply with GitHub's file size limits.

## Deployment Prerequisites

Before proceeding with deployment, ensure that you have the following prerequisites installed and configured on your local machine. These requirements are essential for a successful deployment to Wasmer Edge.

### Wasmer CLI Installation

The Wasmer command-line interface is required for deploying applications to the Wasmer Edge network. If you haven't already installed Wasmer, you can do so using the official installation script. Open your terminal and execute the following command to download and install Wasmer:

```bash
curl https://get.wasmer.io -sSf | sh
```

After installation, verify that Wasmer is correctly installed by checking its version:

```bash
wasmer --version
```

You should see output similar to `wasmer version 4.3.0` or newer. If the installation fails or you encounter issues, refer to the official Wasmer documentation at https://docs.wasmer.io for troubleshooting guidance.

### Git Configuration

Ensure that your Git repository is properly configured and that you have the necessary permissions to push changes to the remote repository. The deployment process will pull the latest code from GitHub, so all changes must be committed and pushed before deployment.

Verify your remote repository configuration:

```bash
git remote -v
git status
```

You should see that your local branch is up to date with the origin branch, and all changes have been committed. If you have uncommitted changes, commit them before proceeding with deployment.

### Python Environment

The deployment requires Python 3.10 or higher with all dependencies installed. Create a virtual environment and install the required packages:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The `requirements.txt` file includes all necessary dependencies: `fastapi` for the web framework, `uvicorn` for the ASGI server, `llama-cpp-python` for model inference, `pydantic` for data validation, and `pydantic-settings` for configuration management.

### Model File Preparation

The Qwen3-0.6B-Q4_K_M-GGUF model file must be downloaded and placed in the `models/` directory before running the application locally or deploying to Wasmer. Download the model from the specified Hugging Face repository:

```bash
mkdir -p models
wget -O models/qwen3-0.6b-q4_k_m.gguf \
  https://huggingface.co/enacimie/Qwen3-0.6B-Q4_K_M-GGUF/resolve/main/qwen3-0.6b-q4_k_m.gguf
```

This 462MB model file is excluded from Git version control via `.gitignore` to comply with repository file size limits. Ensure this file is present in your local environment before testing or deployment.

## Deployment Options

There are multiple deployment options available for the LALS Multi-API Gateway. Choose the option that best fits your infrastructure requirements and workflow preferences.

### Option 1: Direct Wasmer Edge Deployment (Recommended)

The simplest deployment method uses the Wasmer CLI directly. Navigate to your project directory and execute the deployment command:

```bash
cd /path/to/lals
wasmer deploy
```

The Wasmer CLI will read the `wasmer.toml` configuration file, which specifies the deployment target as Wasmer Edge, the package settings, and the build command. The deployment process will automatically install Python dependencies, build the application, and deploy it to the Wasmer Edge network.

After deployment completes, Wasmer will provide you with a deployment URL. For the existing `lals-ai.wasmer.app` namespace, the deployment should update the existing service. Note that DNS propagation may take a few minutes, so the new endpoints may not be immediately available.

### Option 2: Automated Deployment Script

The `deploy.py` script provides an automated deployment workflow that handles preparation steps, dependency installation, and deployment execution. Run the script from your project directory:

```bash
python deploy.py
```

The script performs the following operations: it verifies the Git repository status, pulls the latest changes from the remote, installs or updates Python dependencies, and optionally runs the deployment command. You can modify the script to include additional pre-deployment checks or post-deployment verification steps as needed for your workflow.

### Option 3: Docker Container Deployment

For containerized deployments, build a Docker image containing the application and its dependencies:

```bash
docker build -t lals-gateway:latest .
docker tag lals-gateway:latest registry.wasmer.io/<your-namespace>/lals-gateway:latest
docker push registry.wasmer.io/<your-namespace>/lals-gateway:latest
wasmer deploy
```

This approach ensures consistent deployments across different environments and provides better control over the runtime environment. The Docker build process installs all dependencies within the container, eliminating the need for external package management during deployment.

### Option 4: Manual Deployment

For environments where automated tools are not available, perform a manual deployment:

1. Pull the latest code from GitHub:
   ```bash
   git pull origin main
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Test locally to verify functionality:
   ```bash
   python -m src.main
   ```

4. In a separate terminal, test the local endpoints:
   ```bash
   curl http://localhost:8000/v1/chat/completions
   ```

5. Once verified, deploy to Wasmer using the Wasmer dashboard or CLI.

## Post-Deployment Verification

After deployment completes, verify that all endpoints are functioning correctly. The following curl commands test the major API endpoints:

### OpenAI-Compatible Endpoints

Test the chat completions endpoint using the OpenAI format:

```bash
curl -X POST https://lals-ai.wasmer.app/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-0.6b",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

Expected response format includes the standard OpenAI chat completion structure with `id`, `object`, `created`, `model`, `choices`, and `usage` fields.

Test the models listing endpoint:

```bash
curl https://lals-ai.wasmer.app/v1/models
```

### Anthropic-Compatible Endpoints

Test the messages endpoint using the Anthropic format:

```bash
curl -X POST https://lals-ai.wasmer.app/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-0.6b",
    "max_tokens": 50,
    "system": "You are a helpful assistant.",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ]
  }'
```

Expected response format includes the Anthropic message structure with `id`, `type`, `role`, `content`, `model`, `stop_reason`, and `usage` fields.

### Universal Endpoint

Test the auto-detecting universal inference endpoint:

```bash
curl -X POST https://lals-ai.wasmer.app/v1/inference \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-0.6b",
    "messages": [
      {"role": "user", "content": "Explain quantum computing in simple terms."}
    ],
    "max_tokens": 100
  }'
```

This endpoint automatically detects the request format and routes it to the appropriate protocol handler.

## API Documentation

After deployment, access the interactive API documentation at the following endpoints. The Swagger UI provides a user-friendly interface for exploring and testing the API:

- **Swagger UI**: https://lals-ai.wasmer.app/docs
- **ReDoc**: https://lals-ai.wasmer.app/redoc
- **OpenAPI Schema**: https://lals-ai.wasmer.app/openapi.json

## Troubleshooting

If you encounter issues during or after deployment, the following troubleshooting steps may help resolve common problems.

### Deployment Failures

If `wasmer deploy` fails, first verify that you are logged in to Wasmer:

```bash
wasmer login
```

If you are not logged in, authenticate using your Wasmer credentials. Additionally, ensure that your Wasmer account has permission to deploy to the target namespace.

### Model Loading Errors

If the application starts but model inference fails, verify that the model file is present in the correct location:

```bash
ls -lh models/
```

You should see the `qwen3-0.6b-q4_k_m.gguf` file. If it is missing, download it using the command specified in the Model File Preparation section.

### Endpoint Not Found Errors

If you receive "Not Found" errors when testing endpoints, the application may not have loaded correctly. Check the application logs in the Wasmer dashboard or redeploy to ensure the latest code is active.

### Performance Issues

If response times are slower than expected, consider the following optimizations. The Qwen3-0.6B model is designed for efficiency, but response times depend on the computational resources available on Wasmer Edge. For production workloads, you may need to adjust model parameters such as `max_tokens`, `temperature`, or use model quantization with different precision levels.

## Additional Resources

For further information about the LALS Multi-API Gateway and its capabilities, refer to the following documentation files included in the repository:

- `README.md` - Overview of the LALS project
- `README_MULTI_API.md` - Detailed documentation of the Multi-API Gateway architecture
- `README_LALS.md` - LALS-specific documentation
- `VERIFICATION_REPORT.md` - Verification results and implementation status

For questions or issues related to Wasmer deployment, consult the official documentation at https://docs.wasmer.io or visit the Wasmer community support channels.

## Quick Reference

The following commands provide a quick reference for common deployment tasks:

```bash
# Clone repository
git clone https://github.com/likhon-ai-labs/lals.git
cd lals

# Setup environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download model
mkdir -p models
wget -O models/qwen3-0.6b-q4_k_m.gguf \
  https://huggingface.co/enacimie/Qwen3-0.6B-Q4_K_M-GGUF/resolve/main/qwen3-0.6b-q4_k_m.gguf

# Test locally
python -m src.main

# Deploy to Wasmer
wasmer deploy

# Verify deployment
curl https://lals-ai.wasmer.app/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-0.6b", "messages": [{"role": "user", "content": "Test"}], "max_tokens": 10}
```

The LALS Multi-API Gateway is now ready for deployment. Execute the deployment steps outlined in this guide to make the service available at `lals-ai.wasmer.app` with full OpenAI and Anthropic protocol support.
