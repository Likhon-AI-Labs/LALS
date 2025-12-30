#!/bin/bash

# LALS Multi-API Gateway Deployment Script
# =========================================
# This script deploys LALS to Wasmer Edge
# Run this on your local machine with Wasmer CLI installed

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_step() {
    echo -e "${BLUE}==>${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Print banner
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║     LALS Multi-API Gateway v2.0 - Deployment Script       ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Step 1: Check prerequisites
print_step "Checking prerequisites..."

# Check Python version
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1)
    print_success "Python found: $PYTHON_VERSION"
else
    print_error "Python 3 not found. Please install Python 3.10+"
    exit 1
fi

# Check Git
if command -v git &> /dev/null; then
    print_success "Git found: $(git --version)"
else
    print_error "Git not found. Please install Git"
    exit 1
fi

# Check Wasmer
if command -v wasmer &> /dev/null; then
    WASMER_VERSION=$(wasmer --version)
    print_success "Wasmer CLI found: $WASMER_VERSION"
else
    print_warning "Wasmer CLI not found"
    echo ""
    echo "Installing Wasmer..."
    curl https://get.wasmer.io -sSf | sh
    source ~/.wasmer/wasmer.sh
    print_success "Wasmer installed"
fi

echo ""

# Step 2: Clone or update repository
print_step "Setting up repository..."

if [ -d "lals" ]; then
    print_warning "Directory 'lals' already exists"
    read -p "Do you want to update it? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cd lals
        git pull origin main
        print_success "Repository updated"
    fi
else
    git clone https://github.com/Likhon-AI-Labs/LALS.git
    cd lals
    print_success "Repository cloned"
fi

echo ""

# Step 3: Install dependencies
print_step "Installing dependencies..."

python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies
pip install -r requirements.txt

print_success "Dependencies installed"
echo ""

# Step 4: Download model (if not present)
print_step "Checking model file..."

MODEL_FILE="models/qwen3-0.6b-q4_k_m.gguf"

if [ -f "$MODEL_FILE" ]; then
    print_success "Model file found: $MODEL_FILE"
    ls -lh "$MODEL_FILE"
else
    print_warning "Model file not found"
    echo ""
    echo "Downloading Qwen3-0.6B model (this may take a few minutes)..."
    
    mkdir -p models
    
    wget -O "$MODEL_FILE" \
        "https://huggingface.co/enacimie/Qwen3-0.6B-Q4_K_M-GGUF/resolve/main/qwen3-0.6b-q4_k_m.gguf"
    
    print_success "Model downloaded"
    ls -lh "$MODEL_FILE"
fi

echo ""

# Step 5: Test locally
print_step "Testing locally..."

# Start server in background
source venv/bin/activate
python -m src.main &
SERVER_PID=$!

# Wait for server to start
sleep 5

# Test endpoints
echo ""
echo "Testing endpoints..."

# Test root
if curl -s http://localhost:8000/ | grep -q "LALS"; then
    print_success "Root endpoint: OK"
else
    print_warning "Root endpoint: May need more time to start"
fi

# Test OpenAI
if curl -s -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"qwen3-0.6b","messages":[{"role":"user","content":"Hi"}],"max_tokens":10}' \
    | grep -q "choices"; then
    print_success "OpenAI endpoint: OK"
else
    print_warning "OpenAI endpoint: Testing..."
fi

# Test Anthropic
if curl -s -X POST http://localhost:8000/v1/messages \
    -H "Content-Type: application/json" \
    -d '{"model":"qwen3-0.6b","max_tokens":10,"messages":[{"role":"user","content":"Hi"}]}' \
    | grep -q "content"; then
    print_success "Anthropic endpoint: OK"
else
    print_warning "Anthropic endpoint: Testing..."
fi

# Stop local server
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

echo ""

# Step 6: Deploy to Wasmer Edge
print_step "Deploying to Wasmer Edge..."

# Check if logged in
if ! wasmer whoami &> /dev/null; then
    print_warning "Not logged in to Wasmer"
    echo "Please login:"
    wasmer login
fi

# Deploy
echo ""
echo "Deploying to Wasmer Edge..."
wasmer deploy

echo ""

# Step 7: Verify deployment
print_step "Verifying deployment..."

DEPLOY_URL="https://lals-ai.wasmer.app"

echo "Testing live endpoints..."

# Test root
echo -n "Root: "
if curl -s "$DEPLOY_URL/" | grep -q "LALS"; then
    print_success "OK"
else
    print_warning "May need time to propagate"
fi

# Test OpenAI
echo -n "OpenAI (/v1/chat/completions): "
OPENAI_RESPONSE=$(curl -s -X POST "$DEPLOY_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"qwen3-0.6b","messages":[{"role":"user","content":"Hello!"}],"max_tokens":50}' 2>&1)

if echo "$OPENAI_RESPONSE" | grep -q "choices"; then
    print_success "OK"
    echo "$OPENAI_RESPONSE" | python3 -m json.tool | head -20
else
    print_error "Failed"
    echo "$OPENAI_RESPONSE"
fi

# Test Anthropic
echo -n "Anthropic (/v1/messages): "
ANTHROPIC_RESPONSE=$(curl -s -X POST "$DEPLOY_URL/v1/messages" \
    -H "Content-Type: application/json" \
    -d '{"model":"qwen3-0.6b","max_tokens":50,"messages":[{"role":"user","content":"Hello!"}]}' 2>&1)

if echo "$ANTHROPIC_RESPONSE" | grep -q "content"; then
    print_success "OK"
    echo "$ANTHROPIC_RESPONSE" | python3 -m json.tool | head -20
else
    print_error "Failed"
    echo "$ANTHROPIC_RESPONSE"
fi

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                  Deployment Complete!                      ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "🌐 Live API endpoints:"
echo "   • OpenAI:     $DEPLOY_URL/v1/chat/completions"
echo "   • Anthropic:  $DEPLOY_URL/v1/messages"
echo "   • Docs:       $DEPLOY_URL/docs"
echo ""
echo "📖 Documentation: README_MULTI_API.md"
echo "📊 Report: VERIFICATION_REPORT.md"
echo ""
