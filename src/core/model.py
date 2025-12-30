"""
Model Engine Module
====================
Handles LLM model loading and inference operations.
Supports both synchronous and streaming generation.
Includes automatic model download from MODEL_URL if model file is missing.
"""

import asyncio
import logging
import os
import urllib.request
import hashlib
from pathlib import Path
from typing import Optional, AsyncIterator, Dict, Any
from contextlib import contextmanager

from llama_cpp import Llama

from .config import get_config, ModelConfig

logger = logging.getLogger(__name__)

# Global model instance
_model_instance: Optional[Llama] = None


def download_file(url: str, dest_path: Path, expected_hash: Optional[str] = None) -> bool:
    """
    Download a file from a URL with progress logging.
    
    Args:
        url: URL to download from
        dest_path: Destination file path
        expected_hash: Optional SHA256 hash to verify download
    
    Returns:
        True if download successful, False otherwise
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading model from: {url}")
    logger.info(f"Destination: {dest_path}")
    
    try:
        # Set up request with headers to simulate browser
        req = urllib.request.Request(
            url,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/octet-stream,*/*',
            }
        )
        
        with urllib.request.urlopen(req, timeout=300) as response:
            total_size = int(response.headers.get('Content-Length', 0))
            logger.info(f"Model size: {total_size / (1024*1024):.1f} MB")
            
            downloaded = 0
            chunk_size = 8192
            
            with open(dest_path, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    
                    downloaded += len(chunk)
                    f.write(chunk)
                    
                    # Log progress every 10%
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        if percent > 0 and int(percent) % 10 == 0:
                            logger.info(f"Downloaded: {percent:.0f}% ({downloaded / (1024*1024):.1f} MB)")
        
        # Verify hash if provided
        if expected_hash:
            logger.info("Verifying model file hash...")
            sha256_hash = hashlib.sha256()
            with open(dest_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256_hash.update(chunk)
            actual_hash = sha256_hash.hexdigest()
            
            if actual_hash != expected_hash:
                logger.error(f"Hash mismatch! Expected {expected_hash}, got {actual_hash}")
                dest_path.unlink()
                return False
            logger.info("Model file hash verified successfully")
        
        file_size = dest_path.stat().st_size / (1024 * 1024)
        logger.info(f"Model downloaded successfully: {dest_path} ({file_size:.1f} MB)")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False


class ModelEngine:
    """
    Model inference engine wrapping llama-cpp-python.
    
    Provides both synchronous and asynchronous generation methods
    with support for streaming responses.
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize the model engine.
        
        Args:
            config: Model configuration. If None, loads from settings.
        """
        self.config = config or get_config().model
        self._model: Optional[Llama] = None
        self._lock = asyncio.Lock()
    
    def load(self) -> None:
        """Load the model into memory. Downloads model if not present."""
        if self._model is not None:
            logger.warning("Model already loaded, skipping reload")
            return
        
        model_path = Path(self.config.path)
        
        # Check if model file exists
        if not model_path.exists():
            # Check for MODEL_URL environment variable
            model_url = os.environ.get("MODEL_URL")
            
            if model_url:
                logger.info("Model file not found, attempting download from MODEL_URL")
                if download_file(model_url, model_path):
                    logger.info("Model download complete, proceeding with load")
                else:
                    raise RuntimeError(
                        f"Model file not found and download failed. "
                        f"Please ensure {model_path} exists or MODEL_URL is accessible."
                    )
            else:
                raise RuntimeError(
                    f"Model file not found at {model_path}. "
                    f"Please either:\n"
                    f"  1. Place the model file at {model_path}\n"
                    f"  2. Set the MODEL_URL environment variable"
                )
        
        logger.info(f"Loading model from: {self.config.path}")
        logger.info(f"Context size: {self.config.n_ctx}")
        logger.info(f"Threads: {self.config.n_threads}")
        
        self._model = Llama(
            model_path=self.config.path,
            n_ctx=self.config.n_ctx,
            n_threads=self.config.n_threads,
            n_gpu_layers=self.config.n_gpu_layers,
            verbose=False
        )
        
        logger.info("Model loaded successfully")
    
    def unload(self) -> None:
        """Unload the model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
            logger.info("Model unloaded")
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None
    
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[float] = None,
        stop: Optional[list] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            stop: Stop sequences
            stream: Ignored (use streaming method instead)
        
        Returns:
            Dictionary containing generated text and token usage
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        max_tokens = max_tokens or self.config.max_tokens_default
        temperature = temperature if temperature is not None else self.config.temperature
        top_p = top_p if top_p is not None else self.config.top_p
        top_k = top_k if top_k is not None else self.config.top_k
        stop = stop or []
        
        logger.debug(f"Generating with max_tokens={max_tokens}, temp={temperature}")
        
        output = self._model(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
            echo=False,
            stream=False
        )
        
        return {
            "text": output["choices"][0]["text"],
            "prompt_tokens": output["usage"]["prompt_tokens"],
            "completion_tokens": output["usage"]["completion_tokens"],
            "total_tokens": output["usage"]["total_tokens"],
            "stop_reason": output["choices"][0].get("finish_reason", "stop")
        }
    
    async def generate_async(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[float] = None,
        stop: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Asynchronous generation.
        
        Runs generation in a thread pool to avoid blocking the event loop.
        """
        loop = asyncio.get_event_loop()
        
        result = await loop.run_in_executor(
            None,
            lambda: self.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stop=stop
            )
        )
        
        return result
    
    def generate_streaming(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """
        Generate text with streaming token output.
        
        Args:
            prompt: Input prompt text
            **kwargs: Additional generation parameters
        
        Yields:
            Individual tokens as they are generated
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        max_tokens = kwargs.get("max_tokens") or self.config.max_tokens_default
        temperature = kwargs.get("temperature", self.config.temperature)
        top_p = kwargs.get("top_p", self.config.top_p)
        stop = kwargs.get("stop", [])
        
        stream = self._model(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            echo=False,
            stream=True
        )
        
        for chunk in stream:
            if chunk["choices"][0]["finish_reason"] is None:
                token = chunk["choices"][0]["text"]
                yield token
            else:
                break


# Global model engine instance
_engine: Optional[ModelEngine] = None


def get_model_engine() -> ModelEngine:
    """Get the global model engine instance."""
    global _engine
    if _engine is None:
        _engine = ModelEngine()
    return _engine


def initialize_model_engine(config: Optional[ModelConfig] = None) -> ModelEngine:
    """Initialize and load the model engine."""
    global _engine
    _engine = ModelEngine(config)
    _engine.load()
    return _engine


def shutdown_model_engine() -> None:
    """Shutdown and unload the model engine."""
    global _engine
    if _engine is not None:
        _engine.unload()
        _engine = None


@contextmanager
def model_context(config: Optional[ModelConfig] = None):
    """Context manager for model lifecycle."""
    engine = initialize_model_engine(config)
    try:
        yield engine
    finally:
        shutdown_model_engine()
