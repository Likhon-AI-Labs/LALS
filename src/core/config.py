"""
Configuration Management Module
===============================
Handles all configuration for LALS Multi-API Gateway.
Supports environment variables, config files, and default values.
"""

import os
from typing import Optional
from functools import lru_cache
from pydantic import BaseModel, Field


class DatabaseConfig(BaseModel):
    """Database configuration settings."""
    
    host: str = Field(
        default="localhost",
        description="Database host address"
    )
    port: int = Field(
        default=5432,
        ge=1,
        le=65535,
        description="Database port"
    )
    name: str = Field(
        default="lals_db",
        description="Database name"
    )
    username: str = Field(
        default="",
        description="Database username"
    )
    password: str = Field(
        default="",
        description="Database password"
    )
    enabled: bool = Field(
        default=False,
        description="Enable database connection"
    )
    
    @property
    def connection_string(self) -> str:
        """Get database connection string."""
        if self.username and self.password:
            return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.name}"
        elif self.username:
            return f"postgresql://{self.username}@{self.host}:{self.port}/{self.name}"
        else:
            return f"postgresql://{self.host}:{self.port}/{self.name}"
    
    @property
    def is_configured(self) -> bool:
        """Check if database is properly configured."""
        return bool(self.host and self.enabled)


class ModelConfig(BaseModel):
    """Model configuration settings."""
    
    path: str = Field(
        default="./models/qwen3-0.6b-q4_k_m.gguf",
        description="Path to the GGUF model file"
    )
    n_ctx: int = Field(
        default=2048,
        ge=512,
        le=8192,
        description="Context window size in tokens"
    )
    n_threads: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Number of CPU threads for inference"
    )
    n_gpu_layers: int = Field(
        default=0,
        ge=0,
        description="Number of layers to offload to GPU (0 for CPU-only)"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Default temperature for generation"
    )
    top_p: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Top-p sampling parameter"
    )
    top_k: int = Field(
        default=40,
        ge=1,
        description="Top-k sampling parameter"
    )
    max_tokens_default: int = Field(
        default=1024,
        ge=1,
        le=4096,
        description="Default maximum tokens to generate"
    )


class APIConfig(BaseModel):
    """API configuration settings."""
    
    enable_openai_api: bool = Field(
        default=True,
        description="Enable OpenAI-compatible endpoints"
    )
    enable_anthropic_api: bool = Field(
        default=True,
        description="Enable Anthropic-compatible endpoints"
    )
    enable_universal_router: bool = Field(
        default=True,
        description="Enable auto-detection router"
    )
    enable_streaming: bool = Field(
        default=True,
        description="Enable streaming response support"
    )
    enable_caching: bool = Field(
        default=False,
        description="Enable response caching"
    )
    max_tokens_limit: int = Field(
        default=4096,
        ge=1,
        le=8192,
        description="Maximum allowed max_tokens in requests"
    )
    rate_limit_enabled: bool = Field(
        default=False,
        description="Enable rate limiting"
    )
    rate_limit_requests: int = Field(
        default=100,
        description="Maximum requests per minute"
    )


class ServerConfig(BaseModel):
    """Server configuration settings."""
    
    host: str = Field(
        default="0.0.0.0",
        description="Server host address"
    )
    port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="Server port"
    )
    workers: int = Field(
        default=1,
        ge=1,
        le=8,
        description="Number of worker processes"
    )
    reload: bool = Field(
        default=False,
        description="Enable auto-reload for development"
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    log_format: str = Field(
        default="json",
        description="Log format (json or text)"
    )


class Config(BaseModel):
    """Main configuration container."""
    
    model: ModelConfig = Field(
        default_factory=ModelConfig,
        description="Model configuration"
    )
    api: APIConfig = Field(
        default_factory=APIConfig,
        description="API configuration"
    )
    server: ServerConfig = Field(
        default_factory=ServerConfig,
        description="Server configuration"
    )
    database: DatabaseConfig = Field(
        default_factory=DatabaseConfig,
        description="Database configuration"
    )
    
    @classmethod
    def from_env(cls) -> "Config":
        """Create configuration from environment variables."""
        # Check if database is configured via environment
        db_host = os.environ.get("DB_HOST")
        db_enabled = bool(db_host)
        
        return cls(
            model=ModelConfig(
                path=os.environ.get("MODEL_PATH", ModelConfig().path),
                n_ctx=int(os.environ.get("MODEL_N_CTX", ModelConfig().n_ctx)),
                n_threads=int(os.environ.get("MODEL_N_THREADS", ModelConfig().n_threads)),
                n_gpu_layers=int(os.environ.get("MODEL_N_GPU_LAYERS", ModelConfig().n_gpu_layers)),
                temperature=float(os.environ.get("MODEL_TEMP", ModelConfig().temperature)),
                top_p=float(os.environ.get("MODEL_TOP_P", ModelConfig().top_p)),
                top_k=int(os.environ.get("MODEL_TOP_K", ModelConfig().top_k)),
                max_tokens_default=int(os.environ.get("MODEL_MAX_TOKENS_DEFAULT", ModelConfig().max_tokens_default)),
            ),
            api=APIConfig(
                enable_openai_api=os.environ.get("ENABLE_OPENAI_API", str(APIConfig().enable_openai_api)).lower() == "true",
                enable_anthropic_api=os.environ.get("ENABLE_ANTHROPIC_API", str(APIConfig().enable_anthropic_api)).lower() == "true",
                enable_universal_router=os.environ.get("ENABLE_UNIVERSAL_ROUTER", str(APIConfig().enable_universal_router)).lower() == "true",
                enable_streaming=os.environ.get("ENABLE_STREAMING", str(APIConfig().enable_streaming)).lower() == "true",
                enable_caching=os.environ.get("ENABLE_CACHING", str(APIConfig().enable_caching)).lower() == "true",
                max_tokens_limit=int(os.environ.get("MAX_TOKENS_LIMIT", APIConfig().max_tokens_limit)),
                rate_limit_enabled=os.environ.get("RATE_LIMIT_ENABLED", str(APIConfig().rate_limit_enabled)).lower() == "true",
                rate_limit_requests=int(os.environ.get("RATE_LIMIT_REQUESTS", APIConfig().rate_limit_requests)),
            ),
            server=ServerConfig(
                host=os.environ.get("HOST", ServerConfig().host),
                port=int(os.environ.get("PORT", ServerConfig().port)),
                workers=int(os.environ.get("WORKERS", ServerConfig().workers)),
                reload=os.environ.get("RELOAD", str(ServerConfig().reload)).lower() == "true",
                log_level=os.environ.get("LOG_LEVEL", ServerConfig().log_level),
                log_format=os.environ.get("LOG_FORMAT", ServerConfig().log_format),
            ),
            database=DatabaseConfig(
                host=os.environ.get("DB_HOST", DatabaseConfig().host),
                port=int(os.environ.get("DB_PORT", DatabaseConfig().port)),
                name=os.environ.get("DB_NAME", DatabaseConfig().name),
                username=os.environ.get("DB_USERNAME", DatabaseConfig().username),
                password=os.environ.get("DB_PASSWORD", DatabaseConfig().password),
                enabled=db_enabled,
            ),
        )


@lru_cache()
def get_config() -> Config:
    """Get cached configuration instance."""
    return Config.from_env()


def reload_config() -> Config:
    """Reload configuration (useful for testing)."""
    get_config.cache_clear()
    return get_config()
