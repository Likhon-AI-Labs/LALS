"""
Database Utilities Module
=========================
Handles database connections and operations for LALS Multi-API Gateway.
Supports PostgreSQL database with connection pooling.
"""

import logging
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
from datetime import datetime
from functools import lru_cache

import asyncpg
import aiohttp

from .config import get_config, DatabaseConfig

logger = logging.getLogger(__name__)

# Global async pool
_pool: Optional[asyncpg.Pool] = None


class DatabaseManager:
    """Manages database connections and operations."""
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        """
        Initialize database manager.
        
        Args:
            config: Database configuration. If None, loads from settings.
        """
        self.config = config or get_config().database
        self._pool: Optional[asyncpg.Pool] = None
    
    async def connect(self) -> None:
        """Establish connection pool to database."""
        if self._pool is not None:
            logger.warning("Database pool already connected")
            return
        
        if not self.config.is_configured:
            logger.info("Database not configured, skipping connection")
            return
        
        logger.info(f"Connecting to database: {self.config.host}:{self.config.port}/{self.config.name}")
        
        try:
            self._pool = await asyncpg.create_pool(
                host=self.config.host,
                port=self.config.port,
                database=self.config.name,
                user=self.config.username,
                password=self.config.password,
                min_size=2,
                max_size=10,
                command_timeout=60,
            )
            
            # Test connection
            async with self._pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            
            logger.info("Database connection established successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            self._pool = None
            raise
    
    async def disconnect(self) -> None:
        """Close database connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            logger.info("Database connection pool closed")
    
    @property
    def is_connected(self) -> bool:
        """Check if database is connected."""
        return self._pool is not None
    
    async def execute(self, query: str, *args) -> Any:
        """Execute a query and return result."""
        if not self.is_connected:
            raise RuntimeError("Database not connected")
        
        async with self._pool.acquire() as conn:
            return await conn.fetchval(query, *args)
    
    async def execute_many(self, query: str, *args_list) -> None:
        """Execute a query with multiple argument sets."""
        if not self.is_connected:
            raise RuntimeError("Database not connected")
        
        async with self._pool.acquire() as conn:
            await conn.executemany(query, *args_list)
    
    async def fetch_row(self, query: str, *args) -> Optional[Dict[str, Any]]:
        """Fetch a single row as dictionary."""
        if not self.is_connected:
            raise RuntimeError("Database not connected")
        
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(query, *args)
            return dict(row) if row else None
    
    async def fetch_all(self, query: str, *args) -> List[Dict[str, Any]]:
        """Fetch all rows as list of dictionaries."""
        if not self.is_connected:
            raise RuntimeError("Database not connected")
        
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *args)
            return [dict(row) for row in rows]
    
    async def initialize_schema(self) -> None:
        """Initialize database schema if not exists."""
        if not self.is_connected:
            logger.warning("Cannot initialize schema: database not connected")
            return
        
        logger.info("Initializing database schema...")
        
        # Create conversations table
        await self.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id VARCHAR(64) PRIMARY KEY,
                model VARCHAR(128) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSONB DEFAULT '{}'
            )
        """)
        
        # Create messages table
        await self.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id VARCHAR(64) PRIMARY KEY,
                conversation_id VARCHAR(64) REFERENCES conversations(id) ON DELETE CASCADE,
                role VARCHAR(32) NOT NULL,
                content TEXT NOT NULL,
                token_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create usage logs table
        await self.execute("""
            CREATE TABLE IF NOT EXISTS usage_logs (
                id SERIAL PRIMARY KEY,
                endpoint VARCHAR(128) NOT NULL,
                model VARCHAR(128),
                prompt_tokens INTEGER DEFAULT 0,
                completion_tokens INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                latency_ms INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes
        await self.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_conversation 
            ON messages(conversation_id)
        """)
        
        await self.execute("""
            CREATE INDEX IF NOT EXISTS idx_usage_logs_created 
            ON usage_logs(created_at)
        """)
        
        logger.info("Database schema initialized successfully")
    
    async def log_usage(
        self,
        endpoint: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: int
    ) -> None:
        """Log API usage to database."""
        if not self.is_connected:
            return
        
        try:
            await self.execute("""
                INSERT INTO usage_logs 
                (endpoint, model, prompt_tokens, completion_tokens, total_tokens, latency_ms)
                VALUES ($1, $2, $3, $4, $5, $6)
            """, endpoint, model, prompt_tokens, completion_tokens, 
                prompt_tokens + completion_tokens, latency_ms)
        except Exception as e:
            logger.error(f"Failed to log usage: {e}")


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """Get the global database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


async def initialize_database() -> DatabaseManager:
    """Initialize database connection and schema."""
    manager = get_db_manager()
    await manager.connect()
    await manager.initialize_schema()
    return manager


async def shutdown_database() -> None:
    """Shutdown database connections."""
    global _db_manager
    if _db_manager is not None:
        await _db_manager.disconnect()
        _db_manager = None


@contextmanager
def db_context():
    """Context manager for database operations."""
    manager = get_db_manager()
    try:
        yield manager
    except Exception as e:
        logger.error(f"Database operation failed: {e}")
        raise
