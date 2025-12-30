"""
Streaming Utilities Module
==========================
Helper functions for Server-Sent Events (SSE) streaming.
"""

import asyncio
import json
import logging
from typing import AsyncIterator, Dict, Any, Callable

logger = logging.getLogger(__name__)


async def token_generator(
    model_generator: AsyncIterator[str],
    chunk_size: int = 1
) -> AsyncIterator[str]:
    """
    Yield tokens from model generator.
    
    Args:
        model_generator: Async iterator of tokens
        chunk_size: Number of tokens to yield at once
    
    Yields:
        Token strings
    """
    buffer = []
    
    async for token in model_generator:
        buffer.append(token)
        
        if len(buffer) >= chunk_size:
            yield "".join(buffer)
            buffer = []
    
    # Yield any remaining tokens
    if buffer:
        yield "".join(buffer)


async def batch_tokens(
    tokens: AsyncIterator[str],
    batch_size: int = 5
) -> AsyncIterator[str]:
    """
    Batch tokens together for more efficient streaming.
    
    Args:
        tokens: Async iterator of tokens
        batch_size: Number of tokens per batch
    
    Yields:
        Batched token strings
    """
    batch = []
    
    async for token in tokens:
        batch.append(token)
        
        if len(batch) >= batch_size:
            yield "".join(batch)
            batch = []
    
    # Yield remaining tokens
    if batch:
        yield "".join(batch)


def format_sse_event(
    event_type: str,
    data: Dict[str, Any],
    event_id: Optional[str] = None
) -> str:
    """
    Format data as SSE event.
    
    Args:
        event_type: Event type name
        data: Event data dictionary
        event_id: Optional event ID
    
    Returns:
        SSE-formatted string
    """
    lines = []
    
    if event_id:
        lines.append(f"id: {event_id}")
    
    lines.append(f"event: {event_type}")
    lines.append(f"data: {json.dumps(data)}")
    
    return "\n".join(lines) + "\n\n"


def parse_sse_line(line: str) -> tuple:
    """
    Parse a single SSE line.
    
    Args:
        line: Raw SSE line
    
    Returns:
        Tuple of (field_name, field_value)
    """
    if line.startswith(":"):
        # Comment line
        return None, None
    
    if ": " in line:
        # Standard field
        field, value = line.split(": ", 1)
        return field, value
    
    return line, None


async def parse_sse_stream(
    stream: AsyncIterator[str]
) -> AsyncIterator[Dict[str, Any]]:
    """
    Parse SSE stream into events.
    
    Args:
        stream: Async iterator of SSE lines
    
    Yields:
        Parsed event dictionaries
    """
    current_event = {}
    current_data = []
    
    async for line in stream:
        field, value = parse_sse_line(line)
        
        if field is None:
            # Comment line, skip
            continue
        
        if field == "event":
            # New event started, yield previous if exists
            if current_event and current_data:
                current_event["data"] = current_data
                yield current_event
                current_event = {}
                current_data = []
            current_event["type"] = value
        
        elif field == "data":
            current_data.append(value)
        
        elif field == "id":
            current_event["id"] = value
        
        elif field == "retry":
            current_event["retry"] = value
    
    # Yield final event
    if current_event and current_data:
        current_event["data"] = current_data
        yield current_event


class StreamAdapter:
    """
    Adapter for converting between streaming formats.
    
    Useful for converting model output to different SSE formats.
    """
    
    def __init__(self, protocol: str = "openai"):
        """
        Initialize stream adapter.
        
        Args:
            protocol: Target protocol (openai or anthropic)
        """
        self.protocol = protocol
    
    def adapt_token(self, token: str) -> str:
        """Adapt a single token for the target protocol."""
        # Escape special characters
        escaped = (
            token
            .replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", "\\n")
            .replace("\r", "\\r")
        )
        return escaped
    
    def format_chunk(
        self,
        token: str,
        index: int = 0,
        finish_reason: Optional[str] = None,
        usage: Optional[Dict[str, int]] = None
    ) -> Dict[str, Any]:
        """Format a chunk for the target protocol."""
        if self.protocol == "openai":
            return {
                "id": f"chatcmpl-{id(self)}",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "qwen3-0.6b",
                "choices": [{
                    "index": index,
                    "delta": {"content": token},
                    "finish_reason": finish_reason
                }]
            }
        else:  # anthropic
            return {
                "type": "content_block_delta",
                "index": index,
                "delta": {
                    "type": "text_delta",
                    "text": token
                }
            }
    
    def format_start_event(
        self,
        message_id: str,
        model: str
    ) -> Dict[str, Any]:
        """Format message start event for Anthropic."""
        if self.protocol == "anthropic":
            return {
                "type": "message_start",
                "message": {
                    "id": message_id,
                    "type": "message",
                    "role": "assistant",
                    "model": model
                }
            }
        return {}
    
    def format_stop_event(
        self,
        message_id: str,
        usage: Optional[Dict[str, int]] = None
    ) -> Dict[str, Any]:
        """Format message stop event for Anthropic."""
        if self.protocol == "anthropic":
            event = {
                "type": "message_stop"
            }
            if usage:
                event["usage"] = usage
            return event
        return {}


class StreamingController:
    """
    Controller for managing streaming responses.
    
    Provides utilities for controlling streaming behavior.
    """
    
    def __init__(
        self,
        max_tokens: int = 4096,
        timeout: Optional[float] = None,
        heartbeat_interval: Optional[float] = 15.0
    ):
        """
        Initialize streaming controller.
        
        Args:
            max_tokens: Maximum tokens to generate
            timeout: Timeout for entire stream (None for no timeout)
            heartbeat_interval: Interval for keep-alive heartbeats
        """
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.heartbeat_interval = heartbeat_interval
        self._tokens_generated = 0
        self._cancelled = False
    
    @property
    def tokens_generated(self) -> int:
        """Get number of tokens generated."""
        return self._tokens_generated
    
    @property
    def is_cancelled(self) -> bool:
        """Check if streaming was cancelled."""
        return self._cancelled
    
    def cancel(self) -> None:
        """Cancel streaming."""
        self._cancelled = True
    
    def should_continue(self) -> bool:
        """Check if streaming should continue."""
        if self._cancelled:
            return False
        if self._tokens_generated >= self.max_tokens:
            return False
        return True
    
    async def count_token(self) -> None:
        """Increment token counter."""
        self._tokens_generated += 1
    
    async def send_heartbeat(self) -> None:
        """Send a keep-alive heartbeat."""
        if self.heartbeat_interval:
            await asyncio.sleep(self.heartbeat_interval)
            # Send comment to keep connection alive
            yield ": keepalive\n\n"


async def create_streaming_pipeline(
    model_generator: AsyncIterator[str],
    protocol: str = "openai",
    batch_size: int = 5,
    max_tokens: int = 4096
) -> AsyncIterator[str]:
    """
    Create a complete streaming pipeline.
    
    Args:
        model_generator: Raw token generator from model
        protocol: Target protocol
        batch_size: Token batch size
        max_tokens: Maximum tokens to generate
    
    Yields:
        SSE-formatted strings
    """
    adapter = StreamAdapter(protocol)
    controller = StreamingController(max_tokens=max_tokens)
    
    token_count = 0
    
    async for token in batch_tokens(model_generator, batch_size):
        if not controller.should_continue():
            break
        
        await controller.count_token()
        
        # Format chunk
        chunk = adapter.format_chunk(token=token, index=0)
        yield f"data: {json.dumps(chunk)}\n\n"
    
    # Send final chunk
    final_chunk = adapter.format_chunk(
        token="",
        index=0,
        finish_reason="stop"
    )
    yield f"data: {json.dumps(final_chunk)}\n\n"
    
    # Signal end of stream
    yield "data: [DONE]\n\n"
