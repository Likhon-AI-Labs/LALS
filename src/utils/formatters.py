"""
Response Formatters Module
==========================
Utility functions for formatting responses across protocols.
"""

import re
from typing import List, Dict, Any, Optional


def clean_generated_text(text: str, protocol: str = "openai") -> str:
    """
    Clean generated text by removing prompt artifacts.
    
    Args:
        text: Generated text
        protocol: Target protocol (openai or anthropic)
    
    Returns:
        Cleaned text
    """
    # Remove common artifacts
    artifacts = [
        r"<\|im_end\|>",
        r"<\|im_start\|>",
        r"Human:",
        r"Assistant:",
        r"System:",
        r"\nHuman:",
        r"\nAssistant:",
        r"\nSystem:",
    ]
    
    for artifact in artifacts:
        text = re.sub(artifact, "", text, flags=re.IGNORECASE)
    
    return text.strip()


def count_tokens(text: str) -> int:
    """
    Estimate token count for a text.
    
    This is a rough estimate. For accurate counts, use tiktoken.
    Average token is about 4 characters in English.
    
    Args:
        text: Input text
    
    Returns:
        Estimated token count
    """
    # Simple estimation: 4 characters per token on average
    # Add extra for special characters and newlines
    text_with_spaces = text.replace("\n", " ")
    words = text_with_spaces.split()
    
    # Estimate: 0.75 tokens per word + overhead
    return max(1, int(len(words) * 0.75))


def truncate_text(text: str, max_tokens: int, protocol: str = "openai") -> str:
    """
    Truncate text to fit within token limit.
    
    Args:
        text: Input text
        max_tokens: Maximum tokens allowed
        protocol: Target protocol
    
    Returns:
        Truncated text
    """
    tokens = text.split()
    
    if len(tokens) <= max_tokens:
        return text
    
    # Truncate and add ellipsis
    truncated = " ".join(tokens[:max_tokens])
    
    if protocol == "openai":
        truncated += "..."
    else:
        truncated += "..."
    
    return truncated


def format_error_response(
    error_type: str,
    message: str,
    protocol: str = "openai",
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Format error response in protocol-specific format.
    
    Args:
        error_type: Error type (e.g., "invalid_request_error")
        message: Error message
        protocol: Target protocol
        request_id: Optional request ID
    
    Returns:
        Protocol-specific error response
    """
    error = {
        "error": {
            "type": error_type,
            "message": message
        }
    }
    
    if protocol == "openai":
        if request_id:
            error["error"]["param"] = None
            error["error"]["code"] = None
        return error
    
    elif protocol == "anthropic":
        return {
            "type": "error",
            "error": {
                "type": error_type,
                "message": message
            }
        }
    
    return error


def merge_messages(
    messages: List[Dict[str, Any]],
    system_message: Optional[str] = None,
    protocol: str = "openai"
) -> List[Dict[str, Any]]:
    """
    Merge system message with conversation messages.
    
    Args:
        messages: List of conversation messages
        system_message: Optional system message
        protocol: Target protocol
    
    Returns:
        Merged messages list
    """
    if protocol == "anthropic":
        # Anthropic: system message is separate
        return messages
    
    # OpenAI: system message in messages array
    if system_message:
        # Check if system message already exists
        has_system = any(
            msg.get("role") == "system" 
            for msg in messages
        )
        
        if not has_system:
            # Prepend system message
            messages = [
                {"role": "system", "content": system_message}
            ] + messages
    
    return messages


def extract_system_message(messages: List[Dict[str, Any]]) -> tuple:
    """
    Extract system message from messages array.
    
    Args:
        messages: List of messages
    
    Returns:
        Tuple of (system_message, filtered_messages)
    """
    system_message = None
    filtered_messages = []
    
    for msg in messages:
        if msg.get("role") == "system":
            system_message = msg.get("content")
        else:
            filtered_messages.append(msg)
    
    return system_message, filtered_messages


def standardize_stop_reason(reason: str, protocol: str = "openai") -> str:
    """
    Standardize stop reason across protocols.
    
    Args:
        reason: Original stop reason
        protocol: Target protocol
    
    Returns:
        Standardized stop reason
    """
    # Normalize reason
    reason_lower = reason.lower() if reason else "stop"
    
    # Mapping for different reasons
    reason_map = {
        "stop": {
            "openai": "stop",
            "anthropic": "end_turn"
        },
        "length": {
            "openai": "length",
            "anthropic": "max_tokens"
        },
        "tool_calls": {
            "openai": "tool_calls",
            "anthropic": "tool_use"
        }
    }
    
    # Find the normalized reason
    normalized = reason_lower.split("_")[0] if "_" in reason_lower else reason_lower
    
    for key, protocols in reason_map.items():
        if normalized in key or key in normalized:
            return protocols.get(protocol, reason)
    
    return reason


class ResponseBuilder:
    """Builder class for constructing protocol-specific responses."""
    
    def __init__(self, protocol: str = "openai"):
        self.protocol = protocol
        self._response = {}
    
    def set_id(self, prefix: str = "msg") -> "ResponseBuilder":
        """Set response ID."""
        import uuid
        self._response["id"] = f"{prefix}-{uuid.uuid4().hex[:24]}"
        return self
    
    def set_model(self, model: str) -> "ResponseBuilder":
        """Set model name."""
        self._response["model"] = model
        return self
    
    def set_content(self, content: str) -> "ResponseBuilder":
        """Set response content."""
        if self.protocol == "openai":
            self._response.setdefault("choices", [])
            if not self._response["choices"]:
                self._response["choices"].append({
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop"
                })
            else:
                self._response["choices"][0]["message"]["content"] = content
        elif self.protocol == "anthropic":
            self._response["content"] = [{"type": "text", "text": content}]
        
        return self
    
    def set_usage(
        self,
        prompt_tokens: int,
        completion_tokens: int
    ) -> "ResponseBuilder":
        """Set token usage."""
        self._response["usage"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
        return self
    
    def set_stop_reason(self, reason: str) -> "ResponseBuilder":
        """Set stop reason."""
        if self.protocol == "openai":
            if self._response.get("choices"):
                self._response["choices"][0]["finish_reason"] = reason
        else:
            self._response["stop_reason"] = reason
        return self
    
    def build(self) -> Dict[str, Any]:
        """Build final response."""
        import time
        
        # Add common fields
        if "id" not in self._response:
            self.set_id()
        
        self._response["created"] = int(time.time())
        
        if self.protocol == "openai":
            self._response["object"] = "chat.completion"
        
        return self._response
