"""
Request Validators Module
=========================
Validation utilities for API requests across protocols.
"""

from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, ValidationError, field_validator, model_validator


class ValidationResult(BaseModel):
    """Result of request validation."""
    is_valid: bool
    errors: List[str] = []
    sanitized_request: Optional[Dict[str, Any]] = None


def validate_messages(messages: List[Dict[str, Any]], protocol: str = "openai") -> Tuple[bool, List[str]]:
    """
    Validate messages array.
    
    Args:
        messages: Messages to validate
        protocol: Target protocol
    
    Returns:
        Tuple of (is_valid, errors)
    """
    errors = []
    
    if not isinstance(messages, list):
        errors.append("messages must be an array")
        return False, errors
    
    if len(messages) == 0:
        errors.append("messages array cannot be empty")
        return False, errors
    
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            errors.append(f"message[{i}] must be an object")
            continue
        
        # Validate role
        role = msg.get("role")
        if not role:
            errors.append(f"message[{i}] must have a role")
        elif role not in ["system", "user", "assistant"]:
            errors.append(f"message[{i}] has invalid role: {role}")
        
        # Validate content
        content = msg.get("content")
        if content is None:
            errors.append(f"message[{i}] must have content")
        elif not isinstance(content, str):
            errors.append(f"message[{i}] content must be a string")
    
    return len(errors) == 0, errors


def validate_temperature(temperature: Optional[float], protocol: str = "openai") -> Tuple[bool, List[str]]:
    """
    Validate temperature parameter.
    
    Args:
        temperature: Temperature value
        protocol: Target protocol
    
    Returns:
        Tuple of (is_valid, errors)
    """
    errors = []
    
    if temperature is None:
        return True, errors
    
    if not isinstance(temperature, (int, float)):
        errors.append("temperature must be a number")
        return False, errors
    
    if temperature < 0 or temperature > 2:
        errors.append("temperature must be between 0 and 2")
    
    return len(errors) == 0, errors


def validate_max_tokens(
    max_tokens: Optional[int],
    protocol: str = "openai",
    limit: int = 4096
) -> Tuple[bool, List[str]]:
    """
    Validate max_tokens parameter.
    
    Args:
        max_tokens: Max tokens value
        protocol: Target protocol
        limit: Maximum allowed value
    
    Returns:
        Tuple of (is_valid, errors)
    """
    errors = []
    
    if max_tokens is None:
        return True, errors
    
    if not isinstance(max_tokens, int):
        errors.append("max_tokens must be an integer")
        return False, errors
    
    if max_tokens < 1:
        errors.append("max_tokens must be positive")
    
    if max_tokens > limit:
        errors.append(f"max_tokens exceeds maximum allowed ({limit})")
    
    return len(errors) == 0, errors


def validate_top_p(top_p: Optional[float], protocol: str = "openai") -> Tuple[bool, List[str]]:
    """
    Validate top_p parameter.
    
    Args:
        top_p: Top-p value
        protocol: Target protocol
    
    Returns:
        Tuple of (is_valid, errors)
    """
    errors = []
    
    if top_p is None:
        return True, errors
    
    if not isinstance(top_p, (int, float)):
        errors.append("top_p must be a number")
        return False, errors
    
    if top_p < 0 or top_p > 1:
        errors.append("top_p must be between 0 and 1")
    
    return len(errors) == 0, errors


def validate_stop_sequences(
    stop: Optional[List[str]],
    protocol: str = "openai"
) -> Tuple[bool, List[str]]:
    """
    Validate stop sequences parameter.
    
    Args:
        stop: Stop sequences
        protocol: Target protocol
    
    Returns:
        Tuple of (is_valid, errors)
    """
    errors = []
    
    if stop is None:
        return True, errors
    
    if not isinstance(stop, list):
        errors.append("stop must be an array")
        return False, errors
    
    for i, seq in enumerate(stop):
        if not isinstance(seq, str):
            errors.append(f"stop[{i}] must be a string")
    
    if len(stop) > 4:
        errors.append("stop can have at most 4 sequences")
    
    return len(errors) == 0, errors


def sanitize_request(
    request: Dict[str, Any],
    protocol: str = "openai"
) -> Dict[str, Any]:
    """
    Sanitize request by removing unknown fields.
    
    Args:
        request: Original request
        protocol: Target protocol
    
    Returns:
        Sanitized request
    """
    # Known fields for both protocols
    common_fields = {
        "model",
        "messages",
        "temperature",
        "top_p",
        "stream",
        "stop",
        "max_tokens",
    }
    
    # Protocol-specific fields
    openai_fields = common_fields | {
        "n",
        "presence_penalty",
        "frequency_penalty",
        "logit_bias",
        "user",
    }
    
    anthropic_fields = common_fields | {
        "system",
        "top_k",
        "stop_sequences",
        "metadata",
    }
    # Legacy Anthropic fields
    anthropic_fields |= {"max_tokens_to_sample"}
    
    # Select allowed fields
    allowed_fields = openai_fields if protocol == "openai" else anthropic_fields
    
    # Filter request
    sanitized = {k: v for k, v in request.items() if k in allowed_fields}
    
    return sanitized


def validate_request(
    request: Dict[str, Any],
    protocol: str = "openai",
    max_tokens_limit: int = 4096
) -> ValidationResult:
    """
    Comprehensive request validation.
    
    Args:
        request: Request to validate
        protocol: Target protocol
        max_tokens_limit: Maximum allowed max_tokens
    
    Returns:
        ValidationResult with status and errors
    """
    errors = []
    
    # Check required fields
    if "model" not in request:
        errors.append("model is required")
    
    # Validate messages
    if "messages" in request:
        valid, msg_errors = validate_messages(request["messages"], protocol)
        if not valid:
            errors.extend(msg_errors)
    
    # Validate temperature
    if "temperature" in request:
        valid, temp_errors = validate_temperature(request["temperature"], protocol)
        if not valid:
            errors.extend(temp_errors)
    
    # Validate max_tokens
    if "max_tokens" in request:
        valid, token_errors = validate_max_tokens(
            request["max_tokens"],
            protocol,
            max_tokens_limit
        )
        if not valid:
            errors.extend(token_errors)
    
    # Validate top_p
    if "top_p" in request:
        valid, top_p_errors = validate_top_p(request["top_p"], protocol)
        if not valid:
            errors.extend(top_p_errors)
    
    # Validate stop sequences
    stop_key = "stop_sequences" if protocol == "anthropic" else "stop"
    if stop_key in request:
        valid, stop_errors = validate_stop_sequences(request[stop_key], protocol)
        if not valid:
            errors.extend(stop_errors)
    
    # Sanitize request
    sanitized = sanitize_request(request, protocol)
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        sanitized_request=sanitized if len(errors) == 0 else None
    )


class RequestValidator:
    """Class for validating API requests."""
    
    def __init__(self, protocol: str = "openai", max_tokens_limit: int = 4096):
        """
        Initialize validator.
        
        Args:
            protocol: Target protocol
            max_tokens_limit: Maximum allowed max_tokens
        """
        self.protocol = protocol
        self.max_tokens_limit = max_tokens_limit
    
    def validate(self, request: Dict[str, Any]) -> ValidationResult:
        """Validate a request."""
        return validate_request(request, self.protocol, self.max_tokens_limit)
    
    def validate_and_sanitize(self, request: Dict[str, Any]) -> ValidationResult:
        """Validate and sanitize a request."""
        result = validate_request(request, self.protocol, self.max_tokens_limit)
        
        if result.is_valid and result.sanitized_request:
            return ValidationResult(
                is_valid=True,
                errors=[],
                sanitized_request=result.sanitized_request
            )
        
        return result
