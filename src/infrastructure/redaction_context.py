"""Redaction Context Management.

This module provides context variables for passing redaction logging context
through the validation chain without violating Hexagonal Architecture.

Security Impact:
    - Enables logging of all redactions (ingesters + validators)
    - Maintains architectural integrity (domain doesn't directly import infrastructure)
    - Context is optional (works without context for testing)

Architecture:
    - Uses contextvars for thread-safe context passing
    - Domain validators can access context without direct infrastructure dependency
    - Graceful degradation when context not available (e.g., in tests)
"""

from contextvars import ContextVar
from typing import Optional, Dict, Any
from contextlib import contextmanager

from src.infrastructure.redaction_logger import RedactionLogger

# Context variable for redaction logging
_redaction_context: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
    'redaction_context',
    default=None
)


def get_redaction_context() -> Optional[Dict[str, Any]]:
    """Get current redaction context.
    
    Returns:
        Optional[Dict]: Context dictionary with logger, record_id, source_adapter, etc.
                       Returns None if no context is set (e.g., in tests)
    
    Security Impact:
        - Context is optional - domain code works without it
        - Enables logging when available, graceful degradation when not
    """
    try:
        return _redaction_context.get()
    except LookupError:
        return None


def set_redaction_context(
    logger: Optional[RedactionLogger],
    record_id: Optional[str] = None,
    source_adapter: Optional[str] = None,
    ingestion_id: Optional[str] = None
) -> None:
    """Set redaction context for current execution context.
    
    Parameters:
        logger: RedactionLogger instance or None (when redaction logging is disabled)
        record_id: Optional record identifier
        source_adapter: Optional source adapter identifier
        ingestion_id: Optional ingestion run identifier
    
    Security Impact:
        - Context is thread-local (contextvars)
        - Safe for concurrent execution
        - Context is isolated per execution context
    """
    context = {
        'logger': logger,
        'record_id': record_id,
        'source_adapter': source_adapter,
        'ingestion_id': ingestion_id,
    }
    _redaction_context.set(context)


@contextmanager
def redaction_context(
    logger: Optional[RedactionLogger],
    record_id: Optional[str] = None,
    source_adapter: Optional[str] = None,
    ingestion_id: Optional[str] = None
):
    """Context manager for redaction logging context.
    
    Parameters:
        logger: RedactionLogger instance or None (when redaction logging is disabled)
        record_id: Optional record identifier
        source_adapter: Optional source adapter identifier
        ingestion_id: Optional ingestion run identifier
    
    Yields:
        Dict: Context dictionary
    
    Example:
        ```python
        with redaction_context(logger, record_id="MRN001", source_adapter="csv"):
            patient = PatientRecord(**data)  # Validators can log redactions
        ```
    """
    token = _redaction_context.set({
        'logger': logger,
        'record_id': record_id,
        'source_adapter': source_adapter,
        'ingestion_id': ingestion_id,
    })
    try:
        yield _redaction_context.get()
    finally:
        _redaction_context.reset(token)


def log_redaction_if_context(
    field_name: str,
    original_value: Optional[str],
    rule_triggered: str,
    record_id: Optional[str] = None
) -> None:
    """Log redaction if context is available.
    
    This function is called from domain validators to log redactions
    without directly depending on infrastructure.
    
    Parameters:
        field_name: Name of the field being redacted
        original_value: Original value before redaction
        rule_triggered: Name of the redaction rule
        record_id: Optional record ID (if available, overrides context)
    
    Security Impact:
        - Logs redaction if context available
        - Gracefully degrades if no context (e.g., in tests)
        - Domain code doesn't need to know about logging infrastructure
    """
    context = get_redaction_context()
    if not context:
        return  # No context - skip logging (e.g., in tests)
    
    logger = context.get('logger')
    if not logger:
        return  # No logger in context
    
    if original_value is None or (isinstance(original_value, str) and not original_value.strip()):
        return  # Skip empty values
    
    # Use provided record_id or fall back to context
    effective_record_id = record_id or context.get('record_id')
    
    logger.log_redaction(
        field_name=field_name,
        original_value=original_value,
        rule_triggered=rule_triggered,
        record_id=effective_record_id,
        source_adapter=context.get('source_adapter')
    )

