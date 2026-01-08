"""Audit infrastructure components.

This package provides infrastructure components for audit logging,
including change data capture (CDC) audit logging.
"""

from src.infrastructure.audit.change_audit_logger import ChangeAuditLogger

__all__ = ['ChangeAuditLogger']
