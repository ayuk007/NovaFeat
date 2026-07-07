"""Domain exception hierarchy.

All exceptions raised by the harness inherit from :class:`AFEError` so
callers (CLI, graph nodes) can catch at the right granularity instead of
catching bare ``Exception``.
"""

from __future__ import annotations


class AFEError(Exception):
    """Base class for all harness-specific errors."""


class ConfigurationError(AFEError):
    """Raised when configuration is missing, malformed, or invalid."""


class DataAccessError(AFEError):
    """Raised when the Data Access Service cannot satisfy a request."""


class DatasetTooLargeError(DataAccessError):
    """Raised when a dataset exceeds configured thresholds and requires
    explicit user approval of a sampling/engine strategy before proceeding."""


class PolicyViolationError(AFEError):
    """Raised when an action is blocked by a security/data-access policy
    (e.g. an agent attempting to access a blocked column)."""


class PIIDetectionError(AFEError):
    """Raised when the PII detection pipeline fails unexpectedly."""


class ToolNotFoundError(AFEError):
    """Raised when the Tool Registry has no matching tool and dynamic
    generation is disabled or also fails."""


class ToolValidationError(AFEError):
    """Raised when a dynamically generated tool fails static/security
    validation."""


class SandboxExecutionError(AFEError):
    """Raised when sandboxed code execution fails or is rejected."""


class ApprovalRequiredError(AFEError):
    """Raised when an action requires human approval that has not been
    granted yet; the graph should route to the approval node instead of
    treating this as a fatal error."""


class ModelProviderError(AFEError):
    """Raised when a configured model provider cannot be initialized or
    invoked."""


class StateError(AFEError):
    """Raised for invalid state transitions or corrupted graph state."""
