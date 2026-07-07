"""Tool abstraction for the Tool Registry.

Both prebuilt tools and dynamically generated tools implement this same
interface, which is what lets the Tool Registry treat them uniformly
(Liskov Substitution) and lets new tool categories be added without
modifying the registry (Open/Closed).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from afe.models.tool_metadata import ToolCapability, ToolResult


class Tool(ABC):
    """A single, discoverable, invocable capability."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def capability(self) -> ToolCapability:
        """Structured metadata describing what this tool does, its inputs,
        its risk level, and whether it requires human approval."""

    @abstractmethod
    def validate_params(self, params: dict[str, Any]) -> None:
        """Raise a domain error if params are invalid for this tool."""

    @abstractmethod
    def execute(self, params: dict[str, Any]) -> ToolResult:
        """Execute the tool. Implementations that run generated code must
        do so via the sandbox executor, never in-process."""
