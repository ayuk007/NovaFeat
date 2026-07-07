"""Model provider abstraction.

Every place in the codebase that needs an LLM depends on this Protocol,
never on a concrete LangChain chat model class directly. The concrete
implementation (:class:`afe.services.model_factory.ModelFactory`) is the
only place that calls ``init_chat_model``, which keeps provider-switching
a pure configuration change (Dependency Inversion).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ChatModelLike(Protocol):
    """The minimal surface of a LangChain chat model that the harness relies on."""

    def invoke(self, input: Any, **kwargs: Any) -> Any: ...

    async def ainvoke(self, input: Any, **kwargs: Any) -> Any: ...


class ModelProvider(Protocol):
    """Factory for obtaining a configured chat model for a given role.

    Roles are the named responsibilities from ``model.yaml`` (planner,
    code_generation, pii_detection, ...). This indirection is what makes
    the "multi-model architecture" requirement a configuration concern.
    """

    def get_model(self, role: str) -> ChatModelLike: ...
