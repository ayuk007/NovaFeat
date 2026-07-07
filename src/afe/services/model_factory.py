"""Model factory: the only place ``init_chat_model`` is called.

Implements :class:`afe.interfaces.model_provider.ModelProvider`. Adding a
new role, or switching a role's provider/model, is purely a config
change in ``model.yaml`` — this class never needs to change for that.
"""

from __future__ import annotations

from afe.config.schemas import ModelConfig, ModelRoleConfig
from afe.exceptions.errors import ModelProviderError


class ModelFactory:
    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self._cache: dict[str, object] = {}

    def _role_config(self, role: str) -> ModelRoleConfig:
        if not hasattr(self._config, role):
            raise ModelProviderError(f"Unknown model role '{role}'")
        return getattr(self._config, role)

    def get_model(self, role: str):
        if role in self._cache:
            return self._cache[role]

        role_config = self._role_config(role)
        try:
            from langchain.chat_models import init_chat_model
        except ImportError as exc:  # pragma: no cover - environment issue, not logic
            raise ModelProviderError(
                "langchain.chat_models.init_chat_model is unavailable; ensure "
                "the 'langchain' package is installed"
            ) from exc

        try:
            model = init_chat_model(
                model=role_config.model,
                model_provider=role_config.provider,
                temperature=role_config.temperature,
                max_tokens=role_config.max_tokens,
                timeout=role_config.timeout_seconds,
                max_retries=role_config.max_retries,
                **role_config.extra,
            )
        except Exception as exc:  # noqa: BLE001
            raise ModelProviderError(
                f"Failed to initialize model for role '{role}' "
                f"(provider={role_config.provider}, model={role_config.model}): {exc}"
            ) from exc

        self._cache[role] = model
        return model
