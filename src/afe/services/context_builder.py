"""Context engineering: builds the minimum-necessary, token-budgeted
context handed to an LLM call.

This is deliberately its own service (Single Responsibility) so context
compression/prioritization strategy can evolve independently of the
graph nodes that consume it.
"""

from __future__ import annotations

from dataclasses import dataclass

from afe.config.schemas import ContextEngineeringConfig
from afe.state.graph_state import HarnessState

# Rough heuristic: ~4 characters per token. Swappable later for a real
# tokenizer without changing the public interface.
_CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


@dataclass
class BuiltContext:
    text: str
    estimated_tokens: int
    truncated: bool


class ContextBuilder:
    """Prioritizes and budgets context sections, dropping the
    lowest-priority sections first when the budget is exceeded."""

    def __init__(self, config: ContextEngineeringConfig) -> None:
        self._config = config

    def build(self, state: HarnessState, extra_instruction: str = "") -> BuiltContext:
        budget = self._config.max_context_tokens - self._config.reserve_tokens_for_response

        sections: list[tuple[int, str, str]] = (
            []
        )  # (priority, label, content) — lower priority number = kept first
        if state.dataset_schema is not None:
            visible_cols = [c.name for c in state.dataset_schema.visible_columns()]
            sections.append((0, "schema", f"Visible columns: {', '.join(visible_cols)}"))

        if state.dataset_metadata is not None:
            m = state.dataset_metadata
            sections.append(
                (0, "metadata", f"Dataset: {m.row_count} rows, {m.column_count} cols, version {m.version}")
            )

        if state.active_constraints:
            sections.append((0, "constraints", "User constraints: " + "; ".join(state.active_constraints)))

        if state.column_profiles:
            profile_lines = []
            for name, profile in state.column_profiles.items():
                profile_lines.append(
                    f"{name}: dtype={profile.dtype.value}, nulls={profile.null_percentage}%, "
                    f"unique={profile.unique_count}, mean={profile.mean}"
                )
            sections.append((1, "profiles", "Column profiles:\n" + "\n".join(profile_lines)))

        if state.correlation_summary and state.correlation_summary.high_correlation_pairs:
            pairs = "; ".join(
                f"{a}~{b}={v:.2f}" for a, b, v in state.correlation_summary.high_correlation_pairs
            )
            sections.append((1, "correlation", f"Highly correlated pairs: {pairs}"))

        if state.conversation_summary:
            sections.append((2, "summary", f"Prior conversation summary: {state.conversation_summary}"))

        if state.execution_history:
            recent = state.execution_history[-5:]
            sections.append((2, "history", "Recent execution history: " + "; ".join(recent)))

        if extra_instruction:
            sections.append((0, "instruction", extra_instruction))

        sections.sort(key=lambda s: s[0])

        used = 0
        kept: list[str] = []
        truncated = False
        for _, _, content in sections:
            cost = estimate_tokens(content)
            if used + cost > budget:
                truncated = True
                continue
            kept.append(content)
            used += cost

        text = "\n\n".join(kept)
        return BuiltContext(text=text, estimated_tokens=used, truncated=truncated)
