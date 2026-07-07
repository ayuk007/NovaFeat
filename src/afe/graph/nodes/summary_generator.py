"""Summary Generator node: produces a human-readable execution summary
and asks for feedback."""

from __future__ import annotations

from afe.state.graph_state import GraphPhase, HarnessState


def summary_generator_node(state: HarnessState) -> dict:
    if state.executed_transformations:
        latest = state.executed_transformations[-1]
        summary = (
            f"Executed {len(latest.executed_proposals)} feature(s): "
            f"{', '.join(latest.executed_proposals) or 'none'}. "
            f"Dataset moved from version {latest.dataset_version_before} "
            f"to {latest.dataset_version_after}."
        )
    else:
        summary = "No transformations were executed this round."

    return {
        "conversation_summary": summary,
        "phase": GraphPhase.AWAITING_FEEDBACK,
        "execution_history": [summary],
    }
