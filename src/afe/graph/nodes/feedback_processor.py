"""Feedback Processor node: interrupts for human feedback text, parses it
into constraints that future planning must respect."""

from __future__ import annotations

from langgraph.types import interrupt

from afe.models.messages import FeedbackMessage
from afe.state.graph_state import GraphPhase, HarnessState

_STOP_WORDS = {"the", "a", "an", "please", "and", "or"}


def feedback_processor_node(state: HarnessState) -> dict:
    payload = interrupt(
        {"type": "feedback_request", "prompt": "Any feedback before finishing? (blank to stop)"}
    )
    raw_feedback = (payload or {}).get("feedback", "").strip()

    if not raw_feedback:
        return {"phase": GraphPhase.DONE}

    constraints = _parse_constraints(raw_feedback, state.dataset_schema)
    feedback_message = FeedbackMessage(
        source_agent="human",
        destination_agent="planner",
        raw_feedback=raw_feedback,
        parsed_constraints=constraints,
    )

    return {
        "human_feedback": [feedback_message],
        "active_constraints": [*state.active_constraints, *constraints],
        "phase": GraphPhase.REPLANNING,
        "replan_count": state.replan_count + 1,
    }


def _parse_constraints(feedback: str, schema) -> list[str]:
    """Very small heuristic constraint extractor. A dedicated feedback
    agent (LLM-backed) replaces this in a later phase; the interface
    (return a list[str] of constraints) will not need to change."""
    constraints = [feedback]
    if schema is not None:
        lowered = feedback.lower()
        for col in schema.columns:
            if col.name.lower() in lowered and (
                "don't" in lowered or "do not" in lowered or "avoid" in lowered
            ):
                constraints.append(f"do not modify column '{col.name}'")
    return constraints
