"""Human Approval node.

Uses LangGraph's ``interrupt`` to pause execution and surface the current
plan to the CLI, resuming with the human's decision. This is the
enforcement point for the "human approval is mandatory" requirement.
"""

from __future__ import annotations

from langgraph.types import interrupt

from afe.models.messages import ApprovalDecision
from afe.state.graph_state import GraphPhase, HarnessState


def human_approval_node(state: HarnessState) -> dict:
    if state.current_plan is None:
        return {"phase": GraphPhase.CANCELLED, "errors": ["No plan available for approval"]}

    decision_payload = interrupt(
        {
            "type": "approval_request",
            "plan_id": state.current_plan.plan_id,
            "proposals": [p.model_dump() for p in state.current_plan.proposals],
            "summary": state.current_plan.summary,
        }
    )

    decision = ApprovalDecision(decision_payload.get("decision", "reject"))
    approved_names = []
    if decision == ApprovalDecision.APPROVE:
        approved_names = state.pending_feature_names
        next_phase = GraphPhase.EXECUTING
    elif decision == ApprovalDecision.MODIFY:
        approved_names = decision_payload.get("approved_feature_names", [])
        next_phase = GraphPhase.EXECUTING
    elif decision == ApprovalDecision.SKIP:
        next_phase = GraphPhase.SUMMARIZING
    elif decision == ApprovalDecision.ROLLBACK:
        next_phase = GraphPhase.DONE
    else:
        next_phase = GraphPhase.CANCELLED

    return {
        "last_approval_decision": decision,
        "approved_feature_names": approved_names,
        "phase": next_phase,
    }
