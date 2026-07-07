from __future__ import annotations

from afe.models.messages import ApprovalDecision, FeatureProposal, PlanMessage
from afe.state.graph_state import GraphPhase, HarnessState, TokenUsageRecord


def test_harness_state_defaults():
    state = HarnessState(session_id="abc")
    assert state.phase == GraphPhase.INIT
    assert state.executed_transformations == []
    assert state.total_tokens() == 0


def test_harness_state_total_tokens():
    state = HarnessState(
        session_id="abc",
        token_usage=[
            TokenUsageRecord(role="planner", input_tokens=100, output_tokens=50),
            TokenUsageRecord(role="summarizer", input_tokens=20, output_tokens=10),
        ],
    )
    assert state.total_tokens() == 180


def test_plan_message_serializes_round_trip():
    plan = PlanMessage(
        source_agent="planner",
        destination_agent="human_approval",
        proposals=[
            FeatureProposal(
                name="income_log", operation="log_transform", source_columns=["income"], rationale="test"
            )
        ],
        summary="one feature",
    )
    dumped = plan.model_dump()
    restored = PlanMessage.model_validate(dumped)
    assert restored.proposals[0].name == "income_log"
    assert restored.id == plan.id


def test_approval_decision_enum_values():
    assert ApprovalDecision.APPROVE.value == "approve"
    assert ApprovalDecision.MODIFY.value == "modify"
