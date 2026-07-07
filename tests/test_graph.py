from __future__ import annotations

import json
import uuid
from unittest.mock import MagicMock

import pandas as pd
import pytest
from langgraph.types import Command

from afe.agent.config_bundle import build_dependencies
from afe.graph.build_graph import build_graph
from afe.state.graph_state import GraphPhase, HarnessState


def _mocked_planner_response(proposals: list[dict]):
    response = MagicMock()
    response.content = json.dumps(proposals)
    response.usage_metadata = {"input_tokens": 100, "output_tokens": 30}
    return response


@pytest.fixture
def graph_deps(sample_df):
    numeric_df = sample_df[["age", "income"]].fillna(0)
    deps = build_dependencies(numeric_df, config_dir="config")
    fake_model = MagicMock()
    deps.model_provider.get_model = MagicMock(return_value=fake_model)
    return deps, fake_model


def _new_thread():
    session_id = str(uuid.uuid4())
    return session_id, {"configurable": {"thread_id": session_id}}


def test_graph_reaches_approval_interrupt(graph_deps):
    deps, fake_model = graph_deps
    fake_model.invoke.return_value = _mocked_planner_response(
        [
            {
                "name": "income_log",
                "operation": "log_transform",
                "source_columns": ["income"],
                "rationale": "reduce skew",
                "risk_level": "low",
                "params": {"column": "income"},
            }
        ]
    )
    graph = build_graph(deps)
    session_id, thread_config = _new_thread()
    state = HarnessState(session_id=session_id, phase=GraphPhase.INIT)

    result = graph.invoke(state, config=thread_config)

    assert result["phase"] == GraphPhase.AWAITING_APPROVAL
    assert "__interrupt__" in result
    assert result["__interrupt__"][0].value["type"] == "approval_request"


def test_graph_executes_after_approval(graph_deps):
    deps, fake_model = graph_deps
    fake_model.invoke.return_value = _mocked_planner_response(
        [
            {
                "name": "income_log",
                "operation": "log_transform",
                "source_columns": ["income"],
                "rationale": "reduce skew",
                "risk_level": "low",
                "params": {"column": "income"},
            }
        ]
    )
    graph = build_graph(deps)
    session_id, thread_config = _new_thread()
    state = HarnessState(session_id=session_id, phase=GraphPhase.INIT)

    graph.invoke(state, config=thread_config)
    result = graph.invoke(Command(resume={"decision": "approve"}), config=thread_config)

    assert result["dataset_metadata"].version == 1
    assert result["executed_transformations"][-1].executed_proposals == ["income_log"]
    assert result["phase"] == GraphPhase.AWAITING_FEEDBACK


def test_graph_rejection_skips_execution(graph_deps):
    deps, fake_model = graph_deps
    fake_model.invoke.return_value = _mocked_planner_response(
        [
            {
                "name": "income_log",
                "operation": "log_transform",
                "source_columns": ["income"],
                "rationale": "reduce skew",
                "risk_level": "low",
                "params": {"column": "income"},
            }
        ]
    )
    graph = build_graph(deps)
    session_id, thread_config = _new_thread()
    state = HarnessState(session_id=session_id, phase=GraphPhase.INIT)

    graph.invoke(state, config=thread_config)
    result = graph.invoke(Command(resume={"decision": "reject"}), config=thread_config)

    assert result["phase"] == GraphPhase.CANCELLED
    assert result["dataset_metadata"].version == 0


def test_graph_feedback_triggers_replan(graph_deps):
    deps, fake_model = graph_deps
    fake_model.invoke.return_value = _mocked_planner_response([])
    graph = build_graph(deps)
    session_id, thread_config = _new_thread()
    state = HarnessState(session_id=session_id, phase=GraphPhase.INIT)

    graph.invoke(state, config=thread_config)
    result = graph.invoke(Command(resume={"decision": "skip"}), config=thread_config)
    assert result["phase"] == GraphPhase.AWAITING_FEEDBACK

    result = graph.invoke(Command(resume={"feedback": "don't touch age"}), config=thread_config)
    assert result["phase"] == GraphPhase.AWAITING_APPROVAL  # replanned back through planner
    assert any("age" in c for c in result["active_constraints"])
