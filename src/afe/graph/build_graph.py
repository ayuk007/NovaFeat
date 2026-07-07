"""Assembles the LangGraph StateGraph for the feature engineering harness.

This is a real graph with conditional routing, an interrupt-based human
approval loop, and a replanning loop — not a linear script.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph

from afe.agent.config_bundle import AgentDependencies
from afe.graph.nodes.dataset_analyzer import make_dataset_analyzer_node
from afe.graph.nodes.feature_executor import make_feature_executor_node
from afe.graph.nodes.feedback_processor import feedback_processor_node
from afe.graph.nodes.human_approval import human_approval_node
from afe.graph.nodes.planner import make_planner_node
from afe.graph.nodes.summary_generator import summary_generator_node
from afe.models.messages import ApprovalDecision
from afe.state.graph_state import GraphPhase, HarnessState


def _route_after_approval(state: HarnessState) -> str:
    if state.last_approval_decision == ApprovalDecision.APPROVE:
        return "feature_executor"
    if state.last_approval_decision == ApprovalDecision.MODIFY:
        return "feature_executor"
    if state.last_approval_decision == ApprovalDecision.SKIP:
        return "summary_generator"
    return "exit_node"


def _route_after_feedback(state: HarnessState) -> str:
    if state.phase == GraphPhase.REPLANNING and state.replan_count <= 3:
        return "planner"
    return "exit_node"


def exit_node(state: HarnessState) -> dict:
    final_phase = GraphPhase.CANCELLED if state.phase == GraphPhase.CANCELLED else GraphPhase.DONE
    return {"phase": final_phase, "execution_history": ["Session ended."]}


def build_graph(deps: AgentDependencies):
    graph = StateGraph(HarnessState)

    graph.add_node("dataset_analyzer", make_dataset_analyzer_node(deps.data_access))
    graph.add_node("planner", make_planner_node(deps.model_provider, deps.context_builder))
    graph.add_node("human_approval", human_approval_node)
    graph.add_node("feature_executor", make_feature_executor_node(deps.data_access))
    graph.add_node("summary_generator", summary_generator_node)
    graph.add_node("feedback_processor", feedback_processor_node)
    graph.add_node("exit_node", exit_node)

    graph.set_entry_point("dataset_analyzer")
    graph.add_edge("dataset_analyzer", "planner")
    graph.add_edge("planner", "human_approval")
    graph.add_conditional_edges(
        "human_approval",
        _route_after_approval,
        {
            "feature_executor": "feature_executor",
            "summary_generator": "summary_generator",
            "exit_node": "exit_node",
        },
    )
    graph.add_edge("feature_executor", "summary_generator")
    graph.add_edge("summary_generator", "feedback_processor")
    graph.add_conditional_edges(
        "feedback_processor",
        _route_after_feedback,
        {"planner": "planner", "exit_node": "exit_node"},
    )
    graph.add_edge("exit_node", END)

    checkpointer = InMemorySaver()
    serde = getattr(checkpointer, "serde", None)
    allowed = getattr(serde, "allowed_msgpack_modules", None) if serde else None
    if isinstance(allowed, list):
        for module_name in ("afe.state.graph_state", "afe.models.dataset_metadata", "afe.models.messages"):
            if module_name not in allowed:
                allowed.append(module_name)
    return graph.compile(checkpointer=checkpointer)
