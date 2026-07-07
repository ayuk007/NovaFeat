"""Planner node: asks the planner-role model to propose a feature
engineering plan, grounded only in statistical context (never raw data)."""

from __future__ import annotations

import json

from afe.interfaces.model_provider import ModelProvider
from afe.models.messages import FeatureProposal, PlanMessage
from afe.services.context_builder import ContextBuilder
from afe.state.graph_state import GraphPhase, HarnessState, TokenUsageRecord

_PLANNER_SYSTEM_PROMPT = """You are a feature engineering planner.
You will be given only statistical summaries of a dataset (schema, null %,
distributions, correlations) — never raw rows. Propose a small set of
feature engineering operations as a JSON array. Each item must have:
"name" (new feature name), "operation" (one of: fill_missing, log_transform,
ratio, standard_scale, one_hot_encode, binning, date_features),
"source_columns" (list of existing column names), "rationale" (one sentence),
"risk_level" (low|medium|high), "params" (operation-specific parameters).
Respect any user constraints given. Return ONLY the JSON array, nothing else.
"""


def make_planner_node(model_provider: ModelProvider, context_builder: ContextBuilder):
    def planner_node(state: HarnessState) -> dict:
        context = context_builder.build(state, extra_instruction=_PLANNER_SYSTEM_PROMPT)
        model = model_provider.get_model("planner")

        response = model.invoke(context.text)
        content = getattr(response, "content", str(response))

        proposals = _parse_proposals(content)

        plan = PlanMessage(
            source_agent="planner",
            destination_agent="human_approval",
            proposals=proposals,
            summary=f"Proposed {len(proposals)} feature(s) based on current statistics.",
        )

        usage_metadata = getattr(response, "usage_metadata", None) or {}
        token_record = TokenUsageRecord(
            role="planner",
            input_tokens=usage_metadata.get("input_tokens", 0),
            output_tokens=usage_metadata.get("output_tokens", 0),
        )

        return {
            "current_plan": plan,
            "pending_feature_names": [p.name for p in proposals],
            "phase": GraphPhase.AWAITING_APPROVAL,
            "token_usage": [token_record],
            "execution_history": [plan.summary],
        }

    return planner_node


def _parse_proposals(content: str) -> list[FeatureProposal]:
    try:
        cleaned = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        raw = json.loads(cleaned)
    except (json.JSONDecodeError, AttributeError):
        return []

    proposals: list[FeatureProposal] = []
    for item in raw if isinstance(raw, list) else []:
        try:
            proposals.append(FeatureProposal(**item))
        except Exception:  # noqa: BLE001 - skip malformed individual proposals
            continue
    return proposals
