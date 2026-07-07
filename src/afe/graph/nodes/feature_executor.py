"""Feature Executor node: applies approved proposals via the Data Access
Service's registered transformations."""

from __future__ import annotations

from afe.interfaces.data_access import DataAccessService
from afe.models.messages import ExecutionMessage
from afe.state.graph_state import GraphPhase, HarnessState


def make_feature_executor_node(data_access: DataAccessService):
    def feature_executor_node(state: HarnessState) -> dict:
        if state.current_plan is None:
            return {"phase": GraphPhase.SUMMARIZING, "errors": ["No plan to execute"]}

        version_before = data_access.get_metadata().version
        executed: list[str] = []
        log: list[str] = []

        approved = set(state.approved_feature_names)
        for proposal in state.current_plan.proposals:
            if proposal.name not in approved:
                continue
            try:
                data_access.apply_transformation(proposal.operation, {**proposal.params})
                executed.append(proposal.name)
                log.append(f"Executed '{proposal.name}' via {proposal.operation}")
            except Exception as exc:  # noqa: BLE001 - captured for the execution log, not fatal
                log.append(f"Failed '{proposal.name}': {exc}")

        version_after = data_access.get_metadata().version
        execution_message = ExecutionMessage(
            source_agent="feature_executor",
            destination_agent="summary_generator",
            plan_id=state.current_plan.plan_id,
            executed_proposals=executed,
            dataset_version_before=version_before,
            dataset_version_after=version_after,
            execution_log=log,
        )

        return {
            "dataset_metadata": data_access.get_metadata(),
            "dataset_schema": data_access.get_schema(),
            "column_profiles": data_access.get_all_column_profiles(),
            "executed_transformations": [execution_message],
            "execution_history": log,
            "phase": GraphPhase.SUMMARIZING,
        }

    return feature_executor_node
