"""Dataset Analyzer node: populates schema/profiles/correlation summary
into state from the Data Access Service. Never touches raw data itself."""

from __future__ import annotations

from afe.interfaces.data_access import DataAccessService
from afe.state.graph_state import GraphPhase, HarnessState


def make_dataset_analyzer_node(data_access: DataAccessService):
    def dataset_analyzer_node(state: HarnessState) -> dict:
        metadata = data_access.get_metadata()
        schema = data_access.get_schema()
        profiles = data_access.get_all_column_profiles()
        correlation = data_access.get_correlation_summary()

        return {
            "dataset_metadata": metadata,
            "dataset_schema": schema,
            "column_profiles": profiles,
            "correlation_summary": correlation,
            "phase": GraphPhase.PLANNING,
            "execution_history": [f"Analyzed dataset version {metadata.version}"],
        }

    return dataset_analyzer_node
