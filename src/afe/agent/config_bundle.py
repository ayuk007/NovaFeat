"""Dependency container for the harness.

Rather than having graph nodes reach into global singletons, all shared
services are constructed once here and injected into node factories
(Dependency Inversion). This is also the single place that knows how to
wire config -> services, which keeps ``build_graph`` and the CLI free of
construction details (a small Facade over startup).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from afe.config.loader import get_config
from afe.config.schemas import RootConfig
from afe.interfaces.data_access import DataAccessService
from afe.interfaces.model_provider import ModelProvider
from afe.security.pii.aggregator import PIIAggregator
from afe.security.pii.regex_detector import RegexPIIDetector
from afe.services.context_builder import ContextBuilder
from afe.services.data_access_service import PandasDataAccessService
from afe.services.model_factory import ModelFactory


@dataclass
class AgentDependencies:
    config: RootConfig
    data_access: DataAccessService
    model_provider: ModelProvider
    context_builder: ContextBuilder


def build_dependencies(df: pd.DataFrame, config_dir: str = "config") -> AgentDependencies:
    config = get_config(config_dir)

    pii_aggregator = PIIAggregator(
        detectors=[RegexPIIDetector()],
        confidence_threshold=config.security.pii_confidence_threshold,
    )
    data_access = PandasDataAccessService(
        df=df,
        pii_aggregator=pii_aggregator,
        blocked_columns=config.security.blocked_columns,
        preview_row_cap=config.feature_engineering.preview_row_count,
        temp_dir=config.storage.temp_directory,
    )
    model_provider = ModelFactory(config.model)
    context_builder = ContextBuilder(config.context)

    return AgentDependencies(
        config=config,
        data_access=data_access,
        model_provider=model_provider,
        context_builder=context_builder,
    )
