"""PII detection interface.

Multiple detection stages (regex, NER, LLM classification) each implement
this interface and are composed by an aggregator (Chain of Responsibility
/ Strategy), so a new detection technique can be added without touching
the security pipeline that calls it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from afe.models.pii import ColumnPIIAssessment


class PIIDetector(ABC):
    """A single stage in the PII detection pipeline."""

    @property
    @abstractmethod
    def stage_name(self) -> str:
        """Human-readable identifier used in explanations, e.g. 'regex', 'ner'."""

    @abstractmethod
    def assess(self, column_name: str, sample_values: list[str]) -> ColumnPIIAssessment:
        """Assess whether a column is likely to contain PII.

        ``sample_values`` must already be a small, bounded sample (never
        the full column) — enforcement of that bound is the caller's
        responsibility (the Data Access Service), not this detector's.
        """
