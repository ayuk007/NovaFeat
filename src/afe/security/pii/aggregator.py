"""Aggregates votes from multiple PII detection stages into one decision.

Stages are supplied in priority order (Chain of Responsibility). Today
only the regex stage ships by default; NER and LLM-classification stages
implement the same :class:`~afe.interfaces.pii_detector.PIIDetector`
interface and can be appended here purely via configuration in a later
phase without changing this aggregator.
"""

from __future__ import annotations

from afe.interfaces.pii_detector import PIIDetector
from afe.models.pii import ColumnPIIAssessment, DetectorVote, PIIAction, PIICategory


class PIIAggregator:
    def __init__(self, detectors: list[PIIDetector], confidence_threshold: float = 0.6) -> None:
        if not detectors:
            raise ValueError("PIIAggregator requires at least one detector stage")
        self._detectors = detectors
        self._confidence_threshold = confidence_threshold

    def assess_column(self, column_name: str, sample_values: list[str]) -> ColumnPIIAssessment:
        votes: list[DetectorVote] = []
        for detector in self._detectors:
            result = detector.assess(column_name, sample_values)
            votes.append(
                DetectorVote(
                    stage_name=detector.stage_name,
                    category=result.category,
                    confidence=result.confidence,
                    explanation=result.explanation,
                )
            )

        best_vote = max(votes, key=lambda v: v.confidence, default=None)
        if best_vote is None or best_vote.confidence < self._confidence_threshold:
            category = PIICategory.NONE
            confidence = best_vote.confidence if best_vote else 0.0
            action = PIIAction.ALLOW
        else:
            category = best_vote.category
            confidence = best_vote.confidence
            action = _default_action_for(category)

        explanation = " | ".join(v.explanation for v in votes)
        return ColumnPIIAssessment(
            column_name=column_name,
            category=category,
            confidence=confidence,
            explanation=explanation,
            votes=votes,
            recommended_action=action,
        )


def _default_action_for(category: PIICategory) -> PIIAction:
    from afe.security.pii.regex_detector import _ACTION_MAP  # local import avoids a cycle at module load

    return _ACTION_MAP.get(category, PIIAction.SUMMARIZE_ONLY)
