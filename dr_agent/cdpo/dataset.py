from __future__ import annotations

from typing import Any, Iterable, Sequence

from .types import (
    CDPOStepRecord,
    PartialRolloutRecord,
    RolloutScoreRecord,
    RolloutTrace,
    RubricItem,
    rollout_score_mean,
    rubric_max_score_from_rubrics,
)
from .utils import coerce_float, read_jsonl, safe_mean


def load_partial_rollout_records(path: str) -> list[PartialRolloutRecord]:
    return [PartialRolloutRecord.from_dict(item) for item in read_jsonl(path)]


def load_rollout_score_records(path: str) -> list[RolloutScoreRecord]:
    return [RolloutScoreRecord.from_dict(item) for item in read_jsonl(path)]


def _score_map(records: Sequence[RolloutScoreRecord]) -> dict[tuple[str, int, str], RolloutScoreRecord]:
    return {
        (record.sample_id, record.critical_step, record.context_type): record
        for record in records
    }


def _attach_scores(
    rollouts: Sequence[RolloutTrace],
    score_record: RolloutScoreRecord,
    *,
    source: str,
) -> list[RolloutTrace]:
    score_by_index = {
        score.index: score
        for score in score_record.rollout_scores
        if score.source == source
    }

    attached: list[RolloutTrace] = []
    for index, rollout in enumerate(rollouts):
        copied = RolloutTrace.from_dict(rollout.to_dict(), source=rollout.source, index=rollout.index)
        score = score_by_index.get(index)
        if score is not None:
            copied.total_score = score.total_score
            copied.max_possible_score = score.max_possible_score
            copied.rubric_scores = score.rubric_scores
            if score.error:
                copied.metadata["score_error"] = score.error
        attached.append(copied)
    return attached


def _resolve_rubric_max_score(
    rollouts: Sequence[RolloutTrace],
    rubrics: Sequence[RubricItem],
) -> float:
    explicit = [
        rollout.max_possible_score
        for rollout in rollouts
        if rollout.max_possible_score is not None
    ]
    if explicit:
        return safe_mean(explicit)
    return rubric_max_score_from_rubrics(list(rubrics))


def _valid_scored_rollouts(rollouts: Sequence[RolloutTrace]) -> list[RolloutTrace]:
    return [rollout for rollout in rollouts if rollout.total_score is not None]


def compute_step_weight_normalizer(
    records: Sequence[CDPOStepRecord],
    *,
    alpha: float = 1.0,
    normalization_pool: str = "all",
) -> float:
    pool = list(records)
    if normalization_pool == "verified":
        pool = [record for record in pool if record.verified_critical]
    values = [abs(record.criticality_score) ** alpha for record in pool]
    if not values:
        return 1.0
    mean_value = safe_mean(values)
    return mean_value if mean_value > 0 else 1.0


def build_cdpo_step_records(
    partial_records: Sequence[PartialRolloutRecord] | str,
    score_records: Sequence[RolloutScoreRecord] | str,
    *,
    expert_source: str = "openrouter",
    local_source: str = "dr_tulu",
    alpha: float = 1.0,
    epsilon: float = 0.3,
    epsilon_mode: str = "raw",
    normalization_pool: str = "all",
    verified_only: bool = True,
) -> list[CDPOStepRecord]:
    """Merge rollouts and rubric scores into grouped step-level CDPO records."""

    if isinstance(partial_records, str):
        partial_records = load_partial_rollout_records(partial_records)
    if isinstance(score_records, str):
        score_records = load_rollout_score_records(score_records)

    score_lookup = _score_map(score_records)
    step_records: list[CDPOStepRecord] = []

    for partial in partial_records:
        key = (partial.sample_id, partial.critical_step, partial.context_type)
        score_record = score_lookup.get(key)
        if score_record is None:
            continue

        expert_rollouts = _attach_scores(partial.get_rollouts(expert_source), score_record, source=expert_source)
        local_rollouts = _attach_scores(partial.get_rollouts(local_source), score_record, source=local_source)
        expert_rollouts = _valid_scored_rollouts(expert_rollouts)
        local_rollouts = _valid_scored_rollouts(local_rollouts)

        if not expert_rollouts or not local_rollouts:
            continue

        all_rollouts = [*expert_rollouts, *local_rollouts]
        rubric_max_score = _resolve_rubric_max_score(all_rollouts, partial.all_rubrics)
        if rubric_max_score <= 0:
            continue

        mean_expert = rollout_score_mean(expert_rollouts)
        mean_local = rollout_score_mean(local_rollouts)
        criticality_score = mean_expert - mean_local
        criticality_ratio = criticality_score / rubric_max_score
        verified = (
            criticality_score > epsilon
            if epsilon_mode == "raw"
            else criticality_ratio > epsilon
        )

        record = CDPOStepRecord(
            sample_id=partial.sample_id,
            question=partial.question,
            topic=partial.topic,
            question_type=partial.question_type,
            critical_step=partial.critical_step,
            context_type=partial.context_type,
            context_messages=partial.context_messages,
            all_rubrics=partial.all_rubrics,
            expert_source=expert_source,
            local_source=local_source,
            expert_rollouts=expert_rollouts,
            local_rollouts=local_rollouts,
            rubric_max_score=rubric_max_score,
            mean_expert_score=mean_expert,
            mean_local_score=mean_local,
            criticality_score=criticality_score,
            criticality_ratio=criticality_ratio,
            verified_critical=verified,
            alpha=alpha,
            epsilon=epsilon,
            pair_count=len(expert_rollouts) * len(local_rollouts),
            metadata={
                "score_record_found": True,
                "epsilon_mode": epsilon_mode,
                "expert_rollout_count": len(expert_rollouts),
                "local_rollout_count": len(local_rollouts),
            },
        )
        step_records.append(record)

    z_hat = compute_step_weight_normalizer(
        step_records,
        alpha=alpha,
        normalization_pool=normalization_pool,
    )
    for record in step_records:
        record.z_hat = z_hat
        record.step_weight = (abs(record.criticality_score) ** alpha) / z_hat if z_hat > 0 else 1.0

    if verified_only:
        step_records = [record for record in step_records if record.verified_critical]
    return step_records


def flatten_step_record_to_pair_records(
    record: CDPOStepRecord,
    *,
    text_field: str = "interleaved_text",
) -> list[dict[str, Any]]:
    """Flatten a grouped step record for interoperability with pairwise trainers."""

    pairs: list[dict[str, Any]] = []
    for expert_index, expert_rollout in enumerate(record.expert_rollouts):
        for local_index, local_rollout in enumerate(record.local_rollouts):
            if expert_rollout.total_score is None or local_rollout.total_score is None:
                continue
            pair_weight = (
                (expert_rollout.total_score - local_rollout.total_score)
                / record.rubric_max_score
            )
            pairs.append(
                {
                    "sample_id": record.sample_id,
                    "question": record.question,
                    "topic": record.topic,
                    "question_type": record.question_type,
                    "critical_step": record.critical_step,
                    "context_type": record.context_type,
                    "context_messages": record.context_messages,
                    "expert_source": record.expert_source,
                    "local_source": record.local_source,
                    "expert_index": expert_index,
                    "local_index": local_index,
                    "chosen_text": expert_rollout.get_text(text_field=text_field),
                    "rejected_text": local_rollout.get_text(text_field=text_field),
                    "chosen_score": expert_rollout.total_score,
                    "rejected_score": local_rollout.total_score,
                    "pair_weight": pair_weight,
                    "step_weight": record.step_weight,
                    "criticality_score": record.criticality_score,
                    "criticality_ratio": record.criticality_ratio,
                    "rubric_max_score": record.rubric_max_score,
                }
            )
    return pairs


def summarize_step_records(records: Sequence[CDPOStepRecord]) -> dict[str, Any]:
    if not records:
        return {
            "num_records": 0,
            "verified_fraction": 0.0,
            "avg_pairs_per_record": 0.0,
            "avg_criticality_score": 0.0,
            "avg_step_weight": 0.0,
        }

    return {
        "num_records": len(records),
        "verified_fraction": safe_mean([1.0 if record.verified_critical else 0.0 for record in records]),
        "avg_pairs_per_record": safe_mean([float(record.pair_count) for record in records]),
        "avg_criticality_score": safe_mean([record.criticality_score for record in records]),
        "avg_criticality_ratio": safe_mean([record.criticality_ratio for record in records]),
        "avg_step_weight": safe_mean([record.step_weight for record in records]),
        "z_hat": records[0].z_hat,
    }
