from dr_agent.cdpo.dataset import (
    build_cdpo_step_records,
    flatten_step_record_to_pair_records,
    summarize_step_records,
)
from dr_agent.cdpo.types import PartialRolloutRecord, RolloutScoreRecord


def _partial_record(sample_id: str, critical_step: int, question: str) -> dict:
    return {
        "sample_id": sample_id,
        "question": question,
        "topic": "pubmed",
        "question_type": "review",
        "critical_step": critical_step,
        "context_type": "prevention",
        "context_messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": question},
        ],
        "all_rubrics": [
            {
                "category": "content",
                "title": "criterion_a",
                "description": "criterion a",
                "weight": 0.5,
            },
            {
                "category": "content",
                "title": "criterion_b",
                "description": "criterion b",
                "weight": 0.5,
            },
        ],
        "dr_tulu_results": [
            {
                "model": "dr-tulu",
                "model_answer": "local answer 1",
                "interleaved_text": "local text 1",
                "tool_calls": [],
            },
            {
                "model": "dr-tulu",
                "model_answer": "local answer 2",
                "interleaved_text": "local text 2",
                "tool_calls": [],
            },
        ],
        "openrouter_results": [
            {
                "model": "openrouter",
                "model_answer": "expert answer 1",
                "interleaved_text": "expert text 1",
                "tool_calls": [],
            },
            {
                "model": "openrouter",
                "model_answer": "expert answer 2",
                "interleaved_text": "expert text 2",
                "tool_calls": [],
            },
        ],
        "total_original_steps": 3,
    }


def _score_record(sample_id: str, critical_step: int, local_scores, expert_scores) -> dict:
    rollout_scores = []
    for index, score in enumerate(local_scores):
        rollout_scores.append(
            {
                "source": "dr_tulu",
                "index": index,
                "scores": [],
                "total_score": score,
                "max_possible_score": 3.0,
            }
        )
    for index, score in enumerate(expert_scores):
        rollout_scores.append(
            {
                "source": "openrouter",
                "index": index,
                "scores": [],
                "total_score": score,
                "max_possible_score": 3.0,
            }
        )
    return {
        "sample_id": sample_id,
        "critical_step": critical_step,
        "context_type": "prevention",
        "rollout_scores": rollout_scores,
    }


def test_build_cdpo_step_records_and_flatten_pairs():
    partial_records = [
        PartialRolloutRecord.from_dict(_partial_record("s1", 1, "q1")),
        PartialRolloutRecord.from_dict(_partial_record("s2", 2, "q2")),
    ]
    score_records = [
        RolloutScoreRecord.from_dict(_score_record("s1", 1, local_scores=[0.8, 1.0], expert_scores=[2.4, 2.2])),
        RolloutScoreRecord.from_dict(_score_record("s2", 2, local_scores=[1.7, 1.6], expert_scores=[2.0, 1.9])),
    ]

    records = build_cdpo_step_records(
        partial_records,
        score_records,
        alpha=1.0,
        epsilon=0.3,
        epsilon_mode="raw",
        verified_only=False,
    )

    assert len(records) == 2
    assert records[0].criticality_score > records[1].criticality_score
    assert records[0].verified_critical is True
    assert records[1].verified_critical is True
    assert records[0].step_weight > records[1].step_weight
    assert records[0].pair_count == 4

    flattened = flatten_step_record_to_pair_records(records[0])
    assert len(flattened) == 4
    assert flattened[0]["pair_weight"] > 0
    assert flattened[0]["step_weight"] == records[0].step_weight

    summary = summarize_step_records(records)
    assert summary["num_records"] == 2
    assert summary["avg_pairs_per_record"] == 4.0
