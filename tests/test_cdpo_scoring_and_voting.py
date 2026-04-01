import json

from dr_agent.cdpo.context import parse_interleaved_turns
from dr_agent.cdpo.scoring import LiteLLMRubricScorer, RubricScorerConfig
from dr_agent.cdpo.types import PartialRolloutRecord
from dr_agent.cdpo.voting import (
    CriticalStepJudgeConfig,
    MultiAgentCriticalStepSelector,
)


async def _fake_vote_completion(params):
    name = params["model"]
    if name.endswith("1") or name.endswith("2"):
        content = json.dumps(
            {
                "critical_steps": [
                    {
                        "step_number": 2,
                        "is_critical": True,
                        "confidence": 0.9,
                        "rationale": "Search adaptation failed here.",
                    }
                ]
            }
        )
    else:
        content = json.dumps({"critical_steps": []})
    return {"choices": [{"message": {"content": content}}]}


async def _fake_score_completion(params):
    content = json.dumps(
        {
            "scores": [
                {
                    "rubric_title": "criterion_a",
                    "score": 3,
                    "justification": "fully covered",
                },
                {
                    "rubric_title": "criterion_b",
                    "score": 1,
                    "justification": "partially covered",
                },
            ]
        }
    )
    return {"choices": [{"message": {"content": content}}]}


def test_multi_agent_voting_majority_selects_step_two():
    selector = MultiAgentCriticalStepSelector(
        judges=[
            CriticalStepJudgeConfig(model="judge1"),
            CriticalStepJudgeConfig(model="judge2"),
            CriticalStepJudgeConfig(model="judge3"),
        ],
        completion_fn=_fake_vote_completion,
    )

    turns = parse_interleaved_turns(
        """
        <think>Broad search</think><call_tool name="pubmed_search">broad</call_tool><tool_output>0 results</tool_output>
        <think>Retry poorly</think><call_tool name="pubmed_search">broad broad</call_tool><tool_output>0 results</tool_output>
        """.strip()
    )

    result = __import__("asyncio").run(
        selector.select_steps(
            sample_id="s1",
            question="question",
            turns=turns,
            candidate_steps=[1, 2],
            rubrics=[],
        )
    )

    assert result.critical_steps == [2]
    assert result.vote_summary[0]["step_number"] == 2
    assert result.vote_summary[0]["yes_votes"] == 2


def test_rubric_scorer_computes_weighted_total_locally():
    scorer = LiteLLMRubricScorer(
        RubricScorerConfig(model="judge"),
        completion_fn=_fake_score_completion,
    )
    record = PartialRolloutRecord.from_dict(
        {
            "sample_id": "s1",
            "question": "question",
            "critical_step": 1,
            "context_type": "prevention",
            "context_messages": [],
            "all_rubrics": [
                {
                    "category": "content",
                    "title": "criterion_a",
                    "description": "a",
                    "weight": 0.5,
                },
                {
                    "category": "content",
                    "title": "criterion_b",
                    "description": "b",
                    "weight": 0.5,
                },
            ],
            "dr_tulu_results": [
                {
                    "model": "dr-tulu",
                    "model_answer": "answer",
                    "interleaved_text": "",
                    "tool_calls": [],
                }
            ],
            "openrouter_results": [],
        }
    )

    scored = __import__("asyncio").run(scorer.score_partial_rollout_record(record))
    rollout_score = scored.rollout_scores[0]

    assert rollout_score.total_score == 2.0
    assert rollout_score.max_possible_score == 3.0
