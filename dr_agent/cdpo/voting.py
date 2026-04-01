from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterable, Sequence

import litellm

from .context import parse_interleaved_turns
from .prompts import (
    CRITICAL_STEP_VOTER_SYSTEM_PROMPT,
    build_pubmed_critical_step_prompt,
)
from .types import CriticalStepSelection, CriticalStepVote, RubricItem, TrajectoryTurn
from .utils import clamp, coerce_float, extract_completion_text, strip_code_fences


CompletionFn = Callable[[dict[str, Any]], Awaitable[Any]]


@dataclass(slots=True)
class CriticalStepJudgeConfig:
    model: str
    name: str | None = None
    temperature: float = 0.0
    max_tokens: int = 1200
    api_key: str | None = None
    base_url: str | None = None
    seed: int | None = None


def propose_pubmed_candidate_steps(
    sample: dict[str, Any],
    *,
    max_candidates: int = 6,
) -> list[int]:
    """Cheap pre-filter for open-ended trajectories before multi-agent voting.

    This does not replace voting. It only reduces committee cost by nominating
    high-recall candidates based on tool failure patterns already present in the
    trajectory. PubMed selection remains vote-driven.
    """

    tool_calls = list(sample.get("tool_calls", []) or [])
    if not tool_calls:
        turns = parse_interleaved_turns(str(sample.get("interleaved_text", "")))
        return [turn.step_number for turn in turns[:max_candidates]]

    candidates: list[tuple[int, int]] = []
    previous_queries: list[str] = []

    for step_index, call in enumerate(tool_calls, start=1):
        score = 0
        query = str(call.get("query", "")).strip().lower()
        result = call.get("result", {}) or {}

        error_text = str(result.get("error", "")).lower()
        result_total = result.get("total")
        if error_text:
            score += 3
        if result_total == 0:
            score += 2
        if query and query in previous_queries:
            score += 2
        if "timeout" in error_text:
            score += 1
        if score > 0:
            candidates.append((step_index, score))
        previous_queries.append(query)

    candidates.sort(key=lambda item: (-item[1], item[0]))
    deduped = [step for step, _ in candidates[:max_candidates]]
    if deduped:
        return deduped

    turns = parse_interleaved_turns(str(sample.get("interleaved_text", "")))
    return [turn.step_number for turn in turns[:max_candidates]]


class MultiAgentCriticalStepSelector:
    """Committee-based critical step selector for open-ended trajectories."""

    def __init__(
        self,
        judges: Sequence[CriticalStepJudgeConfig],
        *,
        quorum: int | None = None,
        max_selected_steps: int = 2,
        completion_fn: CompletionFn | None = None,
        max_concurrent_requests: int = 16,
    ) -> None:
        if not judges:
            raise ValueError("At least one judge configuration is required.")
        self.judges = list(judges)
        self.quorum = quorum or (len(judges) // 2 + 1)
        self.max_selected_steps = max_selected_steps
        self.completion_fn = completion_fn
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)

    async def _call_judge(self, judge: CriticalStepJudgeConfig, messages: list[dict[str, str]]) -> str:
        params: dict[str, Any] = {
            "model": judge.model,
            "messages": messages,
            "temperature": judge.temperature,
            "max_tokens": judge.max_tokens,
        }
        if judge.api_key:
            params["api_key"] = judge.api_key
        if judge.base_url:
            params["base_url"] = judge.base_url
        if judge.seed is not None:
            params["seed"] = judge.seed

        async with self._semaphore:
            if self.completion_fn is not None:
                response = await self.completion_fn(params)
            else:
                response = await litellm.acompletion(**params)
        return extract_completion_text(response)

    def _parse_vote_payload(
        self,
        judge_name: str,
        payload_text: str,
    ) -> list[CriticalStepVote]:
        payload_text = strip_code_fences(payload_text)
        if not payload_text:
            return []

        data = json.loads(payload_text)
        raw_items = data.get("critical_steps") or data.get("steps") or []
        votes: list[CriticalStepVote] = []
        for item in raw_items:
            if isinstance(item, int):
                step_number = item
                is_critical = True
                confidence = 1.0
                rationale = ""
            elif isinstance(item, dict):
                step_number = item.get("step_number", item.get("step", item.get("index")))
                is_critical = bool(item.get("is_critical", True))
                confidence = clamp(coerce_float(item.get("confidence"), 1.0) or 1.0, 0.0, 1.0)
                rationale = str(item.get("rationale", ""))
            else:
                continue

            try:
                step_number = int(step_number)
            except (TypeError, ValueError):
                continue
            if step_number < 1:
                continue
            votes.append(
                CriticalStepVote(
                    agent_name=judge_name,
                    step_number=step_number,
                    is_critical=is_critical,
                    confidence=confidence,
                    rationale=rationale,
                )
            )
        return votes

    async def select_steps(
        self,
        *,
        sample_id: str,
        question: str,
        turns: Sequence[TrajectoryTurn],
        candidate_steps: Sequence[int] | None = None,
        rubrics: Sequence[RubricItem] | None = None,
    ) -> CriticalStepSelection:
        if not turns:
            return CriticalStepSelection(
                sample_id=sample_id,
                question=question,
                candidate_steps=[],
                critical_steps=[],
                total_steps=0,
                metadata={"status": "empty_trajectory"},
            )

        candidate_steps = list(candidate_steps or [turn.step_number for turn in turns])
        prompt = build_pubmed_critical_step_prompt(
            question,
            turns,
            candidate_steps=candidate_steps,
            max_selected_steps=self.max_selected_steps,
            rubrics=rubrics,
        )
        messages = [
            {"role": "system", "content": CRITICAL_STEP_VOTER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        async def one_judge(judge: CriticalStepJudgeConfig) -> list[CriticalStepVote]:
            raw = await self._call_judge(judge, messages)
            return self._parse_vote_payload(judge.name or judge.model, raw)

        raw_votes = await asyncio.gather(*(one_judge(judge) for judge in self.judges))
        votes = [vote for judge_votes in raw_votes for vote in judge_votes]
        summary = self._aggregate_votes(votes, candidate_steps)
        selected_steps = [
            item["step_number"]
            for item in summary
            if item["yes_votes"] >= self.quorum
        ][: self.max_selected_steps]

        if not selected_steps and summary:
            selected_steps = [summary[0]["step_number"]]

        return CriticalStepSelection(
            sample_id=sample_id,
            question=question,
            candidate_steps=candidate_steps,
            critical_steps=selected_steps,
            votes=votes,
            vote_summary=summary,
            total_steps=len(turns),
            metadata={
                "num_judges": len(self.judges),
                "quorum": self.quorum,
                "max_selected_steps": self.max_selected_steps,
                "status": "success",
            },
        )

    def _aggregate_votes(
        self,
        votes: Sequence[CriticalStepVote],
        candidate_steps: Sequence[int],
    ) -> list[dict[str, Any]]:
        by_step: dict[int, list[CriticalStepVote]] = {step: [] for step in candidate_steps}
        for vote in votes:
            if vote.step_number in by_step:
                by_step[vote.step_number].append(vote)

        summary: list[dict[str, Any]] = []
        for step in candidate_steps:
            step_votes = by_step.get(step, [])
            yes_votes = sum(1 for vote in step_votes if vote.is_critical)
            mean_conf = 0.0
            if step_votes:
                mean_conf = sum(vote.confidence for vote in step_votes) / len(step_votes)
            summary.append(
                {
                    "step_number": step,
                    "yes_votes": yes_votes,
                    "total_votes": len(step_votes),
                    "mean_confidence": mean_conf,
                    "agents": [vote.agent_name for vote in step_votes],
                    "rationales": [vote.rationale for vote in step_votes if vote.rationale],
                }
            )

        summary.sort(
            key=lambda item: (-item["yes_votes"], -item["mean_confidence"], item["step_number"])
        )
        return summary


async def select_pubmed_critical_steps(
    sample: dict[str, Any],
    selector: MultiAgentCriticalStepSelector,
    *,
    candidate_steps: Sequence[int] | None = None,
) -> CriticalStepSelection:
    turns = parse_interleaved_turns(str(sample.get("interleaved_text", "")))
    rubrics = [RubricItem.from_dict(item) for item in sample.get("all_rubrics", [])]
    return await selector.select_steps(
        sample_id=str(sample.get("sample_id", "")),
        question=str(sample.get("question", "")),
        turns=turns,
        candidate_steps=candidate_steps,
        rubrics=rubrics,
    )
