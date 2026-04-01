from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Sequence

import litellm

from .prompts import RUBRIC_SCORER_SYSTEM_PROMPT, build_rubric_scoring_prompt
from .types import PartialRolloutRecord, RolloutScore, RolloutScoreRecord, RubricItem
from .utils import clamp, coerce_float, extract_completion_text, strip_code_fences


CompletionFn = Callable[[dict[str, Any]], Awaitable[Any]]


@dataclass(slots=True)
class RubricScorerConfig:
    model: str
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.1
    max_tokens: int = 1600
    max_retries: int = 3
    max_concurrent_requests: int = 20


class LiteLLMRubricScorer:
    """Rubric scorer that computes weighted totals locally for stability."""

    def __init__(
        self,
        config: RubricScorerConfig,
        *,
        completion_fn: CompletionFn | None = None,
    ) -> None:
        self.config = config
        self.completion_fn = completion_fn
        self._semaphore = asyncio.Semaphore(config.max_concurrent_requests)

    async def _call_model(self, messages: list[dict[str, str]]) -> str:
        params: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if self.config.api_key:
            params["api_key"] = self.config.api_key
        if self.config.base_url:
            params["base_url"] = self.config.base_url

        async with self._semaphore:
            if self.completion_fn is not None:
                response = await self.completion_fn(params)
            else:
                response = await litellm.acompletion(**params)
        return extract_completion_text(response)

    def _parse_scores(
        self,
        response_text: str,
        rubrics: Sequence[RubricItem],
    ) -> tuple[list[dict[str, Any]], float, float]:
        response_text = strip_code_fences(response_text)
        payload = json.loads(response_text)
        items = payload.get("scores") or []

        rubric_by_title = {rubric.title: rubric for rubric in rubrics}
        weighted_scores: list[dict[str, Any]] = []
        for fallback_index, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            title = str(item.get("rubric_title", "")).strip()
            rubric = rubric_by_title.get(title)
            if rubric is None and fallback_index < len(rubrics):
                rubric = rubrics[fallback_index]
                title = rubric.title
            if rubric is None:
                continue
            score = int(clamp(coerce_float(item.get("score"), 0.0) or 0.0, 0.0, 3.0))
            weighted_scores.append(
                {
                    "rubric_title": title,
                    "score": score,
                    "justification": str(item.get("justification", "")),
                    "weight": rubric.weight,
                    "weighted_score": score * rubric.weight,
                }
            )

        max_possible = sum(rubric.weight * 3.0 for rubric in rubrics)
        total_score = sum(item["weighted_score"] for item in weighted_scores)
        return weighted_scores, total_score, max_possible

    async def score_answer(
        self,
        *,
        question: str,
        answer: str,
        rubrics: Sequence[RubricItem],
    ) -> tuple[list[dict[str, Any]], float, float]:
        messages = [
            {"role": "system", "content": RUBRIC_SCORER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_rubric_scoring_prompt(question, answer, rubrics),
            },
        ]

        last_error: Exception | None = None
        for _ in range(self.config.max_retries):
            try:
                response_text = await self._call_model(messages)
                return self._parse_scores(response_text, rubrics)
            except Exception as exc:  # noqa: BLE001 - surface robust parsing errors
                last_error = exc
                await asyncio.sleep(1.0)

        raise RuntimeError(f"Rubric scoring failed after retries: {last_error}")

    async def score_partial_rollout_record(
        self,
        record: PartialRolloutRecord,
    ) -> RolloutScoreRecord:
        rubrics = record.all_rubrics

        async def score_one(source: str, index: int, answer: str) -> RolloutScore:
            answer = answer or ""
            if not answer.strip():
                max_possible = sum(item.weight * 3.0 for item in rubrics)
                return RolloutScore(
                    source=source,
                    index=index,
                    rubric_scores=[],
                    total_score=0.0,
                    max_possible_score=max_possible,
                    error="empty_answer",
                )
            try:
                rubric_scores, total_score, max_possible = await self.score_answer(
                    question=record.question,
                    answer=answer,
                    rubrics=rubrics,
                )
                return RolloutScore(
                    source=source,
                    index=index,
                    rubric_scores=rubric_scores,
                    total_score=total_score,
                    max_possible_score=max_possible,
                )
            except Exception as exc:  # noqa: BLE001
                max_possible = sum(item.weight * 3.0 for item in rubrics)
                return RolloutScore(
                    source=source,
                    index=index,
                    rubric_scores=[],
                    total_score=0.0,
                    max_possible_score=max_possible,
                    error=str(exc),
                )

        tasks = []
        for index, rollout in enumerate(record.dr_tulu_results):
            tasks.append(score_one("dr_tulu", index, rollout.model_answer))
        for index, rollout in enumerate(record.openrouter_results):
            tasks.append(score_one("openrouter", index, rollout.model_answer))

        rollout_scores = list(await asyncio.gather(*tasks))
        return RolloutScoreRecord(
            sample_id=record.sample_id,
            critical_step=record.critical_step,
            context_type=record.context_type,
            rollout_scores=rollout_scores,
        )
