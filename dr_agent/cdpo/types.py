from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .utils import coerce_float, safe_mean


JsonDict = dict[str, Any]


@dataclass(slots=True)
class RubricItem:
    category: str
    title: str
    description: str
    weight: float = 1.0

    @classmethod
    def from_dict(cls, data: JsonDict) -> "RubricItem":
        return cls(
            category=str(data.get("category", "")),
            title=str(data.get("title", "")),
            description=str(data.get("description", "")),
            weight=coerce_float(data.get("weight"), 1.0) or 1.0,
        )

    def to_dict(self) -> JsonDict:
        return {
            "category": self.category,
            "title": self.title,
            "description": self.description,
            "weight": self.weight,
        }


@dataclass(slots=True)
class TrajectoryTurn:
    step_number: int
    assistant_content: str
    tool_output: str = ""
    think_text: str = ""
    tool_name: str | None = None
    tool_query: str | None = None
    tool_parameters: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return {
            "step_number": self.step_number,
            "assistant_content": self.assistant_content,
            "tool_output": self.tool_output,
            "think_text": self.think_text,
            "tool_name": self.tool_name,
            "tool_query": self.tool_query,
            "tool_parameters": self.tool_parameters,
        }


@dataclass(slots=True)
class RolloutTrace:
    model: str
    model_answer: str = ""
    interleaved_text: str = ""
    tool_calls: list[JsonDict] = field(default_factory=list)
    total_score: float | None = None
    max_possible_score: float | None = None
    rubric_scores: list[JsonDict] = field(default_factory=list)
    source: str | None = None
    index: int | None = None
    metadata: JsonDict = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls,
        data: JsonDict,
        *,
        source: str | None = None,
        index: int | None = None,
    ) -> "RolloutTrace":
        return cls(
            model=str(data.get("model", "")),
            model_answer=str(data.get("model_answer", "")),
            interleaved_text=str(data.get("interleaved_text", "")),
            tool_calls=list(data.get("tool_calls", []) or []),
            total_score=coerce_float(data.get("total_score")),
            max_possible_score=coerce_float(data.get("max_possible_score")),
            rubric_scores=list(data.get("rubric_scores", data.get("scores", [])) or []),
            source=source or data.get("source"),
            index=index if index is not None else data.get("index"),
            metadata=dict(data.get("metadata", {}) or {}),
        )

    def get_text(self, text_field: str = "interleaved_text") -> str:
        if text_field == "interleaved_text" and self.interleaved_text:
            return self.interleaved_text
        if text_field == "model_answer" and self.model_answer:
            return self.model_answer
        return self.interleaved_text or self.model_answer

    def to_dict(self) -> JsonDict:
        output = {
            "model": self.model,
            "model_answer": self.model_answer,
            "interleaved_text": self.interleaved_text,
            "tool_calls": self.tool_calls,
        }
        if self.total_score is not None:
            output["total_score"] = self.total_score
        if self.max_possible_score is not None:
            output["max_possible_score"] = self.max_possible_score
        if self.rubric_scores:
            output["rubric_scores"] = self.rubric_scores
        if self.source is not None:
            output["source"] = self.source
        if self.index is not None:
            output["index"] = self.index
        if self.metadata:
            output["metadata"] = self.metadata
        return output


@dataclass(slots=True)
class PartialRolloutRecord:
    sample_id: str
    question: str
    topic: str = ""
    question_type: str = ""
    critical_step: int = 1
    context_type: str = "prevention"
    context_messages: list[JsonDict] = field(default_factory=list)
    dr_tulu_results: list[RolloutTrace] = field(default_factory=list)
    openrouter_results: list[RolloutTrace] = field(default_factory=list)
    total_original_steps: int = 0
    processed_time: str | None = None
    tool_rubrics: list[RubricItem] = field(default_factory=list)
    content_rubrics: list[RubricItem] = field(default_factory=list)
    all_rubrics: list[RubricItem] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: JsonDict) -> "PartialRolloutRecord":
        tool_rubrics = [RubricItem.from_dict(item) for item in data.get("tool_rubrics", [])]
        content_rubrics = [RubricItem.from_dict(item) for item in data.get("content_rubrics", [])]
        all_rubrics = [RubricItem.from_dict(item) for item in data.get("all_rubrics", [])]
        if not all_rubrics:
            all_rubrics = [*tool_rubrics, *content_rubrics]

        return cls(
            sample_id=str(data.get("sample_id", "")),
            question=str(data.get("question", "")),
            topic=str(data.get("topic", "")),
            question_type=str(data.get("question_type", "")),
            critical_step=int(data.get("critical_step", 1)),
            context_type=str(data.get("context_type", "prevention")),
            context_messages=list(data.get("context_messages", []) or []),
            dr_tulu_results=[
                RolloutTrace.from_dict(item, source="dr_tulu", index=i)
                for i, item in enumerate(data.get("dr_tulu_results", []) or [])
            ],
            openrouter_results=[
                RolloutTrace.from_dict(item, source="openrouter", index=i)
                for i, item in enumerate(data.get("openrouter_results", []) or [])
            ],
            total_original_steps=int(data.get("total_original_steps", 0) or 0),
            processed_time=data.get("processed_time"),
            tool_rubrics=tool_rubrics,
            content_rubrics=content_rubrics,
            all_rubrics=all_rubrics,
        )

    def get_rollouts(self, source: str) -> list[RolloutTrace]:
        normalized = source.lower().replace("-", "_")
        if normalized in {"dr_tulu", "local", "policy"}:
            return self.dr_tulu_results
        if normalized in {"openrouter", "expert"}:
            return self.openrouter_results
        raise KeyError(f"Unknown rollout source: {source}")

    def to_dict(self) -> JsonDict:
        return {
            "sample_id": self.sample_id,
            "question": self.question,
            "topic": self.topic,
            "question_type": self.question_type,
            "critical_step": self.critical_step,
            "context_type": self.context_type,
            "context_messages": self.context_messages,
            "dr_tulu_results": [item.to_dict() for item in self.dr_tulu_results],
            "openrouter_results": [item.to_dict() for item in self.openrouter_results],
            "total_original_steps": self.total_original_steps,
            "processed_time": self.processed_time,
            "tool_rubrics": [item.to_dict() for item in self.tool_rubrics],
            "content_rubrics": [item.to_dict() for item in self.content_rubrics],
            "all_rubrics": [item.to_dict() for item in self.all_rubrics],
        }


@dataclass(slots=True)
class RolloutScore:
    source: str
    index: int
    rubric_scores: list[JsonDict] = field(default_factory=list)
    total_score: float | None = None
    max_possible_score: float | None = None
    error: str | None = None

    @classmethod
    def from_dict(cls, data: JsonDict) -> "RolloutScore":
        return cls(
            source=str(data.get("source", "")),
            index=int(data.get("index", 0)),
            rubric_scores=list(data.get("rubric_scores", data.get("scores", [])) or []),
            total_score=coerce_float(data.get("total_score")),
            max_possible_score=coerce_float(data.get("max_possible_score")),
            error=data.get("error"),
        )

    def to_dict(self) -> JsonDict:
        output = {
            "source": self.source,
            "index": self.index,
            "scores": self.rubric_scores,
            "total_score": self.total_score,
            "max_possible_score": self.max_possible_score,
        }
        if self.error:
            output["error"] = self.error
        return output


@dataclass(slots=True)
class RolloutScoreRecord:
    sample_id: str
    critical_step: int
    context_type: str
    rollout_scores: list[RolloutScore] = field(default_factory=list)
    scored_time: str | None = None

    @classmethod
    def from_dict(cls, data: JsonDict) -> "RolloutScoreRecord":
        return cls(
            sample_id=str(data.get("sample_id", "")),
            critical_step=int(data.get("critical_step", 1)),
            context_type=str(data.get("context_type", "prevention")),
            rollout_scores=[
                RolloutScore.from_dict(item)
                for item in data.get("rollout_scores", []) or []
            ],
            scored_time=data.get("scored_time"),
        )

    def to_dict(self) -> JsonDict:
        return {
            "sample_id": self.sample_id,
            "critical_step": self.critical_step,
            "context_type": self.context_type,
            "rollout_scores": [item.to_dict() for item in self.rollout_scores],
            "scored_time": self.scored_time,
        }


@dataclass(slots=True)
class CriticalStepVote:
    agent_name: str
    step_number: int
    is_critical: bool
    confidence: float
    rationale: str = ""

    def to_dict(self) -> JsonDict:
        return {
            "agent_name": self.agent_name,
            "step_number": self.step_number,
            "is_critical": self.is_critical,
            "confidence": self.confidence,
            "rationale": self.rationale,
        }


@dataclass(slots=True)
class CriticalStepSelection:
    sample_id: str
    question: str
    candidate_steps: list[int]
    critical_steps: list[int]
    votes: list[CriticalStepVote] = field(default_factory=list)
    vote_summary: list[JsonDict] = field(default_factory=list)
    total_steps: int = 0
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return {
            "sample_id": self.sample_id,
            "question": self.question,
            "candidate_steps": self.candidate_steps,
            "critical_steps": self.critical_steps,
            "votes": [vote.to_dict() for vote in self.votes],
            "vote_summary": self.vote_summary,
            "total_steps": self.total_steps,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class CDPOStepRecord:
    sample_id: str
    question: str
    topic: str
    question_type: str
    critical_step: int
    context_type: str
    context_messages: list[JsonDict]
    all_rubrics: list[RubricItem]
    expert_source: str
    local_source: str
    expert_rollouts: list[RolloutTrace]
    local_rollouts: list[RolloutTrace]
    rubric_max_score: float
    mean_expert_score: float
    mean_local_score: float
    criticality_score: float
    criticality_ratio: float
    verified_critical: bool
    step_weight: float = 1.0
    z_hat: float | None = None
    alpha: float = 1.0
    epsilon: float = 0.3
    pair_count: int = 0
    metadata: JsonDict = field(default_factory=dict)

    @property
    def all_scores(self) -> list[float]:
        return [
            *[item.total_score for item in self.expert_rollouts if item.total_score is not None],
            *[item.total_score for item in self.local_rollouts if item.total_score is not None],
        ]

    def to_dict(self) -> JsonDict:
        return {
            "sample_id": self.sample_id,
            "question": self.question,
            "topic": self.topic,
            "question_type": self.question_type,
            "critical_step": self.critical_step,
            "context_type": self.context_type,
            "context_messages": self.context_messages,
            "all_rubrics": [item.to_dict() for item in self.all_rubrics],
            "expert_source": self.expert_source,
            "local_source": self.local_source,
            "expert_rollouts": [item.to_dict() for item in self.expert_rollouts],
            "local_rollouts": [item.to_dict() for item in self.local_rollouts],
            "rubric_max_score": self.rubric_max_score,
            "mean_expert_score": self.mean_expert_score,
            "mean_local_score": self.mean_local_score,
            "criticality_score": self.criticality_score,
            "criticality_ratio": self.criticality_ratio,
            "verified_critical": self.verified_critical,
            "step_weight": self.step_weight,
            "z_hat": self.z_hat,
            "alpha": self.alpha,
            "epsilon": self.epsilon,
            "pair_count": self.pair_count,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: JsonDict) -> "CDPOStepRecord":
        return cls(
            sample_id=str(data.get("sample_id", "")),
            question=str(data.get("question", "")),
            topic=str(data.get("topic", "")),
            question_type=str(data.get("question_type", "")),
            critical_step=int(data.get("critical_step", 1)),
            context_type=str(data.get("context_type", "prevention")),
            context_messages=list(data.get("context_messages", []) or []),
            all_rubrics=[RubricItem.from_dict(item) for item in data.get("all_rubrics", [])],
            expert_source=str(data.get("expert_source", "openrouter")),
            local_source=str(data.get("local_source", "dr_tulu")),
            expert_rollouts=[
                RolloutTrace.from_dict(item, source=data.get("expert_source", "openrouter"))
                for item in data.get("expert_rollouts", []) or []
            ],
            local_rollouts=[
                RolloutTrace.from_dict(item, source=data.get("local_source", "dr_tulu"))
                for item in data.get("local_rollouts", []) or []
            ],
            rubric_max_score=coerce_float(data.get("rubric_max_score"), 1.0) or 1.0,
            mean_expert_score=coerce_float(data.get("mean_expert_score"), 0.0) or 0.0,
            mean_local_score=coerce_float(data.get("mean_local_score"), 0.0) or 0.0,
            criticality_score=coerce_float(data.get("criticality_score"), 0.0) or 0.0,
            criticality_ratio=coerce_float(data.get("criticality_ratio"), 0.0) or 0.0,
            verified_critical=bool(data.get("verified_critical", False)),
            step_weight=coerce_float(data.get("step_weight"), 1.0) or 1.0,
            z_hat=coerce_float(data.get("z_hat")),
            alpha=coerce_float(data.get("alpha"), 1.0) or 1.0,
            epsilon=coerce_float(data.get("epsilon"), 0.3) or 0.3,
            pair_count=int(data.get("pair_count", 0) or 0),
            metadata=dict(data.get("metadata", {}) or {}),
        )


def rubric_max_score_from_rubrics(rubrics: list[RubricItem]) -> float:
    if not rubrics:
        return 0.0
    return sum(item.weight for item in rubrics) * 3.0


def rollout_score_mean(rollouts: list[RolloutTrace]) -> float:
    return safe_mean([item.total_score for item in rollouts if item.total_score is not None])
