from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .context import parse_interleaved_turns
from .utils import normalize_whitespace


DEFAULT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "but",
    "by",
    "can",
    "for",
    "from",
    "had",
    "has",
    "have",
    "in",
    "is",
    "it",
    "not",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "with",
    "will",
}


@dataclass(slots=True)
class MCQCriticalTurnResult:
    critical_turns: list[int]
    error_subtype: str
    gap_trajectory: list[float]
    lct: int | None
    mst: int | None
    diagnosis: str
    turn_scores: list[dict[str, float]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "critical_turns": self.critical_turns,
            "error_subtype": self.error_subtype,
            "gap_trajectory": self.gap_trajectory,
            "lct": self.lct,
            "mst": self.mst,
            "diagnosis": self.diagnosis,
            "turn_scores": self.turn_scores,
        }


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _keyword_profiles(options: dict[str, str]) -> dict[str, dict[str, float]]:
    option_tokens: dict[str, set[str]] = {}
    for label, text in options.items():
        tokens = {token for token in _tokenize(text) if token not in DEFAULT_STOPWORDS}
        option_tokens[label] = tokens

    profiles: dict[str, dict[str, float]] = {}
    for label, tokens in option_tokens.items():
        others = set().union(*(option_tokens[other] for other in option_tokens if other != label))
        unique_tokens = tokens - others
        shared_tokens = tokens - unique_tokens
        weighted = {token: 1.0 for token in unique_tokens}
        weighted.update({token: 0.3 for token in shared_tokens})
        profiles[label] = weighted
    return profiles


def _resolve_option_label(answer: str, options: dict[str, str]) -> str | None:
    answer = (answer or "").strip().upper()
    if answer in options:
        return answer
    for label, text in options.items():
        if answer and answer in text.upper():
            return label
    return None


def _count_keywords(text: str, keyword_weights: dict[str, float]) -> float:
    tokens = _tokenize(text)
    score = 0.0
    for token in tokens:
        score += keyword_weights.get(token, 0.0)
    return score


def localize_mcq_critical_turn(
    *,
    question: str,
    options: dict[str, str],
    correct_answer: str,
    model_answer: str,
    interleaved_text: str,
) -> MCQCriticalTurnResult:
    """Rule-based MCQ critical-turn localization for closed-form tasks.

    This is the appendix algorithm used for definitive-answer benchmarks such as
    CureBench or MedBrowseComp. It is intentionally kept separate from the
    PubMed multi-agent voting pipeline.
    """

    turns = parse_interleaved_turns(interleaved_text)
    profiles = _keyword_profiles(options)

    correct_label = _resolve_option_label(correct_answer, options)
    wrong_label = _resolve_option_label(model_answer, options)
    if correct_label is None or wrong_label is None:
        raise ValueError("Could not resolve correct/model answer labels from options.")

    turn_scores: list[dict[str, float]] = []
    for turn in turns:
        reasoning = normalize_whitespace(turn.think_text)
        turn_scores.append(
            {
                label: _count_keywords(reasoning, keyword_weights)
                for label, keyword_weights in profiles.items()
            }
        )

    gap = [
        float(scores.get(wrong_label, 0.0) - scores.get(correct_label, 0.0))
        for scores in turn_scores
    ]

    lct = None
    for idx, value in enumerate(gap):
        if value < 0:
            lct = idx + 1

    mst = None
    best_delta = 0.0
    for idx in range(1, len(gap)):
        delta = gap[idx] - gap[idx - 1]
        if delta > best_delta:
            best_delta = delta
            mst = idx + 1

    correct_keywords = {
        token for token, weight in profiles[correct_label].items() if weight >= 1.0
    }
    any_correct_evidence = False
    for turn in turns:
        output_tokens = set(_tokenize(turn.tool_output))
        if correct_keywords & output_tokens:
            any_correct_evidence = True
            break

    correct_ever_favored = lct is not None

    if correct_ever_favored and mst is not None:
        critical_turns = sorted({lct, mst})
        subtype = "misled_by_evidence"
        diagnosis = (
            f"Model favored the correct answer until turn {lct}, then shifted most sharply "
            f"toward the wrong answer at turn {mst}."
        )
    elif correct_ever_favored and mst is None:
        critical_turns = [max(1, len(turns) - 1), len(turns)] if turns else []
        subtype = "final_reasoning_error"
        diagnosis = "Model leaned correct during evidence gathering but produced the wrong final choice."
    elif any_correct_evidence:
        matching_turns = []
        for idx, turn in enumerate(turns, start=1):
            output_tokens = set(_tokenize(turn.tool_output))
            if correct_keywords & output_tokens:
                matching_turns.append(idx)
                if idx < len(turns):
                    matching_turns.append(idx + 1)
        critical_turns = sorted(set(matching_turns))
        subtype = "evidence_ignored"
        diagnosis = "Correct evidence appeared in tool outputs but never became dominant in reasoning."
    else:
        critical_turns = [1]
        if gap:
            max_wrong_turn = max(range(len(gap)), key=lambda idx: gap[idx]) + 1
            critical_turns.append(max_wrong_turn)
        critical_turns = sorted(set(critical_turns))
        subtype = "insufficient_search"
        diagnosis = "The model never retrieved evidence supporting the correct option."

    return MCQCriticalTurnResult(
        critical_turns=critical_turns,
        error_subtype=subtype,
        gap_trajectory=gap,
        lct=lct,
        mst=mst,
        diagnosis=diagnosis,
        turn_scores=turn_scores,
    )
