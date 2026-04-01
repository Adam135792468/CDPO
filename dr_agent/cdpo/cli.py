from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

import typer

from .context import DEFAULT_PUBMED_SYSTEM_PROMPT, build_partial_context_records
from .dataset import (
    build_cdpo_step_records,
    flatten_step_record_to_pair_records,
    load_partial_rollout_records,
    load_rollout_score_records,
    summarize_step_records,
)
from .mcq_localization import localize_mcq_critical_turn
from .scoring import LiteLLMRubricScorer, RubricScorerConfig
from .types import CDPOStepRecord, PartialRolloutRecord
from .utils import read_jsonl, write_jsonl
from .voting import (
    CriticalStepJudgeConfig,
    MultiAgentCriticalStepSelector,
    propose_pubmed_candidate_steps,
    select_pubmed_critical_steps,
)


app = typer.Typer(help="CDPO dataset construction and analysis utilities.", no_args_is_help=True)


def _build_judges(
    judge_models: list[str],
    *,
    api_key: str | None,
    base_url: str | None,
    temperature: float,
    committee_size: int,
) -> list[CriticalStepJudgeConfig]:
    if not judge_models:
        raise typer.BadParameter("At least one --judge-model must be provided.")

    if len(judge_models) == 1 and committee_size > 1:
        model = judge_models[0]
        return [
            CriticalStepJudgeConfig(
                model=model,
                name=f"{model}#{idx+1}",
                temperature=temperature,
                api_key=api_key,
                base_url=base_url,
                seed=idx + 1,
            )
            for idx in range(committee_size)
        ]

    return [
        CriticalStepJudgeConfig(
            model=model,
            name=f"{model}#{idx+1}",
            temperature=temperature,
            api_key=api_key,
            base_url=base_url,
        )
        for idx, model in enumerate(judge_models)
    ]


@app.command("select-pubmed-critical-steps")
def select_pubmed_critical_steps_cli(
    input_file: Path = typer.Option(..., exists=True, readable=True, help="Trajectory JSONL with question/interleaved_text."),
    output_file: Path = typer.Option(..., help="Output JSONL of committee-selected critical steps."),
    judge_model: list[str] = typer.Option(..., "--judge-model", help="Judge model(s). Repeat for a heterogeneous committee."),
    api_key: str | None = typer.Option(None, help="Optional API key passed to LiteLLM."),
    base_url: str | None = typer.Option(None, help="Optional LiteLLM base URL."),
    temperature: float = typer.Option(0.0, help="Judge temperature."),
    committee_size: int = typer.Option(3, help="Committee size when only one judge model is provided."),
    quorum: int | None = typer.Option(None, help="Votes required to accept a step. Defaults to majority."),
    max_steps: int = typer.Option(2, help="Maximum critical steps to select per sample."),
    candidate_mode: str = typer.Option("heuristic", help="Candidate prefilter: heuristic or all."),
) -> None:
    judges = _build_judges(
        judge_model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        committee_size=committee_size,
    )
    selector = MultiAgentCriticalStepSelector(judges, quorum=quorum, max_selected_steps=max_steps)

    async def _run() -> list[dict]:
        rows: list[dict] = []
        for sample in read_jsonl(input_file):
            if candidate_mode == "heuristic":
                candidate_steps = propose_pubmed_candidate_steps(sample)
            else:
                candidate_steps = None
            selection = await select_pubmed_critical_steps(
                sample,
                selector,
                candidate_steps=candidate_steps,
            )
            rows.append(selection.to_dict())
        return rows

    rows = asyncio.run(_run())
    write_jsonl(output_file, rows)
    typer.echo(f"Wrote {len(rows)} critical-step selections to {output_file}")


@app.command("build-partial-contexts")
def build_partial_contexts_cli(
    trajectory_file: Path = typer.Option(..., exists=True, readable=True, help="Trajectory JSONL with interleaved_text."),
    critical_step_file: Path = typer.Option(..., exists=True, readable=True, help="Critical-step JSONL from select-pubmed-critical-steps."),
    output_file: Path = typer.Option(..., help="Output JSONL containing prevention/correction contexts."),
    system_prompt_file: Path | None = typer.Option(None, exists=True, readable=True, help="Optional custom system prompt file."),
) -> None:
    system_prompt = DEFAULT_PUBMED_SYSTEM_PROMPT
    if system_prompt_file is not None:
        system_prompt = system_prompt_file.read_text(encoding="utf-8")

    critical_map = {
        row["sample_id"]: row
        for row in read_jsonl(critical_step_file)
    }

    rows: list[dict] = []
    for sample in read_jsonl(trajectory_file):
        sample_id = sample.get("sample_id")
        critical_info = critical_map.get(sample_id)
        if not critical_info:
            continue
        critical_steps = critical_info.get("critical_steps", [])
        rows.extend(
            build_partial_context_records(
                sample,
                critical_steps,
                system_prompt=system_prompt,
            )
        )

    write_jsonl(output_file, rows)
    typer.echo(f"Wrote {len(rows)} partial context rows to {output_file}")


@app.command("score-rollouts")
def score_rollouts_cli(
    partial_rollout_file: Path = typer.Option(..., exists=True, readable=True, help="Partial rollout JSONL with dr_tulu_results/openrouter_results."),
    output_file: Path = typer.Option(..., help="Output rollout-score JSONL."),
    model: str = typer.Option(..., help="Judge model used for rubric scoring."),
    api_key: str | None = typer.Option(None, help="Optional API key passed to LiteLLM."),
    base_url: str | None = typer.Option(None, help="Optional LiteLLM base URL."),
    temperature: float = typer.Option(0.1, help="Judge temperature."),
    max_tokens: int = typer.Option(1600, help="Max tokens per scoring call."),
    concurrency: int = typer.Option(20, help="Maximum concurrent scoring calls."),
) -> None:
    partial_records = load_partial_rollout_records(str(partial_rollout_file))
    scorer = LiteLLMRubricScorer(
        RubricScorerConfig(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            max_concurrent_requests=concurrency,
        )
    )

    async def _run() -> list[dict]:
        results = []
        for record in partial_records:
            scored = await scorer.score_partial_rollout_record(record)
            results.append(scored.to_dict())
        return results

    rows = asyncio.run(_run())
    write_jsonl(output_file, rows)
    typer.echo(f"Wrote {len(rows)} rollout-score rows to {output_file}")


@app.command("build-step-records")
def build_step_records_cli(
    partial_rollout_file: Path = typer.Option(..., exists=True, readable=True, help="Partial rollout JSONL."),
    score_file: Path = typer.Option(..., exists=True, readable=True, help="Rollout-score JSONL."),
    output_file: Path = typer.Option(..., help="Output grouped CDPO step-record JSONL."),
    expert_source: str = typer.Option("openrouter", help="Expert rollout source key."),
    local_source: str = typer.Option("dr_tulu", help="Local rollout source key."),
    alpha: float = typer.Option(1.0, help="Criticality weighting exponent."),
    epsilon: float = typer.Option(0.3, help="Criticality threshold."),
    epsilon_mode: str = typer.Option("raw", help="Threshold space: raw or ratio."),
    normalization_pool: str = typer.Option("all", help="Normalizer pool: all or verified."),
    verified_only: bool = typer.Option(True, help="Keep only verified critical steps."),
) -> None:
    records = build_cdpo_step_records(
        str(partial_rollout_file),
        str(score_file),
        expert_source=expert_source,
        local_source=local_source,
        alpha=alpha,
        epsilon=epsilon,
        epsilon_mode=epsilon_mode,
        normalization_pool=normalization_pool,
        verified_only=verified_only,
    )
    write_jsonl(output_file, [record.to_dict() for record in records])
    typer.echo(json.dumps(summarize_step_records(records), ensure_ascii=False, indent=2))
    typer.echo(f"Wrote {len(records)} grouped CDPO step records to {output_file}")


@app.command("flatten-pairs")
def flatten_pairs_cli(
    step_record_file: Path = typer.Option(..., exists=True, readable=True, help="Grouped CDPO step-record JSONL."),
    output_file: Path = typer.Option(..., help="Output flattened pair JSONL."),
    text_field: str = typer.Option("interleaved_text", help="Rollout text field: interleaved_text or model_answer."),
) -> None:
    rows: list[dict] = []
    for item in read_jsonl(step_record_file):
        record = CDPOStepRecord.from_dict(item)
        rows.extend(flatten_step_record_to_pair_records(record, text_field=text_field))
    write_jsonl(output_file, rows)
    typer.echo(f"Wrote {len(rows)} flattened pair rows to {output_file}")


@app.command("localize-mcq-critical-turns")
def localize_mcq_critical_turns_cli(
    input_file: Path = typer.Option(..., exists=True, readable=True, help="MCQ trajectory JSONL with options/correct_answer/model_answer/interleaved_text."),
    output_file: Path = typer.Option(..., help="Output JSONL with MCQ critical-turn localization."),
) -> None:
    rows: list[dict] = []
    for sample in read_jsonl(input_file):
        result = localize_mcq_critical_turn(
            question=str(sample.get("question", "")),
            options=dict(sample.get("options", {}) or {}),
            correct_answer=str(sample.get("correct_answer", "")),
            model_answer=str(sample.get("model_answer", "")),
            interleaved_text=str(sample.get("interleaved_text", "")),
        )
        rows.append(
            {
                "sample_id": sample.get("sample_id", sample.get("id", "")),
                "question": sample.get("question", ""),
                **result.to_dict(),
            }
        )
    write_jsonl(output_file, rows)
    typer.echo(f"Wrote {len(rows)} MCQ localization rows to {output_file}")


if __name__ == "__main__":
    app()
