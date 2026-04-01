# CDPO Implementation Guide

## 1. Scope

This document describes the repository implementation of Critical Step Preference Optimization (CDPO) under `dr_agent/cdpo/`, including:

- benchmark-specific critical-step localization
- partial-context construction
- rubric-based rollout scoring
- grouped CDPO dataset construction
- grouped all-pairs CDPO loss

This document is implementation-oriented. It focuses on what the code in this repository does, what data format it expects, and how the pieces connect in practice.

## 2. Design Boundary

The most important benchmark-specific design rule is:

- `PubMed Search` must use multi-agent voting for critical-step selection.
- `CureBench` and `MedBrowseComp` may use the rule-based MCQ localizer.

These two paths are intentionally separated in code.

Reason:

- `PubMed Search` is an open-ended research benchmark. Critical steps are not reliably reducible to option-word frequency or other closed-form heuristics.
- `CureBench` and `MedBrowseComp` are definitive-answer benchmarks where option-aware heuristics are meaningful and cheaper.

In other words, the rule-based MCQ localizer is an appendix-style utility for closed-form tasks only. It must not be reused as the PubMed critical-step selector.

## 3. Package Layout

The CDPO implementation lives in:

```text
dr_agent/cdpo/
├── __init__.py
├── cli.py
├── context.py
├── dataset.py
├── loss.py
├── mcq_localization.py
├── prompts.py
├── scoring.py
├── types.py
├── utils.py
└── voting.py
```

Module responsibilities:

- `context.py`: parse `interleaved_text` into tool-use turns and export prevention/correction contexts
- `voting.py`: select PubMed critical steps with committee voting
- `scoring.py`: score rollout answers against weighted rubrics
- `dataset.py`: merge rollouts and scores into step-level CDPO records and flattened pairs
- `loss.py`: grouped all-pairs rubric CDPO loss
- `mcq_localization.py`: closed-form MCQ critical-turn localization
- `types.py`: stable dataclasses for all intermediate records
- `cli.py`: command-line entrypoints for the full pipeline

## 4. End-to-End Pipeline

The intended pipeline is:

1. Start from trajectory JSONL with `question`, `interleaved_text`, and rubric metadata.
2. Select critical steps.
3. Build partial contexts around those steps.
4. Generate local-policy and expert rollouts from each partial context.
5. Score each rollout with rubric-based evaluation.
6. Merge scores and rollouts into grouped CDPO step records.
7. Train with grouped all-pairs CDPO loss, or flatten the grouped records for pairwise training infrastructure.

The repository exposes this pipeline through `dr-cdpo` and `python -m dr_agent.cdpo.cli`.

## 5. Data Contracts

### 5.1 Trajectory Input

The raw trajectory input is expected to contain fields such as:

```json
{
  "sample_id": "pubmed-001",
  "question": "Which beta blocker blocked the tumorigenic effect?",
  "topic": "pubmed",
  "question_type": "review",
  "interleaved_text": "<think>...</think><call_tool name=\"pubmed_search\">...</call_tool><tool_output>...</tool_output>",
  "tool_calls": [],
  "all_rubrics": [
    {
      "category": "content",
      "title": "Evidence completeness",
      "description": "Covers the main evidence required by the question.",
      "weight": 0.5
    }
  ]
}
```

The parser in `context.py` assumes one tool-use turn is serialized as:

```text
<think>...</think><call_tool ...>...</call_tool><tool_output>...</tool_output>
```

Each parsed turn becomes a `TrajectoryTurn` with:

- `step_number`
- `assistant_content`
- `tool_output`
- `think_text`
- `tool_name`
- `tool_query`
- `tool_parameters`

### 5.2 Partial Rollout Record

The partial rollout format is designed to match the JSONL structure already used around this repository:

```json
{
  "sample_id": "pubmed-001",
  "question": "...",
  "topic": "pubmed",
  "question_type": "review",
  "critical_step": 2,
  "context_type": "prevention",
  "context_messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "all_rubrics": [],
  "dr_tulu_results": [],
  "openrouter_results": []
}
```

Important conventions:

- `dr_tulu_results` is the local-policy rollout group
- `openrouter_results` is the expert rollout group
- both rollout groups are stored side by side in the same partial-rollout record

### 5.3 Rollout Score Record

Rubric scoring produces a separate score file:

```json
{
  "sample_id": "pubmed-001",
  "critical_step": 2,
  "context_type": "prevention",
  "rollout_scores": [
    {
      "source": "dr_tulu",
      "index": 0,
      "scores": [],
      "total_score": 1.4,
      "max_possible_score": 3.0
    },
    {
      "source": "openrouter",
      "index": 0,
      "scores": [],
      "total_score": 2.3,
      "max_possible_score": 3.0
    }
  ]
}
```

### 5.4 Step-Level CDPO Record

After merging partial rollouts and score records, the repository builds grouped step-level records:

```json
{
  "sample_id": "pubmed-001",
  "critical_step": 2,
  "context_type": "prevention",
  "expert_source": "openrouter",
  "local_source": "dr_tulu",
  "expert_rollouts": [],
  "local_rollouts": [],
  "rubric_max_score": 3.0,
  "mean_expert_score": 2.25,
  "mean_local_score": 1.45,
  "criticality_score": 0.80,
  "criticality_ratio": 0.2667,
  "verified_critical": true,
  "step_weight": 1.12,
  "z_hat": 0.71,
  "pair_count": 4
}
```

This grouped representation is the core training artifact for CDPO.

## 6. PubMed Critical-Step Selection

### 6.1 Why Voting

For `PubMed Search`, the implementation uses a committee of judges instead of option-word heuristics.

The code treats a step as critical when:

- the model capability bottleneck is likely at that step
- a stronger expert continuation from the same state would materially improve the final answer
- the error has meaningful downstream impact under the rubric

This is operationalized through multi-agent voting in `voting.py`.

### 6.2 Candidate Prefilter

The function `propose_pubmed_candidate_steps()` is a cheap prefilter that raises recall-oriented candidates by looking for:

- tool errors
- zero-result searches
- repeated low-quality queries
- timeout-like failures

This prefilter is only a cost reducer. It is not the final selector.

### 6.3 Committee Vote

The final selector is `MultiAgentCriticalStepSelector`.

Each judge receives:

- the question
- the serialized trajectory turns
- optional candidate steps
- optional rubric signals

Each judge returns JSON like:

```json
{
  "critical_steps": [
    {
      "step_number": 3,
      "is_critical": true,
      "confidence": 0.87,
      "rationale": "Search adaptation failed here."
    }
  ]
}
```

The selector then:

1. parses every judge vote
2. aggregates votes by step
3. computes `yes_votes` and average confidence
4. accepts steps that meet quorum
5. falls back to the top-ranked step if no step reaches quorum

The output includes:

- `candidate_steps`
- `critical_steps`
- raw `votes`
- `vote_summary`
- `metadata`

## 7. Closed-Form MCQ Localization

The file `mcq_localization.py` implements a separate localizer for closed-form tasks.

It works by:

1. building token profiles from answer options
2. comparing reasoning traces against the correct and incorrect option profiles
3. estimating where the model shifted toward the wrong answer
4. assigning a subtype such as:
   - `misled_by_evidence`
   - `final_reasoning_error`
   - `evidence_ignored`
   - `insufficient_search`

This is useful for:

- `CureBench`
- `MedBrowseComp`

This is not valid as the primary PubMed critical-step selector.

## 8. Partial Context Construction

CDPO needs partial contexts that branch from the original trajectory at the identified critical step.

The repository supports two context types:

- `prevention`: truncate before the critical step
- `correction`: include the critical step and its tool output

### 8.1 Prevention Context

`build_prevention_context()` keeps all turns strictly before `critical_step`.

This is used when the training target is:

- avoid making the critical mistake in the first place

### 8.2 Correction Context

`build_correction_context()` includes all turns up to and including `critical_step`.

This is used when the training target is:

- recover correctly after the critical step has already happened

### 8.3 Exported Message Format

The exported `context_messages` follow chat-style message lists:

```json
[
  {"role": "system", "content": "..."},
  {"role": "user", "content": "question text"},
  {"role": "assistant", "content": "<think>...</think><call_tool ...>...</call_tool>"},
  {"role": "user", "content": "<tool_output>...</tool_output>"}
]
```

This preserves compatibility with the repository's rollout-generation pipeline.

## 9. Rubric-Based Rollout Scoring

The scorer in `scoring.py` uses a model judge through LiteLLM, but it does not trust the model to compute the final weighted total.

Instead:

1. the judge returns one integer score in `{0, 1, 2, 3}` per rubric item
2. the repository computes the weighted total locally

For rubric item `k` with weight `w_k` and integer score `s_k`:

```text
weighted_score_k = w_k * s_k
```

The total score is:

```text
total_score = sum_k weighted_score_k
```

The maximum possible score is:

```text
max_possible_score = sum_k (3 * w_k)
```

This local aggregation avoids instability from model-side arithmetic errors and keeps score semantics deterministic.

## 10. Step Verification and Weighting

After rollouts are scored, `dataset.py` constructs grouped step records.

For one step:

- let `E` be the expert rollout group
- let `L` be the local rollout group

Then:

```text
mean_expert_score = mean(score(e) for e in E)
mean_local_score = mean(score(l) for l in L)
criticality_score = mean_expert_score - mean_local_score
criticality_ratio = criticality_score / rubric_max_score
```

A step is marked as verified critical when either:

```text
criticality_score > epsilon
```

or:

```text
criticality_ratio > epsilon
```

depending on `epsilon_mode`.

### 10.1 Step Weight Normalization

With exponent `alpha`, the raw step importance is:

```text
raw_step_weight = |criticality_score|^alpha
```

The normalizer is:

```text
z_hat = mean(raw_step_weight over the normalization pool)
```

Then:

```text
step_weight = raw_step_weight / z_hat
```

The normalization pool is configurable:

- `all`
- `verified`

### 10.2 Pair Weight

For each expert-local rollout pair `(i, j)` inside one step:

```text
pair_weight_ij = (expert_score_i - local_score_j) / rubric_max_score
```

The grouped step record can also be flattened into explicit pair rows for trainer interoperability.

## 11. Grouped All-Pairs CDPO Loss

The loss in `loss.py` implements grouped all-pairs rubric CDPO.

For each step:

- compare every expert rollout against every local rollout
- weigh each pair by the normalized rubric score gap
- multiply the step contribution by `step_weight`

In code form, the key ingredients are:

```text
expert_logratio_i = expert_policy_logp_i - expert_reference_logp_i
local_logratio_j = local_policy_logp_j - local_reference_logp_j
logit_ij = beta * (expert_logratio_i - local_logratio_j)
pair_weight_ij = (expert_score_i - local_score_j) / rubric_max_score
```

The per-pair term is:

```text
- pair_weight_ij * log_sigmoid(logit_ij)
```

The per-step loss is the average over valid pairs, scaled by `step_weight`.

This grouped form keeps the distributional signal of multiple expert and local continuations instead of collapsing a step to one winner and one loser.

## 12. CLI Workflow

The repository exposes the following commands:

### 12.1 Select PubMed Critical Steps

```bash
dr-cdpo select-pubmed-critical-steps \
  --input-file trajectories.jsonl \
  --output-file critical_steps.jsonl \
  --judge-model openai/gpt-4.1-mini \
  --committee-size 3 \
  --candidate-mode heuristic
```

Use this for PubMed only.

### 12.2 Build Partial Contexts

```bash
dr-cdpo build-partial-contexts \
  --trajectory-file trajectories.jsonl \
  --critical-step-file critical_steps.jsonl \
  --output-file partial_contexts.jsonl
```

### 12.3 Score Rollouts

```bash
dr-cdpo score-rollouts \
  --partial-rollout-file partial_rollouts_with_generations.jsonl \
  --output-file rollout_scores.jsonl \
  --model openai/gpt-4.1-mini
```

### 12.4 Build Grouped Step Records

```bash
dr-cdpo build-step-records \
  --partial-rollout-file partial_rollouts_with_generations.jsonl \
  --score-file rollout_scores.jsonl \
  --output-file cdpo_step_records.jsonl \
  --epsilon 0.3 \
  --epsilon-mode raw
```

### 12.5 Flatten Pair Records

```bash
dr-cdpo flatten-pairs \
  --step-record-file cdpo_step_records.jsonl \
  --output-file cdpo_pairs.jsonl
```

### 12.6 Localize MCQ Critical Turns

```bash
dr-cdpo localize-mcq-critical-turns \
  --input-file mcq_trajectories.jsonl \
  --output-file mcq_critical_turns.jsonl
```

Use this only for closed-form MCQ benchmarks.

## 13. Training Integration Notes

The grouped CDPO records are the preferred training interface because they preserve:

- multiple expert continuations
- multiple local continuations
- step-level criticality
- pair-level score gaps

If the trainer can directly consume grouped tensors, use:

- expert policy log-probabilities
- local policy log-probabilities
- optional reference log-probabilities
- expert rollout scores
- local rollout scores
- `rubric_max_scores`
- `step_weights`

and call `rubric_cdpo_loss()`.

If the trainer only supports flat pairwise data, use `flatten-pairs`, but note that this loses the grouped structure at the data interface level.

## 14. Verification Status

The repository includes CDPO-focused tests covering:

- trajectory parsing
- prevention/correction context construction
- grouped step-record construction
- MCQ localizer behavior
- PubMed committee voting behavior
- rubric scorer weighted-total computation
- loss tensor shape behavior when `torch` is installed

At the time this document was added, the non-`torch` CDPO tests passed locally under Python 3.13 in the repository virtual environment.

## 15. Practical Recommendations

- Use multi-agent voting for `PubMed Search` even if a heuristic candidate prefilter is enabled.
- Keep rubric weights normalized and explicit in input data.
- Prefer grouped step-level training over immediately flattening to pairs.
- Treat `epsilon=0.3` in raw score space as a reasonable starting point for the current PubMed-style rubric scale where `max_possible_score` is often around `3.0`.
- Do not mix the MCQ localizer into the PubMed pipeline.

## 16. Summary

This repository implementation of CDPO is designed to be:

- benchmark-aware
- schema-compatible with the surrounding rollout data
- modular enough for offline dataset construction and training
- strict about the PubMed versus MCQ localization boundary

For open-ended PubMed trajectories, the critical-step selector is committee-based.
For closed-form MCQ benchmarks, the repository provides a separate rule-based localizer.
The downstream training signal is constructed through rubric-scored grouped step records and optimized with grouped all-pairs CDPO loss.
