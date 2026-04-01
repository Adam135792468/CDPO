# Skill-CDPO

**Skill-CDPO: Evolving Agent Tool-Use via Critical Step Preference Optimization**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

A progressive framework for evolving agent tool-use capabilities in compact language models through Critical Step Preference Optimization.

## Overview

Compact open-source language models lag behind their larger counterparts in agentic tool-use reliability, yet standard remedies face fundamental obstacles: supervised fine-tuning suffers from exposure bias, while reinforcement learning is hampered by sparse credit assignment over long tool-interaction trajectories.

Skill-CDPO addresses this challenge through a two-stage approach:
1. **Training-Free Skill Acquisition**: Inference-time skill synthesis via static tool analysis and dynamic strategy refinement
2. **Critical Step DPO (CDPO)**: Distills error-correction signals into parameter updates through fine-grained preference optimization

## Framework

![Skill-CDPO Framework](./figures/framework.png)

## Architecture

### Phase 1: Training-Free Skill Acquisition

| Component | Description |
|-----------|-------------|
| **Static Optimization** | Analyzes tool implementations to build accurate mental models of tool capabilities |
| **Dynamic Optimization** | Learns refined usage strategies from runtime execution logs and error patterns |

### Phase 2: Critical Step DPO

CDPO identifies the specific trajectory steps where model capability is the bottleneck—through rollout divergence between a local policy and an expert model—and constructs distributional preference pairs from all cross-group rollouts at those steps, weighted by both step-level criticality and pair-level score gaps.

## CDPO Code

The repository includes an engineering-grade implementation of the paper's CDPO pipeline under `dr_agent/cdpo/`.

Important benchmark split:

- **PubMed Search**: critical-step selection is implemented with **multi-agent voting** over trajectory turns. A lightweight heuristic candidate proposer is available only to reduce committee cost; it does **not** make the final decision.
- **CureBench / MedBrowseComp**: the appendix-style rule-based critical-turn localizer is implemented separately for closed-form MCQ settings and should **not** be used for PubMed.

Core modules:

| Module | Purpose |
|--------|---------|
| `dr_agent/cdpo/context.py` | Parse `interleaved_text` and build prevention/correction partial contexts |
| `dr_agent/cdpo/voting.py` | PubMed critical-step proposal + multi-agent voting committee |
| `dr_agent/cdpo/scoring.py` | Rubric-based rollout scoring via LiteLLM, with weighted totals computed locally |
| `dr_agent/cdpo/dataset.py` | Merge partial rollouts and rubric scores into grouped CDPO step records |
| `dr_agent/cdpo/loss.py` | Grouped all-pairs rubric CDPO loss |
| `dr_agent/cdpo/mcq_localization.py` | Closed-form MCQ critical-turn localization for CureBench / MedBrowseComp |
| `dr_agent/cdpo/cli.py` | End-to-end dataset construction CLI |

The implementation is aligned with the JSONL schemas already used in this repository and adjacent data pipelines, including fields such as `context_messages`, `critical_step`, `dr_tulu_results`, `openrouter_results`, and `all_rubrics`.

Implementation notes and algorithm details are documented in [docs/CDPO.md](./docs/CDPO.md).

### Theoretical Properties

- **Variance Reduction**: $nm$-fold reduction in gradient variance (e.g., 16× with $n=m=4$)
- **Unbiased Estimation**: Criticality scores converge to true advantage gap
- **Sample Efficiency**: $\frac{nm}{\rho}$-fold reduction in preference pairs needed

## Installation

```bash
pip install -e .
# or, for tests as well:
pip install -e '.[dev]'
```

After installation, the CDPO CLI is available as:

```bash
dr-cdpo --help
```

If you prefer not to install the package in editable mode, you can also invoke it with:

```bash
python -m dr_agent.cdpo.cli --help
```

## CDPO Workflow

The intended open-source workflow is:

1. Run PubMed critical-step selection with committee voting.
2. Build prevention/correction partial contexts.
3. Collect local-policy and expert rollouts into the existing partial-rollout JSONL format.
4. Score rollouts against rubric items.
5. Build grouped step-level CDPO records.
6. Flatten grouped records into pairwise preference rows if needed by the trainer.

Example commands:

```bash
dr-cdpo select-pubmed-critical-steps \
  --input-file trajectories.jsonl \
  --output-file critical_steps.jsonl \
  --judge-model openai/gpt-4.1-mini \
  --committee-size 3 \
  --candidate-mode heuristic

dr-cdpo build-partial-contexts \
  --trajectory-file trajectories.jsonl \
  --critical-step-file critical_steps.jsonl \
  --output-file partial_contexts.jsonl

dr-cdpo score-rollouts \
  --partial-rollout-file partial_rollouts_with_generations.jsonl \
  --output-file rollout_scores.jsonl \
  --model openai/gpt-4.1-mini

dr-cdpo build-step-records \
  --partial-rollout-file partial_rollouts_with_generations.jsonl \
  --score-file rollout_scores.jsonl \
  --output-file cdpo_step_records.jsonl \
  --epsilon 0.3 \
  --epsilon-mode raw

dr-cdpo flatten-pairs \
  --step-record-file cdpo_step_records.jsonl \
  --output-file cdpo_pairs.jsonl
```

For closed-form MCQ benchmarks only:

```bash
dr-cdpo localize-mcq-critical-turns \
  --input-file mcq_trajectories.jsonl \
  --output-file mcq_critical_turns.jsonl
```

## MCP Tools

Our agent interacts with the environment through a suite of MCP (Model Context Protocol) tools:

| Tool | Description | Used In |
|------|-------------|---------|
| `google_search` | Web search via Serper API | All benchmarks |
| `browse_webpage` | Fetch URL content in Markdown | All benchmarks |
| `fda_drug_search` | FDA drug label database query | CureBench |
| `pubmed_search` | PubMed literature search | PubMed Search |
| `medbrowsecomp_search` | Medical info search with routing | MedBrowseComp |
| `get_trial_info` | ClinicalTrials.gov query | MedBrowseComp |
| `get_drug_patents` | FDA Orange Book patents | MedBrowseComp |
| `get_drug_approvals` | FDA drug approvals | MedBrowseComp |
| `get_drug_exclusivities` | FDA drug exclusivities | MedBrowseComp |

## Benchmarks

We evaluate on three medical agent benchmarks:

| Benchmark | Description | Evaluation |
|-----------|-------------|------------|
| **PubMed Search** (contributed) | PubMed-based deep research benchmark for biomedical literature retrieval | Rubric-based scoring |
| **CureBench** | Multiple-choice medical QA | Accuracy |
| **MedBrowseComp** | Closed-form medical QA | Accuracy |

### Dataset Statistics

| Dataset | Samples | Topics | Question Types |
|---------|---------|--------|----------------|
| PubMed Search Lite | 147 | 111 | 5 |
| PubMed Search Full | 711 | 416 | 6 |

## Main Results

| Method | PubMed Search ↑ | CureBench ↑ | MedBrowseComp ↑ |
|--------|-----------------|-------------|-----------------|
| GPT-5.2 (closed) | 19.08 | 71.05% | 27.93% |
| DR-Tulu-8B (base) | 15.48 | 59.30% | 25.12% |
| + Skill | 17.75 | 62.41% | 26.61% |
| + SFT | 17.93 | 63.42% | 27.44% |
| + Trajectory DPO | 18.47 | 63.05% | 27.19% |
| + Step DPO | 18.83 | 64.25% | 28.06% |
| **Skill-CDPO** | **19.58** | **66.63%** | **29.09%** |

Skill-CDPO achieves **competitive or superior performance compared to GPT-5.2** on retrieval-intensive tasks using only an 8B-parameter model.

## Key Findings

1. **Skill synthesis is necessary**: Training-free skill acquisition alone yields substantial gains (+2.27 on PubMed Search) without parameter updates

2. **Standard training methods are insufficient**: SFT and trajectory-level DPO provide only marginal improvements on top of skill augmentation

3. **CDPO fully unlocks potential**: Critical step preference optimization captures learning signals that other methods miss

## Repository Structure

```
mcp_r/
├── dr_agent/              # Core agent implementation
│   ├── cdpo/             # CDPO dataset builders, scoring, voting, and loss
│   ├── mcp_backend/      # MCP tool backend
│   │   ├── apis/         # API implementations
│   │   └── main.py       # MCP server
│   └── tool_interface/    # Tool interfaces
├── evaluation/            # Evaluation scripts
├── figures/               # Framework diagrams
├── docs/                  # Additional implementation documentation
├── tests/                 # Unit tests, including CDPO coverage
├── PubMed Search Lite.jsonl    # Test set (147 samples)
├── PubMed Search Full.jsonl     # Full test set (711 samples)
├── benchmark_visualization.html # Data statistics
├── pyproject.toml         # Project metadata and dependencies
└── LICENSE               # MIT License
```
