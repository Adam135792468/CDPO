# Skill-CDPO

**Skill-CDPO: Evolving Agent Tool-Use via Critical Step Preference Optimization**

This repository contains the official implementation of Skill-CDPO, a progressive framework for evolving agent tool-use capabilities in compact language models.

## Overview

Compact open-source language models lag behind their larger counterparts in agentic tool-use reliability, yet standard remedies face fundamental obstacles: supervised fine-tuning suffers from exposure bias, while reinforcement learning is hampered by sparse credit assignment over long tool-interaction trajectories.

Skill-CDPO addresses this challenge through a two-stage approach:
1. **Training-Free Skill Acquisition**: Inference-time skill synthesis via static tool analysis and dynamic strategy refinement
2. **Critical Step DPO (CDPO)**: Distills error-correction signals into parameter updates through fine-grained preference optimization

## Architecture

### Phase 1: Training-Free Skill Acquisition

| Component | Description |
|-----------|-------------|
| **Static Optimization** | Analyzes tool implementations to build accurate mental models of tool capabilities |
| **Dynamic Optimization** | Learns refined usage strategies from runtime execution logs and error patterns |

### Phase 2: Critical Step DPO

CDPO identifies the specific trajectory steps where model capability is the bottleneck—through rollout divergence between a local policy and an expert model—and constructs distributional preference pairs from all cross-group rollouts at those steps, weighted by both step-level criticality and pair-level score gaps.

### Theoretical Properties

- **Variance Reduction**: $nm$-fold reduction in gradient variance (e.g., 16× with $n=m=4$)
- **Unbiased Estimation**: Criticality scores converge to true advantage gap
- **Sample Efficiency**: $\frac{nm}{\rho}$-fold reduction in preference pairs needed

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

## Citation

```bibtex
Coming soon
```

## Repository Structure

```
mcp_r/
├── dr_agent/              # Core agent implementation
│   ├── mcp_backend/      # MCP tool backend
│   │   ├── apis/         # API implementations
│   │   └── main.py        # MCP server
│   └── tool_interface/    # Tool interfaces
├── evaluation/           # Evaluation scripts
├── PubMed Search Lite.jsonl    # Test set (147 samples)
├── PubMed Search Full.jsonl     # Full test set (711 samples)
└── benchmark_visualization.html # Data statistics
```
