# Skill-CDPO

Official implementation of the paper: **Skill-CDPO: Evolving Agent Tool-Use via Critical Step Preference Optimization**

## Overview

Compact open-source language models lag behind their larger counterparts in agentic tool-use reliability, yet standard remedies face fundamental obstacles: supervised fine-tuning suffers from exposure bias, while reinforcement learning is hampered by sparse credit assignment over long tool-interaction trajectories.

Skill-CDPO introduces a progressive framework that first acquires tool-use skills at inference time through static tool analysis and dynamic strategy refinement, then distills the resulting error-correction signals into parameter updates via Critical Step DPO (CDPO).

## Architecture

- **Sub-Agent Pool**: Manages specialized sub-agents for different tasks
- **Reflexion Manager**: Handles self-reflection and correction
- **MCP Client**: Connects to MCP servers for tool execution
- **Tool Dispatcher**: Routes tools to appropriate agents

## MCP Tools

| Tool | Description |
|------|-------------|
| Google Search | Web search via Serper.dev |
| Google Scholar | Academic search via Serper.dev |
| PubMed | Medical literature search |
| Semantic Scholar | Academic paper search |
| FDA Drug Label | Drug information search |
| Webpage Fetch | Content extraction via Crawl4AI/Jina |

## Benchmarks

We contribute the following medical agent benchmarks:

### PubMed Search Lite
Test set for evaluating agent performance on medical literature retrieval tasks.

### PubMed Search Full
Training set for medical agent tool-use optimization.

## Citation

```
Coming soon
```
