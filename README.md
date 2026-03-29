# Skill-CDPO

Official implementation of the paper: **Skill-CDPO: Evolving Agent Tool-Use via Critical Step Preference Optimization**

## Overview

This repository contains the official implementation of Skill-CDPO, a method for evolving agent tool-use through critical step preference optimization.

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

## Citation

```
Coming soon
```
