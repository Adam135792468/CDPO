# MCP-R

Official implementation of the paper: **MCP-R: Multi-Agent MCP Tools for RAG with Reflexion and Self-Correction**

## Overview

This repository contains the official implementation of MCP-R, a multi-agent system for deep research with MCP-based tool backend. The system features reflexion and self-correction capabilities.

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
