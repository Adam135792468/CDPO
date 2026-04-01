__version__ = "0.0.1"

__all__ = [
    # Core components
    "BaseAgent",
    "LLMToolClient",
    "GenerateWithToolsOutput",
    "GenerationConfig",
    "BaseWorkflow",
    "BaseWorkflowConfiguration",
    # Tool interface - Core
    "BaseTool",
    "ToolInput",
    "ToolOutput",
    "AgentAsTool",
    "ChainedTool",
    # Tool interface - Data types
    "Document",
    "DocumentToolOutput",
    # Tool interface - MCP Tools
    "MCPMixin",
    "SemanticScholarSnippetSearchTool",
    "SerperSearchTool",
    "MassiveServeSearchTool",
    "SerperBrowseTool",
    "VllmHostedRerankerTool",
    # Tool interface - Parsing
    "ToolCallInfo",
    "ToolCallParser",
    # Prompts
    "UNIFIED_TOOL_CALLING_PROMPTS",
]

# Import the full agent stack lazily so that lightweight subpackages such as
# `dr_agent.cdpo` remain usable in minimal environments that do not install the
# complete runtime dependency set. This keeps the package importable for data
# processing and training utilities while preserving the original symbols when
# dependencies are available.
try:  # pragma: no cover - exercised indirectly in integration environments
    # Main library components
    from .agent_interface import BaseAgent
    from .client import GenerateWithToolsOutput, GenerationConfig, LLMToolClient

    # Shared prompts
    from .shared_prompts import UNIFIED_TOOL_CALLING_PROMPTS

    # Tool interface components
    from .tool_interface import (
        AgentAsTool,
        BaseTool,
        ChainedTool,
        Document,
        DocumentToolOutput,
        MassiveServeSearchTool,
        MCPMixin,
        SemanticScholarSnippetSearchTool,
        SerperBrowseTool,
        SerperSearchTool,
        ToolCallInfo,
        ToolCallParser,
        ToolInput,
        ToolOutput,
        VllmHostedRerankerTool,
    )
    from .workflow import BaseWorkflow, BaseWorkflowConfiguration
except Exception:  # noqa: BLE001
    pass
