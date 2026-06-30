"""
Tool Registry for the AI Data Analyst Agent.
Defines the interface for all tools accessible to the Gemini LLM.
Phase 3: Tool Implementation
"""

from dataclasses import dataclass
from typing import Any, Optional, Callable
import importlib

@dataclass
class ToolResult:
    """
    Standardised result returned by every tool.
    """
    success: bool
    tool_name: str
    output: Any           # the actual result (df, chart, string, etc.)
    output_type: str      # "dataframe", "chart", "string", "error"
    error_message: Optional[str]
    execution_time_ms: float
    code_executed: Optional[str]   # the SQL or Python that ran, if any

def get_tool_by_name(name: str) -> Callable:
    """
    Returns the actual Python function for a tool name.
    Maps string names to imported functions.
    """
    tool_map = {
        "schema_inspector": "tools.schema_inspector",
        "sql_executor": "tools.sql_executor",
        "python_executor": "tools.python_executor",
        "chart_generator": "tools.chart_generator",
        "insight_narrator": "tools.insight_narrator",
        "anomaly_detector": "tools.anomaly_detector",
        "memory_retriever": "tools.memory_retriever",
        "report_builder": "tools.report_builder"
    }
    
    if name not in tool_map:
        raise ValueError(f"Tool '{name}' not found in registry.")
    
    module_path = tool_map[name]
    module = importlib.import_module(module_path)
    
    return getattr(module, "run")
