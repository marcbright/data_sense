"""
Schema Inspector Tool for the AI Data Analyst Agent.
Phase 3: Tool Implementation
"""

import time
from data_engine.schema_inspector import inspect_schema, schema_to_prompt_string
from tools.registry import ToolResult

def run(session_state: dict) -> ToolResult:
    """
    Analyzes the active dataset and generates context for the LLM.
    """
    start_time = time.time()
    
    df = session_state.get("active_dataframe")
    filename = session_state.get("active_filename", "unknown_file")
    
    if df is None:
        execution_time = (time.time() - start_time) * 1000
        return ToolResult(
            success=False,
            tool_name="schema_inspector",
            output=None,
            output_type="error",
            error_message="No dataset loaded. Please upload a file first.",
            execution_time_ms=execution_time,
            code_executed=None
        )
    
    try:
        # Profile the schema
        profile = inspect_schema(df, filename)
        prompt_string = schema_to_prompt_string(profile)
        
        # Store in session state for other tools/agent to use
        session_state["schema_profile"] = profile
        session_state["schema_context"] = prompt_string
        
        execution_time = (time.time() - start_time) * 1000
        
        return ToolResult(
            success=True,
            tool_name="schema_inspector",
            output=prompt_string,
            output_type="string",
            error_message=None,
            execution_time_ms=execution_time,
            code_executed=f"inspect_schema(df, '{filename}')"
        )
        
    except Exception as e:
        execution_time = (time.time() - start_time) * 1000
        return ToolResult(
            success=False,
            tool_name="schema_inspector",
            output=None,
            output_type="error",
            error_message=f"Schema inspection failed: {str(e)}",
            execution_time_ms=execution_time,
            code_executed=None
        )
