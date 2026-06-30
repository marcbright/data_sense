"""
Python Executor Tool for the AI Data Analyst Agent.
Executes sandboxed Python/Pandas code for data analysis.
Phase 3: Tool Implementation
"""

import time
import pandas as pd
import numpy as np
import sys
import io
import threading
from typing import Tuple
from tools.registry import ToolResult

BLOCKED_KEYWORDS = [
    "import os",
    "import sys",
    "import subprocess",
    "import socket",
    "import requests",
    "import urllib",
    "__import__",
    "__builtins__",
    "eval(",
    "exec(",
    "open(",
    "os.remove",
    "os.unlink",
    "os.rmdir",
    "shutil",
    "subprocess",
    "sys.exit",
]

def is_safe_code(code: str) -> Tuple[bool, str]:
    """
    Checks code against blocked keywords.
    """
    code_lower = code.lower()
    for keyword in BLOCKED_KEYWORDS:
        if keyword.lower() in code_lower:
            return False, f"Potentially unsafe keyword detected: '{keyword}'"
    return True, ""

def run(code: str, explanation: str, session_state: dict) -> ToolResult:
    """
    Executes sandboxed Python code against the active DataFrame.
    """
    start_time = time.time()
    df = session_state.get("active_dataframe")
    
    if df is None:
        execution_time = (time.time() - start_time) * 1000
        return ToolResult(
            success=False,
            tool_name="python_executor",
            output=None,
            output_type="error",
            error_message="No active dataset found in session.",
            execution_time_ms=execution_time,
            code_executed=code
        )
        
    # Security check
    is_safe, reason = is_safe_code(code)
    if not is_safe:
        execution_time = (time.time() - start_time) * 1000
        return ToolResult(
            success=False,
            tool_name="python_executor",
            output=None,
            output_type="error",
            error_message=f"Security Block: {reason}",
            execution_time_ms=execution_time,
            code_executed=code
        )

    # Set up execution namespace
    # We pass a copy of df to prevent accidental corruption of original state
    namespace = {
        "df": df.copy(),
        "pd": pd,
        "np": np,
        "result": None
    }
    
    stdout_capture = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = stdout_capture
    
    result_container = []
    error_container = []
    
    def target():
        try:
            exec(code, namespace)
            result_container.append(namespace.get("result"))
        except Exception as e:
            error_container.append(e)
            
    # Execute with timeout
    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout=30)
    
    sys.stdout = old_stdout
    captured_output = stdout_capture.getvalue()
    
    execution_time = (time.time() - start_time) * 1000
    
    if thread.is_alive():
        return ToolResult(
            success=False,
            tool_name="python_executor",
            output=None,
            output_type="error",
            error_message="Code execution timed out after 30 seconds.",
            execution_time_ms=execution_time,
            code_executed=code
        )

    if error_container:
        e = error_container[0]
        # Try to get line info if possible
        import traceback
        tb = traceback.format_exc()
        return ToolResult(
            success=False,
            tool_name="python_executor",
            output=None,
            output_type="error",
            error_message=f"Python Error: {type(e).__name__}: {str(e)}\n{tb}",
            execution_time_ms=execution_time,
            code_executed=code
        )
        
    final_result = result_container[0] if result_container else None
    
    # Store result if it's a dataframe for downstream tools
    if isinstance(final_result, pd.DataFrame):
        session_state["last_result_df"] = final_result
        output_type = "dataframe"
    else:
        output_type = "string"
        
    # If code didn't set a result but printed something, use print output
    if final_result is None and captured_output:
        final_result = captured_output
        output_type = "string"

    return ToolResult(
        success=True,
        tool_name="python_executor",
        output=final_result,
        output_type=output_type,
        error_message=None,
        execution_time_ms=execution_time,
        code_executed=code
    )
