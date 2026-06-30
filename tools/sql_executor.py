"""
SQL Executor Tool for the AI Data Analyst Agent.
Executes DuckDB SQL queries against DataFrames.
Phase 3: Tool Implementation
"""

import time
import duckdb
import pandas as pd
from tools.registry import ToolResult

def run(query: str, explanation: str, session_state: dict) -> ToolResult:
    """
    Executes a SQL query on the active DataFrame using DuckDB.
    """
    start_time = time.time()
    
    df = session_state.get("active_dataframe")
    
    if df is None:
        execution_time = (time.time() - start_time) * 1000
        return ToolResult(
            success=False,
            tool_name="sql_executor",
            output=None,
            output_type="error",
            error_message="No active dataset found in session.",
            execution_time_ms=execution_time,
            code_executed=query
        )
    
    try:
        # Connect to an in-memory database
        con = duckdb.connect()
        # Register the dataframe so duckdb can see it
        con.register("df", df)
        
        # Execute query
        result_df = con.execute(query).df()
        
        # Store result in session for potential graphing/narration
        session_state["last_result_df"] = result_df
        
        if result_df.empty:
            warning = "Query returned no results."
        else:
            warning = None
            
        execution_time = (time.time() - start_time) * 1000
        
        return ToolResult(
            success=True,
            tool_name="sql_executor",
            output=result_df,
            output_type="dataframe",
            error_message=warning,
            execution_time_ms=execution_time,
            code_executed=query
        )
        
    except Exception as e:
        execution_time = (time.time() - start_time) * 1000
        return ToolResult(
            success=False,
            tool_name="sql_executor",
            output=None,
            output_type="error",
            error_message=f"SQL Error: {type(e).__name__}: {str(e)}",
            execution_time_ms=execution_time,
            code_executed=query
        )
    finally:
        try:
            con.close()
        except:
            pass
