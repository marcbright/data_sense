"""
Insight Narrator Tool for the AI Data Analyst Agent.
Generates human-readable summaries of data analysis using Gemini.
Phase 3: Tool Implementation
"""

import time
from config import generate_content
from data_engine.utils import load_prompt
from tools.registry import ToolResult

def run(result_summary: str, question_asked: str,
          session_state: dict) -> ToolResult:
    """
    Narrates analysis results.
    """
    start_time = time.time()
    
    try:
        system = load_prompt("insight_narrator")
        
        schema_context = session_state.get("schema_context", "N/A")
        
        user_message = (
            f"Question asked: {question_asked}\n"
            f"Analysis result: {result_summary}\n"
            f"Schema context: {schema_context}\n"
            f"Generate a plain English insight based on the provided instructions."
        )
        
        narration = generate_content(user_message, system).strip()
        
        # Store in session
        session_state["last_narration"] = narration
        
        execution_time = (time.time() - start_time) * 1000
        
        return ToolResult(
            success=True,
            tool_name="insight_narrator",
            output=narration,
            output_type="string",
            error_message=None,
            execution_time_ms=execution_time,
            code_executed=None
        )
        
    except Exception as e:
        execution_time = (time.time() - start_time) * 1000
        return ToolResult(
            success=False,
            tool_name="insight_narrator",
            output=None,
            output_type="error",
            error_message=f"Narration failed: {str(e)}",
            execution_time_ms=execution_time,
            code_executed=None
        )
