"""
Agent Loop module for the AI Data Analyst Agent.
The master ReAct orchestrator that runs the Reason-Act-Observe loop.
Phase 4: Agent Brain
"""

import json
import time
import re
from dataclasses import dataclass, asdict
from typing import Any, List, Optional, Dict
import pandas as pd

from config import generate_content

from data_engine.utils import load_prompt
from agent.session_state import SessionState
from agent.intent_classifier import classify_intent, IntentType
from agent.planner import create_plan, ExecutionPlan, ExecutionStep
from tools.registry import get_tool_by_name, ToolResult

@dataclass
class AgentResponse:
    """
    Final response returned to the UI.
    """
    success: bool
    answer_text: str          # Main text response to show user
    narration: str            # Plain English insight
    chart: Any                # Plotly figure or None
    chart_reason: str         # Why this chart type was chosen
    result_dataframe: Any     # Pandas DataFrame or None
    tool_call_log: List[Dict] # Full audit trail for "Show my work"
    clarification_needed: bool
    clarification_question: Optional[str]
    confidence_level: str     # "High", "Medium", or "Low"
    error_message: Optional[str]
    execution_time_ms: float

def _build_tool_dict(session_state: SessionState) -> dict:
    """
    Builds a clean dict from SessionState for passing to tools.
    Explicitly includes active_dataframe to prevent Streamlit
    serialisation from dropping it.
    """
    return {
        "active_dataframe": session_state.active_dataframe,
        "active_filename": session_state.active_filename,
        "active_sheet": session_state.active_sheet,
        "all_dataframes": session_state.all_dataframes,
        "schema_profile": session_state.schema_profile,
        "schema_context": session_state.schema_context,
        "last_result_df": session_state.last_result_df,
        "last_chart": session_state.last_chart,
        "last_chart_reason": session_state.last_chart_reason,
        "last_narration": session_state.last_narration,
        "retrieved_memory": session_state.retrieved_memory,
        "anomaly_findings": session_state.anomaly_findings,
        "conversation_history": session_state.conversation_history,
        "session_id": session_state.session_id,
        "retry_count": session_state.retry_count,
        "max_retries": session_state.max_retries,
        "tool_call_log": session_state.tool_call_log,
    }

def _generate_sql_for_question(question: str, session_state: SessionState, prior_error: Optional[str] = None) -> str:
    """
    Calls Gemini to generate a DuckDB SQL query.
    """
    if not session_state.schema_context:
        print("[DEBUG] schema_context is empty — cannot ground SQL generation")
        return "CANNOT_SQL"

    try:
        system = load_prompt("agent_system")
        full_prompt = f"""
Dataset schema:
{session_state.schema_context}

Prior conversation context:
{session_state.retrieved_memory or 'None'}

User question: {question}

{f'Previous SQL failed with error: {prior_error}. Write a corrected query.' if prior_error else ''}

Write a single DuckDB SQL query to answer this question.
The table name is always 'df'.
IMPORTANT CHART RULES:
- Always SELECT both the grouping column AND the aggregated numeric value as a named alias.
- NEVER return only one column when the question involves comparison, ranking, or totals.
- CORRECT:   SELECT region, SUM(sales_usd) AS total_sales
             FROM df GROUP BY region
             ORDER BY total_sales DESC
- INCORRECT: SELECT region FROM df GROUP BY region
             ORDER BY SUM(sales_usd) DESC
- Always give aggregates a clean alias: total_sales, avg_revenue, count_orders, etc.
- Aliases must use underscores, no spaces.
Respond with ONLY the SQL query — no explanation, no markdown, no code fences.
If the question cannot be answered with SQL, respond with: CANNOT_SQL
"""
        sql = generate_content(full_prompt, system)
        sql = sql.strip().replace("```sql", "").replace("```", "").strip()
        print(f"[DEBUG] Generated SQL: {sql[:200]}")
        return sql
    except Exception as e:
        print(f"[DEBUG] Gemini API error in SQL generation: {type(e).__name__}: {str(e)}")
        raise

def _generate_python_for_question(question: str, session_state: SessionState, prior_error: Optional[str] = None) -> str:
    """
    Calls Gemini to generate Pandas Python code.
    """
    try:
        system = load_prompt("agent_system")
        full_prompt = f"""
Dataset schema:
{session_state.schema_context}

User question: {question}

{f'Previous Python failed with error: {prior_error}. Write corrected code.' if prior_error else ''}

Generate Python code using Pandas to answer this question.
- DataFrame is available as variable 'df'
- Must assign the final result to a variable named 'result'
- result should be a DataFrame or a scalar value
- Respond with ONLY Python code, no explanation, no markdown fences
- Available libraries: pandas as pd, numpy as np

If the question cannot be answered with Python, respond with: CANNOT_PYTHON
"""
        py_code = generate_content(full_prompt, system)
        py_code = py_code.strip().replace("```python", "").replace("```", "").strip()
        return py_code
    except Exception as e:
        print(f"[DEBUG] Gemini API error in Python generation: {type(e).__name__}: {str(e)}")
        return "CANNOT_PYTHON"

def _generate_chart_params(question: str, result_df: pd.DataFrame, session_state: SessionState) -> Dict:
    """
    Calls Gemini to choose chart type and columns.
    """
    try:
        system = load_prompt("chart_selector")
        user_message = f"""
Question: {question}
Result DataFrame columns: {list(result_df.columns)}
Result sample (3 rows): {result_df.head(3).to_string()}

Based on the chart selection rules, choose the best chart type and columns.

Respond ONLY with valid JSON:
{{
  "chart_type": "bar|line|scatter|histogram|grouped_bar",
  "x_column": "column_name",
  "y_column": "column_name",
  "chart_reason": "one sentence reason",
  "title": "Descriptive title"
}}
"""
        clean_json = generate_content(user_message, system).strip()
        
        # Strip markdown fences
        if clean_json.startswith("```"):
            clean_json = clean_json.split("\n", 1)[1].rsplit("\n", 1)[0]
        if clean_json.startswith("json"):
            clean_json = clean_json.replace("json", "", 1).strip()

        chart_params = json.loads(clean_json)

        # Validate columns exist in the DataFrame
        available_cols = list(result_df.columns)
        x_col = chart_params.get("x_column")
        y_col = chart_params.get("y_column")

        # If x_column is missing or not in df, use first column
        if not x_col or x_col not in available_cols:
            x_col = available_cols[0]

        # If y_column is missing or not in df:
        # Try to find any numeric column that isn't x_col
        if not y_col or y_col not in available_cols:
            numeric_cols = result_df.select_dtypes(
                include=["number"]).columns.tolist()
            non_x_numeric = [c for c in numeric_cols if c != x_col]
            if non_x_numeric:
                y_col = non_x_numeric[0]
            elif len(available_cols) > 1:
                # Fall back to second column whatever it is
                y_col = available_cols[1]
            else:
                # Only one column — cannot chart
                print(f"[DEBUG] Chart skipped: only one column available: {available_cols}")
                return None   # caller must handle None gracefully

        chart_params["x_column"] = x_col
        chart_params["y_column"] = y_col
        return chart_params
    except Exception:
        # Defaults
        cols = list(result_df.columns)
        return {
            "chart_type": "bar",
            "x_column": cols[0] if cols else "",
            "y_column": cols[1] if len(cols) > 1 else (cols[0] if cols else ""),
            "chart_reason": "Default bar chart selected due to error or ambiguity",
            "title": "Data Overview"
        }

def run_agent(question: str, session_state: SessionState) -> AgentResponse:
    """
    The master ReAct loop orchestrator.
    """
    agent_start = time.time()
    session_state.current_question = question
    
    try:
        # STEP 1: CLASSIFY INTENT
        intent_result = classify_intent(question, session_state)
        
        # Handle non-data intents immediately
        if intent_result.intent == IntentType.GREETING:
            elapsed = (time.time() - agent_start) * 1000
            return AgentResponse(
                success=True,
                answer_text=f"Hello! I'm ready to help you analyse {session_state.active_filename or 'your dataset'}. What would you like to know?",
                narration="", chart=None, chart_reason="",
                result_dataframe=None,
                tool_call_log=session_state.tool_call_log,
                clarification_needed=False,
                clarification_question=None,
                confidence_level="High",
                error_message=None,
                execution_time_ms=elapsed
            )
            
        if intent_result.intent == IntentType.OUT_OF_SCOPE:
            elapsed = (time.time() - agent_start) * 1000
            return AgentResponse(
                success=True,
                answer_text=f"That question appears to be outside the scope of the loaded dataset ({session_state.active_filename}). Try asking about the data directly — for example: 'What are the top 5 regions by sales?'",
                narration="", chart=None, chart_reason="",
                result_dataframe=None,
                tool_call_log=session_state.tool_call_log,
                clarification_needed=False,
                clarification_question=None,
                confidence_level="Low",
                error_message=None,
                execution_time_ms=elapsed
            )
            
        if intent_result.intent == IntentType.AMBIGUOUS or intent_result.requires_clarification:
            elapsed = (time.time() - agent_start) * 1000
            return AgentResponse(
                success=True,
                answer_text=intent_result.clarification_question or "I need more information to answer your question accurately.",
                narration="", chart=None, chart_reason="",
                result_dataframe=None,
                tool_call_log=session_state.tool_call_log,
                clarification_needed=True,
                clarification_question=intent_result.clarification_question,
                confidence_level="Low",
                error_message=None,
                execution_time_ms=elapsed
            )

        # STEP 2: CREATE PLAN
        plan = create_plan(intent_result, question, session_state)
        if plan.needs_clarification:
            elapsed = (time.time() - agent_start) * 1000
            return AgentResponse(
                success=True,
                answer_text=plan.clarification_question or "Could you clarify your request?",
                narration="", chart=None, chart_reason="",
                result_dataframe=None,
                tool_call_log=session_state.tool_call_log,
                clarification_needed=True,
                clarification_question=plan.clarification_question,
                confidence_level="Low",
                error_message=None,
                execution_time_ms=elapsed
            )

        # STEP 3: EXECUTE PLAN
        final_answer = ""
        retry_count = 0
        last_error = None
        sql_result: Optional[ToolResult] = None
        step_results: Dict[int, ToolResult] = {}
        
        # ReAct execution loop
        for step in plan.steps:
            # Skip logic
            if step.depends_on_step is not None:
                dep_result = step_results.get(step.depends_on_step)
                if (not dep_result or not dep_result.success) and step.is_required:
                    continue # Skip this step and move on

            # Tool Execution with specific logic per tool type
            if step.tool_name == "sql_executor":
                try:
                    while retry_count <= session_state.max_retries:
                        sql = _generate_sql_for_question(question, session_state, last_error)
                        
                        if sql == "CANNOT_SQL":
                            print(f"[DEBUG] SQL generation returned CANNOT_SQL. Schema context length: {len(session_state.schema_context)}")
                            print(f"[DEBUG] Active dataframe is None: {session_state.active_dataframe is None}")
                            # Try python fallback logic if SQL fails or is impossible
                            py_code = _generate_python_for_question(question, session_state, last_error)
                            if py_code != "CANNOT_PYTHON":
                                tool_dict = _build_tool_dict(session_state)
                                result = get_tool_by_name("python_executor")(py_code, question, tool_dict)
                                session_state.log_tool_call("python_executor", result.success, result.execution_time_ms, result.error_message)
                                if result.success:
                                    sql_result = result
                                    step_results[step.step_number] = result
                                    session_state.last_result_df = tool_dict.get("last_result_df")
                                    break
                            break # Cannot do either

                        tool_dict = _build_tool_dict(session_state)
                        result = get_tool_by_name("sql_executor")(sql, step.tool_params["explanation"], tool_dict)
                        session_state.log_tool_call("sql_executor", result.success, result.execution_time_ms, result.error_message)

                        if result.success:
                            sql_result = result
                            step_results[step.step_number] = result
                            session_state.last_result_df = tool_dict.get("last_result_df")
                            break
                        else:
                            last_error = result.error_message
                            retry_count += 1
                            if retry_count > session_state.max_retries:
                                # Final fallback to python
                                py_code = _generate_python_for_question(question, session_state, last_error)
                                tool_dict = _build_tool_dict(session_state)
                                py_result = get_tool_by_name("python_executor")(py_code, question, tool_dict)
                                session_state.log_tool_call("python_executor", py_result.success, py_result.execution_time_ms, py_result.error_message)
                                if py_result.success:
                                    sql_result = py_result
                                    step_results[step.step_number] = py_result
                                    session_state.last_result_df = tool_dict.get("last_result_df")
                                break
                except Exception as sql_err:
                    print(f"[SQL STEP ERROR] {type(sql_err).__name__}: {sql_err}")
                    session_state.log_tool_call("sql_executor", False, 0.0, str(sql_err))
            
            elif step.tool_name == "chart_generator":
                if sql_result and sql_result.success and isinstance(sql_result.output, pd.DataFrame):
                    chart_params = _generate_chart_params(question, sql_result.output, session_state)
                    if chart_params is None:
                        print("[DEBUG] Chart skipped: insufficient columns in result DataFrame")
                        result = ToolResult(
                            success=False,
                            tool_name="chart_generator",
                            output=None,
                            output_type="error",
                            error_message="Skipped: result has only one column",
                            execution_time_ms=0.0,
                            code_executed=None
                        )
                        session_state.log_tool_call("chart_generator", result.success, result.execution_time_ms, result.error_message)
                        step_results[step.step_number] = result
                        continue
                    tool_dict = _build_tool_dict(session_state)
                    result = get_tool_by_name("chart_generator")(
                        chart_params["chart_type"],
                        chart_params["x_column"],
                        chart_params["y_column"],
                        chart_params.get("title", question[:60]),
                        chart_params["chart_reason"],
                        tool_dict
                    )
                    session_state.log_tool_call("chart_generator", result.success, result.execution_time_ms, result.error_message)
                    session_state.last_chart = tool_dict.get("last_chart")
                    session_state.last_chart_reason = tool_dict.get("last_chart_reason", "")
                    step_results[step.step_number] = result
                else:
                    # Skip chart silently or log it
                    pass

            elif step.tool_name == "insight_narrator":
                result_summary = ""
                if sql_result and sql_result.success:
                    if isinstance(sql_result.output, pd.DataFrame):
                        result_summary = sql_result.output.to_string(max_rows=10)
                    else:
                        result_summary = str(sql_result.output)
                
                tool_dict = _build_tool_dict(session_state)
                result = get_tool_by_name("insight_narrator")(result_summary, question, tool_dict)
                session_state.log_tool_call("insight_narrator", result.success, result.execution_time_ms, result.error_message)
                session_state.last_narration = tool_dict.get("last_narration", "")
                step_results[step.step_number] = result
                if result.success:
                    final_answer = result.output

            elif step.tool_name == "memory_retriever":
                tool_dict = _build_tool_dict(session_state)
                result = get_tool_by_name("memory_retriever")(step.tool_params["query"], tool_dict)
                session_state.retrieved_memory = tool_dict.get("retrieved_memory", "")
                session_state.log_tool_call("memory_retriever", result.success, result.execution_time_ms, result.error_message)
                step_results[step.step_number] = result
                
            elif step.tool_name == "report_builder":
                tool_dict = _build_tool_dict(session_state)
                result = get_tool_by_name("report_builder")(tool_dict)
                session_state.log_tool_call("report_builder", result.success, result.execution_time_ms, result.error_message)
                step_results[step.step_number] = result
                if result.success:
                    final_answer = f"Report generated: {result.output}"

        # STEP 4: DETERMINE CONFIDENCE
        confidence_level = "High"
        if retry_count > 0:
            confidence_level = "Medium"
        if retry_count >= session_state.max_retries:
            confidence_level = "Low"
        if intent_result.confidence < 0.6:
            confidence_level = "Medium"
        if sql_result is None or not sql_result.success:
            confidence_level = "Low"
        if session_state.schema_profile and session_state.schema_profile.data_quality_score < 60:
            confidence_level = "Medium"

        if sql_result is None:
            df_status = (
                "Yes" if session_state.active_dataframe is not None
                else "**No — this is the problem**"
            )
            schema_status = (
                "Loaded" if session_state.schema_context
                else "**Empty — this is the problem**"
            )
            final_answer = (
                "I was unable to query your data. This is usually "
                "caused by one of these issues:\n\n"
                "1. **Schema not loaded** — try re-uploading "
                "your file\n"
                "2. **Column name mismatch** — check the Column "
                "Details panel in the sidebar\n"
                "3. **API issue** — check your terminal for "
                "[DEBUG] messages\n\n"
                f"Dataset loaded: {df_status}\n"
                f"Schema context: {schema_status}"
            )

        # STEP 5: SAVE TO MEMORY
        if final_answer and sql_result and sql_result.success:
            from tools.memory_retriever import store_qa_pair
            store_qa_pair(
                question=question,
                answer=final_answer[:500],
                session_id=session_state.session_id
            )
            session_state.add_to_history(
                question=question,
                answer=final_answer,
                narration=session_state.last_narration,
                chart_included=session_state.last_chart is not None
            )

        # STEP 6: BUILD RESPONSE
        elapsed = (time.time() - agent_start) * 1000
        
        if not final_answer:
            if retry_count >= session_state.max_retries:
                final_answer = f"I wasn't able to compute this accurately after {session_state.max_retries} attempts. Could you rephrase the question or be more specific about which columns to use?"
            else:
                final_answer = "I processed your request but could not generate a text summary. Please check the chart and data table above."

        return AgentResponse(
            success=True,
            answer_text=final_answer,
            narration=session_state.last_narration,
            chart=session_state.last_chart,
            chart_reason=session_state.last_chart_reason,
            result_dataframe=session_state.last_result_df,
            tool_call_log=session_state.tool_call_log,
            clarification_needed=False,
            clarification_question=None,
            confidence_level=confidence_level,
            error_message=None,
            execution_time_ms=elapsed
        )

    except Exception as e:
        elapsed = (time.time() - agent_start) * 1000
        error_str = str(e)

        # Groq rate limit detection (works without importing 
        # a Groq-specific exception class)
        if "rate_limit" in error_str.lower() or "429" in error_str:
            return AgentResponse(
                success=False,
                answer_text=(
                    "⚠️ **API Rate Limit Reached**\n\n"
                    "The Groq free tier quota has been temporarily "
                    "exhausted.\n\n"
                    "**Per-minute limit:** Wait 60 seconds and "
                    "try again.\n\n"
                    "**Daily limit:** Wait until tomorrow, or "
                    "upgrade your plan at https://console.groq.com"
                ),
                narration="",
                chart=None,
                chart_reason="",
                result_dataframe=None,
                tool_call_log=session_state.tool_call_log,
                clarification_needed=False,
                clarification_question=None,
                confidence_level="Low",
                error_message=f"RateLimit: {error_str[:200]}",
                execution_time_ms=elapsed,
            )

        return AgentResponse(
            success=False,
            answer_text="An unexpected error occurred during analysis.",
            narration="",
            chart=None,
            chart_reason="",
            result_dataframe=None,
            tool_call_log=session_state.tool_call_log,
            clarification_needed=False,
            clarification_question=None,
            confidence_level="Low",
            error_message=error_str,
            execution_time_ms=elapsed
        )    

    except Exception as e:
        import traceback
        elapsed = (time.time() - agent_start) * 1000
        full_error = traceback.format_exc()
        print(f"[AGENT ERROR] {type(e).__name__}: {str(e)}")
        print(f"[AGENT TRACEBACK]\n{full_error}")

        error_detail = str(e)
        answer = (
            f"An error occurred during analysis.\n\n"
            f"**Error type:** `{type(e).__name__}`\n"
            f"**Detail:** {error_detail[:300]}\n\n"
            f"If this is a file system or database error, "
            f"try re-uploading your dataset."
        )

        return AgentResponse(
            success=False,
            answer_text=answer,
            narration="",
            chart=None,
            chart_reason="",
            result_dataframe=None,
            tool_call_log=session_state.tool_call_log,
            clarification_needed=False,
            clarification_question=None,
            confidence_level="Low",
            error_message=error_detail,
            execution_time_ms=elapsed
        )
