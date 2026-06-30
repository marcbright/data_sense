"""
Chart Generator Tool for the AI Data Analyst Agent.
Generates Plotly visualizations based on analysis results.
Phase 3: Tool Implementation
"""

import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from tools.registry import ToolResult

CHART_COLOUR_PALETTE = [
    "#4F46E5", "#7C3AED", "#2563EB", "#059669", 
    "#D97706", "#DC2626", "#0891B2"
]

def run(chart_type: str, x_column: str, y_column: str,
          title: str, chart_reason: str,
          session_state: dict) -> ToolResult:
    """
    Generates a Plotly chart.
    """
    start_time = time.time()
    
    # Try last_result_df, fallback to active_dataframe
    df = session_state.get("last_result_df")
    if df is None:
        df = session_state.get("active_dataframe")
        
    if df is None:
        execution_time = (time.time() - start_time) * 1000
        return ToolResult(
            success=False,
            tool_name="chart_generator",
            output=None,
            output_type="error",
            error_message="No data found to visualize. Run a query first.",
            execution_time_ms=execution_time,
            code_executed=None
        )

    # Validate columns
    available_cols = df.columns.tolist()
    missing = []
    if x_column not in available_cols:
        missing.append(x_column)
    if y_column not in available_cols and chart_type != "histogram":
        missing.append(y_column)
        
    if missing:
        execution_time = (time.time() - start_time) * 1000
        return ToolResult(
            success=False,
            tool_name="chart_generator",
            output=None,
            output_type="error",
            error_message=f"Columns not found: {missing}. Available: {available_cols}",
            execution_time_ms=execution_time,
            code_executed=None
        )

    try:
        fig = None
        chart_type = chart_type.lower()
        
        if chart_type == "bar":
            fig = px.bar(
                df, x=x_column, y=y_column, title=title,
                color_discrete_sequence=CHART_COLOUR_PALETTE
            )
        elif chart_type == "line":
            fig = px.line(
                df, x=x_column, y=y_column, title=title,
                color_discrete_sequence=CHART_COLOUR_PALETTE
            )
        elif chart_type == "scatter":
            fig = px.scatter(
                df, x=x_column, y=y_column, title=title,
                color_discrete_sequence=CHART_COLOUR_PALETTE
            )
        elif chart_type == "histogram":
            fig = px.histogram(
                df, x=x_column, title=title,
                color_discrete_sequence=CHART_COLOUR_PALETTE
            )
        elif chart_type == "grouped_bar":
            fig = px.bar(
                df, x=x_column, y=y_column, title=title,
                barmode="group",
                color_discrete_sequence=CHART_COLOUR_PALETTE
            )
        else:
            # Fallback to bar chart
            fig = px.bar(
                df, x=x_column, y=y_column, title=title,
                color_discrete_sequence=CHART_COLOUR_PALETTE
            )
            title = f"{title} (Defaulting to Bar Chart)"

        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=16, family="Inter, sans-serif"),
            ),
            template="plotly_white",
            margin=dict(l=40, r=40, t=60, b=40),
            showlegend=(chart_type == "grouped_bar"),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )

        try:
            # Test that figure serialises without error
            fig.to_json()
        except Exception as e:
            return ToolResult(
                success=False,
                tool_name="chart_generator",
                output=None,
                output_type="error",
                error_message=f"Visualization failed during validation: {str(e)}",
                execution_time_ms=(time.time()-start_time)*1000,
                code_executed=f"{chart_type}: {x_column} vs {y_column}"
            )

        # Store metadata in session
        session_state["last_chart"] = fig
        session_state["last_chart_reason"] = chart_reason
        
        execution_time = (time.time() - start_time) * 1000
        
        return ToolResult(
            success=True,
            tool_name="chart_generator",
            output=fig,
            output_type="chart",
            error_message=None,
            execution_time_ms=execution_time,
            code_executed=f"{chart_type} chart: {x_column} vs {y_column}"
        )

    except Exception as e:
        execution_time = (time.time() - start_time) * 1000
        return ToolResult(
            success=False,
            tool_name="chart_generator",
            output=None,
            output_type="error",
            error_message=f"Visualization failed: {str(e)}",
            execution_time_ms=execution_time,
            code_executed=None
        )
