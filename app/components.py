"""
UI Components for the AI Data Analyst Agent.
Handles reusable rendering logic for the Streamlit interface.
Phase 5: UI Implementation
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from agent.session_state import SessionState
from config import settings
from data_engine.schema_inspector import inspect_schema, schema_to_prompt_string

def render_header() -> None:
    """
    Renders the app header at the top of the page.
    """
    st.markdown('''
      <div style="text-align:center; padding: 1rem 0 0.5rem;">
        <h1 style="font-size:2rem; font-weight:700; 
                   color:#4F46E5; margin-bottom:0.2rem;">
          AI Data Analyst
        </h1>
        <p style="color:#6B7280; font-size:1rem; margin:0;">
          Chat with your data in plain English
        </p>
      </div>
      <hr style="border:none; border-top:2px solid #4F46E5; 
                 margin: 0.5rem 0 1.5rem;">
    ''', unsafe_allow_html=True)

def render_file_uploader() -> any:
    """
    Renders the file upload widget.
    """
    uploaded_file = st.file_uploader(
        label="Upload your dataset",
        type=["csv", "xlsx", "xls", "pdf"],
        help="Supported formats: CSV, Excel (.xlsx/.xls), PDF. Max 50MB.",
        key="file_uploader"
    )
    
    if not st.session_state.get('file_loaded', False):
        st.info("📊 Upload a dataset to begin. You can then ask questions about it in plain English.")
        
    return uploaded_file

def render_dataset_summary(session_state: SessionState) -> None:
    """
    Renders a compact summary card of the loaded dataset in the sidebar.
    """
    with st.sidebar.expander("Dataset Overview", expanded=True):
        st.markdown(f"**Filename:** {session_state.active_filename}")
        
        if session_state.active_dataframe is not None:
            rows, cols = session_state.active_dataframe.shape
            st.markdown(f"**Rows:** {rows:,}")
            st.markdown(f"**Columns:** {cols}")
            
        if session_state.schema_profile:
            score = session_state.schema_profile.data_quality_score
            indicator = "🔴" if score < 60 else "🟡" if score < 80 else "🟢"
            st.markdown(f"**Quality Score:** {indicator} {score:.0f}/100")
            
        # Sheet selector for multi-sheet Excel
        if len(session_state.all_dataframes) > 1:
            sheets = list(session_state.all_dataframes.keys())
            current_index = sheets.index(session_state.active_sheet) if session_state.active_sheet in sheets else 0
            
            new_sheet = st.selectbox(
                "Switch Sheet",
                options=sheets,
                index=current_index
            )
            
            if new_sheet != session_state.active_sheet:
                session_state.active_sheet = new_sheet
                session_state.active_dataframe = session_state.all_dataframes[new_sheet]
                # Re-run schema inspection
                session_state.schema_profile = inspect_schema(
                    session_state.active_dataframe,
                    session_state.active_filename,
                    new_sheet
                )
                session_state.schema_context = schema_to_prompt_string(session_state.schema_profile)
                st.rerun()

        if session_state.active_dataframe is not None:
            st.markdown("**Preview (First 5 rows):**")
            st.dataframe(session_state.active_dataframe.head(5), width="stretch")

def render_schema_panel(session_state: SessionState) -> None:
    """
    Renders column details and data quality warnings.
    """
    with st.expander("Column Details", expanded=False):
        st.code(session_state.schema_context, language=None)
        
    if session_state.anomaly_findings:
        with st.expander("⚠️ Data Quality Warnings", expanded=False):
            for finding in session_state.anomaly_findings:
                st.warning(finding)


def render_proactive_insights(insights: list, session_state) -> None:
    """
    Renders the proactive insights panel below the file upload section.
    """
    if not insights:
        return

    with st.expander("💡 Auto-detected Insights", expanded=True):
        st.markdown(
            f"*I analysed your dataset and found {len(insights)} insights before you asked anything:*"
        )

        for insight in insights:
            icon = "🔴" if insight.severity == "critical" else "🟡" if insight.severity == "warning" else "🟢"
            colour = {
                "trend": "#4F46E5",
                "top_bottom": "#059669",
                "correlation": "#7C3AED",
                "distribution": "#2563EB",
                "data_quality": "#D97706",
                "outlier": "#DC2626"
            }.get(insight.category, "#6B7280")

            st.markdown(f'''
                <div style="border-left: 4px solid {colour};
                            padding: 0.75rem 1rem;
                            margin: 0.5rem 0;
                            border-radius: 0 8px 8px 0;
                            background: var(--secondary-bg);">
                  <div style="font-weight:600; font-size:0.9rem;
                              margin-bottom:0.25rem;">
                    {icon} {insight.title}
                  </div>
                  <div style="font-size:0.85rem; 
                              color: #6B7280;
                              margin-bottom:0.4rem;">
                    {insight.finding}
                  </div>
                  <div style="font-size:0.75rem; 
                              color: #9CA3AF;">
                    📊 {insight.supporting_stat} · 
                    Columns: {", ".join(insight.affected_columns) or "general"}
                  </div>
                </div>
              ''', unsafe_allow_html=True)

            if insight.chart_suggestion is not None:
                if st.button(
                    f"📈 Visualise this insight",
                    key=f"vis_{insight.insight_id}"
                ):
                    suggestion = insight.chart_suggestion
                    prompt = (
                        f"Show me a {suggestion.get('chart_type','bar')} "
                        f"chart of {suggestion.get('title', insight.title)}"
                    )
                    st.session_state["pending_question"] = prompt
                    st.rerun()


def render_chat_message(role: str, content: str,
                         message_index: int,
                         chart=None,
                         result_df=None,
                         chart_reason: str = "",
                         confidence: str = "High",
                         tool_log: list = None) -> None:
    """
    Renders a single chat message bubble.
    """
    with st.chat_message(role):
        if role == "assistant":
            # Confidence badge
            color = "#10B981" if confidence == "High" else "#F59E0B" if confidence == "Medium" else "#EF4444"
            st.markdown(f'<span style="background:{color}; color:white; padding:2px 10px; border-radius:12px; font-size:0.75rem; font-weight:600;">{confidence} confidence</span>', unsafe_allow_html=True)
            
        st.markdown(content)
        
        if result_df is not None and not result_df.empty:
            st.dataframe(
                result_df,
                width="stretch",
                height=250,
                key=f"df_{message_index}"
            )
            
        if chart is not None:
            st.plotly_chart(
                chart,
                width="stretch",
                key=f"chart_{message_index}"
            )
            if chart_reason:
                st.caption(f"📊 Chart choice: {chart_reason}")
                
        if role == "assistant" and tool_log:
            with st.expander(f"🔍 Show my work {message_index}", expanded=False):
                for entry in tool_log:
                    icon = "✅" if entry.get("success") else "❌"
                    st.markdown(
                        f"{icon} **{entry.get('tool')}** — {entry.get('execution_time_ms', 0):.0f}ms"
                        + (f" — ⚠️ {entry.get('error')}" if entry.get('error') else "")
                    )

def render_typing_indicator() -> st.empty:
    """
    Returns a placeholder showing a typing indicator.
    """
    placeholder = st.empty()
    placeholder.markdown('''
      <div style="display:flex; align-items:center; gap:8px; color:#6B7280; font-size:0.9rem; padding:8px 0;">
        <span>Agent is thinking</span>
        <span style="animation: blink 1s infinite;">...</span>
      </div>
      <style>
        @keyframes blink {
          0%,100%{opacity:1} 50%{opacity:0.3}
        }
      </style>
    ''', unsafe_allow_html=True)
    return placeholder

def render_export_button(session_state) -> None:
    """
    Renders the full export UI in the sidebar.
    Handles the download flow entirely within this function.
    """
    st.sidebar.markdown("### 📄 Export Report")
    
    has_content = (
        len(session_state.conversation_history) > 0 or
        len(st.session_state.get("proactive_insights", [])) > 0
    )
    
    if not has_content:
        st.sidebar.caption(
            "Ask at least one question to enable export.")
        return
    
    # Summary of what will be in the report
    turns = len(session_state.conversation_history)
    insights = len(st.session_state.get("proactive_insights", []))
    
    st.sidebar.markdown(
        f"Report will include:\n"
        f"- **{turns}** analysis questions\n"
        f"- **{insights}** auto-detected insights\n"
        f"- Dataset overview + quality notes\n"
        f"- Charts (if generated)"
    )
    
    if st.sidebar.button(
        "📥 Generate PDF Report",
        use_container_width=True,
        type="primary",
        key="export_btn"
    ):
        with st.sidebar.spinner("Building your report..."):
            # Build tool dict
            tool_dict = {
                "active_filename": session_state.active_filename,
                "schema_context": session_state.schema_context,
                "schema_profile": session_state.schema_profile,
                "conversation_history": session_state.conversation_history,
                "anomaly_findings": session_state.anomaly_findings,
                "last_chart": session_state.last_chart,
                "last_chart_reason": session_state.last_chart_reason,
                "proactive_insights": st.session_state.get("proactive_insights", []),
                "session_id": session_state.session_id,
            }
            
            from tools.report_builder import run as build
            result = build(tool_dict)
        
        if result.success:
            # Read PDF bytes for download
            with open(result.output, "rb") as f:
                pdf_bytes = f.read()
            
            st.sidebar.success("✅ Report ready!")
            
            st.sidebar.download_button(
                label="⬇️ Download PDF",
                data=pdf_bytes,
                file_name=Path(result.output).name,
                mime="application/pdf",
                use_container_width=True,
                key="download_pdf_btn"
            )
        else:
            st.sidebar.error(
                f"Export failed: {result.error_message}")

def render_sidebar_controls(session_state: SessionState) -> None:
    """
    Renders all sidebar elements.
    """
    render_dataset_summary(session_state)
    render_schema_panel(session_state)
    st.sidebar.divider()
    
    render_export_button(session_state)
    
    st.sidebar.divider()
    st.sidebar.caption(f"Session: {session_state.session_id}\nModel: {settings.model_name}")
