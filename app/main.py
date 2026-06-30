"""
Main entry point for the AI Data Analyst Agent Streamlit application.
Initialises state, handles file uploads, and coordinates the agent loop.
Phase 5: UI Implementation
"""

import streamlit as st
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# st.set_page_config MUST be the first Streamlit command
st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

from config import settings, validate_config
from data_engine.file_handler import ingest_file, save_uploaded_file
from data_engine.schema_inspector import inspect_schema, schema_to_prompt_string
from agent.session_state import SessionState
from agent.loop import run_agent
from tools.anomaly_detector import run as detect_anomalies
from app.components import (
    render_header, render_file_uploader, 
    render_chat_message, render_typing_indicator,
    render_sidebar_controls, render_proactive_insights
)
from agent.insights_engine import generate_proactive_insights
from data_engine.utils import sanitise_filename

def main():
    # 1. VALIDATE CONFIG
    try:
        validate_config()
    except ValueError as e:
        st.error(f"⚠️ Configuration Error: {str(e)}")
        st.markdown("""
        Please ensure you have a `.env` file with a valid `GEMINI_API_KEY`.
        Refer to `.env.example` for the required format.
        """)
        st.stop()

    # 2. INITIALISE SESSION STATE
    if "agent_session" not in st.session_state:
        st.session_state.agent_session = SessionState()

    # Restore DataFrame if Streamlit dropped it from session
    if ("agent_session" in st.session_state and
            st.session_state.agent_session.active_dataframe is None and
            "_active_df_backup" in st.session_state):
        st.session_state.agent_session.active_dataframe = (
            st.session_state["_active_df_backup"]
        )
        
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    if "file_loaded" not in st.session_state:
        st.session_state.file_loaded = False

    if "proactive_insights" not in st.session_state:
        st.session_state.proactive_insights = []

    session = st.session_state.agent_session

    # 3. RENDER HEADER
    render_header()

    # 4. FILE UPLOAD SECTION
    col_upload, col_info = st.columns([2, 1])
    
    with col_upload:
        uploaded_file = render_file_uploader()
        
    if uploaded_file is not None:
        # Compare sanitized names (saved on disk) to avoid re-ingesting on every rerun
        is_new_file = sanitise_filename(uploaded_file.name) != session.active_filename
        if is_new_file:
            with st.spinner("Processing your dataset..."):
                # Save file
                file_path = save_uploaded_file(uploaded_file.read(), uploaded_file.name)
                
                # Ingest
                ingestion = ingest_file(file_path)
                
                if not ingestion.success:
                    st.error(f"Failed to load file: {ingestion.error_message}")
                else:
                    # Update session
                    first_sheet = list(ingestion.dataframes.keys())[0]
                    session.active_dataframe = ingestion.dataframes[first_sheet]
                    session.active_filename = ingestion.filename
                    session.active_sheet = first_sheet
                    session.all_dataframes = ingestion.dataframes
                    
                    # Schema inspection
                    session.schema_profile = inspect_schema(
                        session.active_dataframe,
                        session.active_filename,
                        first_sheet
                    )
                    session.schema_context = schema_to_prompt_string(session.schema_profile)
                    
                    # Proactive anomaly detection
                    anomaly_result = detect_anomalies(session.__dict__)
                    session.anomaly_findings = anomaly_result.output if anomaly_result.success else []

                    # Generate proactive insights
                    with st.spinner("Generating automatic insights..."):
                        insights = generate_proactive_insights(
                            session.active_dataframe,
                            session.active_filename,
                            session.schema_context
                        )
                    st.session_state.proactive_insights = insights
                    
                    st.session_state.file_loaded = True
                    
                    # Welcome Message
                    quality = session.schema_profile.data_quality_score
                    quality_label = "🟢 Good" if quality >= 80 else "🟡 Fair" if quality >= 60 else "🔴 Poor"
                    
                    welcome = (
                        f"I've loaded **{session.active_filename}** "
                        f"({session.active_dataframe.shape[0]:,} rows × "
                        f"{session.active_dataframe.shape[1]} columns). "
                        f"Data quality: {quality_label} ({quality:.0f}/100).\n\n"
                        f"I automatically analysed your data and found "
                        f"**{len(insights)} insights** — see below.\n\n"
                        f"**What would you like to know about this data?**"
                    )
                    
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": welcome,
                        "chart": None,
                        "result_df": None,
                        "chart_reason": "",
                        "confidence": "High",
                        "tool_log": []
                    })
                    
                    st.rerun()

    # 5. SIDEBAR
    if st.session_state.get("file_loaded"):
        render_sidebar_controls(session)
    else:
        st.sidebar.title("📂 Dataset Controls")
        st.sidebar.info("Upload a dataset to see controls here.")

    # Proactive insights panel
    if st.session_state.get("proactive_insights"):
        render_proactive_insights(
            st.session_state["proactive_insights"],
            session
        )

    # 6. CHAT HISTORY DISPLAY
    chat_container = st.container()
    with chat_container:
        for idx, message in enumerate(st.session_state.chat_history):
            render_chat_message(
                role=message["role"],
                content=message["content"],
                message_index=idx,
                chart=message.get("chart"),
                result_df=message.get("result_df"),
                chart_reason=message.get("chart_reason", ""),
                confidence=message.get("confidence", "High"),
                tool_log=message.get("tool_log", [])
            )

    # 7. CHAT INPUT
    user_input = st.chat_input(
        placeholder=(
            "Ask a question about your data... e.g. 'Which region had the highest sales in Q3?'"
            if st.session_state.file_loaded 
            else "Upload a dataset above to start asking questions"
        ),
        disabled=not st.session_state.file_loaded
    )

    # Handle questions triggered by insight buttons
    pending = st.session_state.pop("pending_question", None)
    if pending:
        user_input = pending

    # 8. HANDLE USER INPUT
    if user_input:
        # Add to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input,
            "chart": None, "result_df": None,
            "chart_reason": "", "confidence": "High",
            "tool_log": []
        })
        
        # Show message immediately
        st.rerun()

    # This part handles the actual agent execution if the last message was a user message
    if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
        last_user_message = st.session_state.chat_history[-1]["content"]
        
        # Show typing indicator
        typing = render_typing_indicator()
        
        # Run agent
        with st.spinner("Analyzing..."):
            response = run_agent(last_user_message, session)
        
        # Clear typing indicator
        typing.empty()
        
        # Append answer
        answer_content = response.clarification_question if response.clarification_needed else response.answer_text
        
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer_content,
            "chart": response.chart,
            "result_df": response.result_dataframe,
            "chart_reason": response.chart_reason,
            "confidence": response.confidence_level,
            "tool_log": response.tool_call_log
        })
        
        # Sync session back
        st.session_state.agent_session = session

        # Force DataFrame reference to survive Streamlit rerun
        if session.active_dataframe is not None:
            st.session_state["_active_df_backup"] = session.active_dataframe

        st.rerun()

    # 9. EMPTY STATE
    if not st.session_state.file_loaded and not st.session_state.chat_history:
        st.markdown("""
            <div style="text-align:center; padding:3rem 0; color:#9CA3AF;">
              <div style="font-size:3rem;">📊</div>
              <h3 style="color:#6B7280; font-weight:500;">
                Upload a dataset to get started
              </h3>
              <p style="font-size:0.9rem;">
                Supported: CSV, Excel, PDF · Max 50MB
              </p>
              <div style="font-size:0.85rem; margin-top:1rem; max-width:400px; margin-left:auto; margin-right:auto;">
                Then ask questions like:<br>
                <em>"Which product had the highest revenue?"</em><br>
                <em>"Show me monthly sales trends"</em><br>
                <em>"Are there any outliers in the data?"</em>
              </div>
            </div>
          """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
