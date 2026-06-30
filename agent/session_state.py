"""
Session State module for the AI Data Analyst Agent.
Handles the shared state container passed through the agent pipeline.
Phase 4: Agent Brain
"""

from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional
import pandas as pd
import uuid
from datetime import datetime

@dataclass
class SessionState:
    """
    Working memory for a single conversation session.
    """
    # Session identity
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Dataset
    active_dataframe: Optional[pd.DataFrame] = None
    active_filename: str = ""
    active_sheet: str = "main"
    all_dataframes: Dict[str, pd.DataFrame] = field(default_factory=dict)

    # Schema
    schema_profile: Any = None       # SchemaProfile object
    schema_context: str = ""         # Formatted string for prompts

    # Analysis results
    last_result_df: Optional[pd.DataFrame] = None
    last_chart: Any = None           # Plotly figure
    last_chart_reason: str = ""
    last_narration: str = ""

    # Memory
    retrieved_memory: str = ""
    anomaly_findings: List[str] = field(default_factory=list)

    # Conversation
    conversation_history: List[Dict] = field(default_factory=list)
    current_question: str = ""
    retry_count: int = 0
    max_retries: int = 3

    # Audit trail
    tool_call_log: List[Dict] = field(default_factory=list)

    def add_to_history(self, question: str, answer: str, narration: str, chart_included: bool) -> None:
        """
        Appends a completed Q&A turn to conversation_history.
        """
        self.conversation_history.append({
            "turn": len(self.conversation_history) + 1,
            "question": question,
            "answer": answer,
            "narration": narration,
            "chart_included": chart_included,
            "timestamp": datetime.now().isoformat()
        })
        self.retry_count = 0

    def log_tool_call(self, tool_name: str, success: bool, execution_time_ms: float, error: Optional[str] = None) -> None:
        """
        Appends a tool call record to tool_call_log.
        """
        self.tool_call_log.append({
            "tool": tool_name,
            "success": success,
            "execution_time_ms": execution_time_ms,
            "error": error,
            "timestamp": datetime.now().isoformat()
        })

    def get_context_summary(self) -> str:
        """
        Returns a compact string summarising session so far.
        """
        history = self.conversation_history
        tools_used = list(set(log["tool"] for log in self.tool_call_log))
        
        return (
            f"Session: {self.session_id} | Dataset: {self.active_filename or 'None'}\n"
            f"Turns completed: {len(history)}\n"
            f"Last question: {history[-1]['question'] if history else 'None'}\n"
            f"Tools used this session: {', '.join(tools_used) if tools_used else 'None'}\n"
            f"Anomalies found: {len(self.anomaly_findings)}"
        )

    def to_dict(self) -> Dict:
        """
        Serialises session to a plain dict (for Streamlit session_state storage).
        Excludes non-serialisable objects (DataFrames, Plotly figures).
        """
        return {
            "session_id": self.session_id,
            "active_filename": self.active_filename,
            "conversation_history": self.conversation_history,
            "tool_call_log": self.tool_call_log,
            "anomaly_findings": self.anomaly_findings,
            "schema_context": self.schema_context,
            "turns_completed": len(self.conversation_history)
        }
