"""
Tests for Phase 4: Agent Brain.
Validates session state, intent classification, and planning logic.
"""

import json
import os

import pandas as pd
import pytest
from agent.intent_classifier import (
    IntentResult,
    IntentType,
    _extract_columns_from_session,
    classify_intent,
)
from agent.loop import run_agent
from agent.planner import ExecutionPlan, create_plan
from agent.session_state import SessionState

def test_session_state_creation():
    """
    Verifies that SessionState initializes with expected defaults.
    """
    state = SessionState()
    assert len(state.session_id) == 8
    assert state.conversation_history == []
    assert state.active_dataframe is None
    assert state.retry_count == 0

def test_add_to_history():
    """
    Verifies that turns are added correctly to history.
    """
    state = SessionState()
    state.add_to_history("What is total sales?", "Total sales is 5000", "Sales are good", False)
    
    assert len(state.conversation_history) == 1
    assert state.conversation_history[0]["turn"] == 1
    assert state.conversation_history[0]["question"] == "What is total sales?"
    
    state.add_to_history("Next question?", "Answer 2", "Narration 2", True)
    assert len(state.conversation_history) == 2
    assert state.conversation_history[1]["turn"] == 2

def test_get_context_summary_empty():
    """
    Verifies summary works on empty session.
    """
    state = SessionState()
    summary = state.get_context_summary()
    assert "Dataset: None" in summary
    assert "Turns completed: 0" in summary
    assert "Last question: None" in summary

def test_plan_greeting():
    """
    Verifies planning for greetings.
    """
    intent = IntentResult(
        intent=IntentType.GREETING,
        confidence=1.0,
        requires_clarification=False,
        clarification_question=None,
        detected_columns=[],
        detected_time_range=None,
        detected_metric=None
    )
    state = SessionState()
    plan = create_plan(intent, "Hello", state)
    
    assert plan.steps == []
    assert "greeting" in plan.plan_summary.lower()

def test_plan_data_query_has_four_steps():
    """
    Verifies planning for standard data queries.
    """
    intent = IntentResult(
        intent=IntentType.DATA_QUERY,
        confidence=0.9,
        requires_clarification=False,
        clarification_question=None,
        detected_columns=["sales"],
        detected_time_range=None,
        detected_metric="sales"
    )
    state = SessionState()
    plan = create_plan(intent, "Show sales by region", state)
    
    # Steps: memory_retriever, sql_executor, chart_generator, insight_narrator
    assert len(plan.steps) == 4
    tool_names = [s.tool_name for s in plan.steps]
    assert "memory_retriever" in tool_names
    assert "sql_executor" in tool_names
    assert "chart_generator" in tool_names
    assert "insight_narrator" in tool_names

def test_plan_export_has_one_step():
    """
    Verifies planning for export requests.
    """
    intent = IntentResult(
        intent=IntentType.EXPORT,
        confidence=1.0,
        requires_clarification=False,
        clarification_question=None,
        detected_columns=[],
        detected_time_range=None,
        detected_metric=None
    )
    state = SessionState()
    plan = create_plan(intent, "Download PDF report", state)
    
    assert len(plan.steps) == 1
    assert plan.steps[0].tool_name == "report_builder"


def test_extract_columns_from_session_uses_loaded_dataframe():
    """The classifier should read actual dataframe columns when present."""
    state = SessionState()
    state.active_dataframe = pd.DataFrame({
        "region": ["North", "South"],
        "sales_usd": [1200, 850],
        "product": ["Laptop", "Phone"],
    })

    columns = _extract_columns_from_session(state)

    assert columns == ["region", "sales_usd", "product"]


def test_classify_intent_uses_real_columns_to_avoid_clarification(monkeypatch):
    """Questions naming real columns should not trigger unnecessary clarification."""

    def fake_generate_content(prompt):
        return json.dumps({
            "intent": "data_query",
            "confidence": 0.9,
            "requires_clarification": True,
            "clarification_question": "Which column do you mean?",
            "detected_columns": [],
            "detected_time_range": None,
            "detected_metric": None,
        })

    monkeypatch.setattr("agent.intent_classifier.generate_content", fake_generate_content)

    state = SessionState()
    state.active_dataframe = pd.DataFrame({
        "department": ["HR", "IT"],
        "salary": [3000, 4500],
        "hire_date": ["2020", "2021"],
    })

    result = classify_intent("Which department has the highest salary?", state)

    assert result.requires_clarification is False
    assert "department" in result.detected_columns
    assert result.intent == IntentType.DATA_QUERY

@pytest.mark.skipif(not os.environ.get("GEMINI_API_KEY"), reason="Gemini API key not set")
def test_agent_response_greeting_integration():
    """
    Integration test for greeting flow if API key is present.
    """
    state = SessionState()
    response = run_agent("Hi there", state)
    
    assert response.success is True
    assert "Hello" in response.answer_text
    assert response.clarification_needed is False
