"""
Tests for Phase 7: PDF Report Builder.
Validates PDF generation, styling, and content inclusion.
"""

import pytest
import pandas as pd
from pathlib import Path
from datetime import datetime
from tools.report_builder import ReportStyles, run


def test_report_styles_initialise():
    """Verifies ReportStyles initialises with all expected styles."""
    styles = ReportStyles()
    assert styles.title is not None
    assert styles.body is not None
    assert styles.section_header is not None
    assert styles.subsection is not None
    assert styles.question is not None
    assert styles.answer is not None


def test_run_no_dataset():
    """Verifies run() rejects empty session."""
    result = run({})
    assert result.success is False
    assert "No dataset" in result.error_message


def test_run_no_content():
    """Verifies run() rejects session with dataset but no analysis."""
    result = run({
        "active_filename": "test.csv",
        "conversation_history": [],
        "proactive_insights": []
    })
    assert result.success is False
    assert "No analysis" in result.error_message


def test_report_generates_pdf():
    """Verifies complete PDF generation with minimal content."""
    session_state = {
        "active_filename": "test.csv",
        "schema_context": "region (string), sales (float)",
        "schema_profile": None,
        "conversation_history": [{
            "turn": 1,
            "question": "Which region had highest sales?",
            "answer": "East region had the highest sales.",
            "narration": "East outperformed all other regions.",
            "chart_included": False,
            "timestamp": "2024-01-01"
        }],
        "anomaly_findings": ["No anomalies found"],
        "last_chart": None,
        "last_chart_reason": "",
        "proactive_insights": [],
        "session_id": "test001",
    }

    result = run(session_state)
    assert result.success is True
    assert result.output.endswith(".pdf")
    assert Path(result.output).exists()
    assert Path(result.output).stat().st_size > 1000


def test_pdf_has_content():
    """Verifies PDF contains expected text content."""
    try:
        import fitz
    except ImportError:
        pytest.skip("PyMuPDF (fitz) not installed")

    session_state = {
        "active_filename": "test.csv",
        "schema_context": "region (string), sales (float)",
        "schema_profile": None,
        "conversation_history": [{
            "turn": 1,
            "question": "Which region had highest sales?",
            "answer": "East region had the highest sales.",
            "narration": "East outperformed all other regions.",
            "chart_included": False,
            "timestamp": "2024-01-01"
        }],
        "anomaly_findings": [],
        "last_chart": None,
        "last_chart_reason": "",
        "proactive_insights": [],
        "session_id": "test001",
    }

    result = run(session_state)
    assert result.success is True

    doc = fitz.open(result.output)
    text = "".join([p.get_text() for p in doc])
    
    assert "Data Analysis Report" in text
    assert "test.csv" in text
    assert "Which region" in text
