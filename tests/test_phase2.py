"""
Tests for Phase 2: Data Engineering.
Validates file ingestion and schema profiling.
"""

import pytest
import pandas as pd
import numpy as np
import os
from pathlib import Path
from data_engine.file_handler import ingest_file, FileIngestionResult
from data_engine.schema_inspector import (
    inspect_schema, 
    calculate_quality_score, 
    schema_to_prompt_string
)

def test_csv_ingestion(tmp_path):
    """
    Verifies that a basic CSV can be ingested.
    """
    d = tmp_path / "data"
    d.mkdir()
    csv_file = d / "test.csv"
    csv_content = "name,age,city\nAlice,30,New York\nBob,25,Los Angeles\nCharlie,35,Chicago"
    csv_file.write_text(csv_content)
    
    result = ingest_file(str(csv_file))
    
    assert result.success is True
    assert result.file_type == "csv"
    assert "main" in result.dataframes
    df = result.dataframes["main"]
    assert len(df) == 3
    assert list(df.columns) == ["name", "age", "city"]

def test_unsupported_file_type(tmp_path):
    """
    Verifies that unsupported file types return a clean error.
    """
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("not a csv")
    
    result = ingest_file(str(txt_file))
    
    assert result.success is False
    assert "Unsupported" in result.error_message

def test_file_too_large(tmp_path, monkeypatch):
    """
    Verifies file size limits are enforced.
    """
    # Create a small file but mock the limit to be even smaller
    small_file = tmp_path / "small.csv"
    small_file.write_text("id,val\n1,10")
    
    # Mock settings.max_file_size_mb to 0 for this test
    # (Actually we need to patch the config imports or use a mock object)
    # For simplicity, let's assume we can set it via monkeypatch on the settings object
    from config import settings
    monkeypatch.setattr(settings, "max_file_size_mb", 0.000001) # tiny
    
    result = ingest_file(str(small_file))
    
    assert result.success is False
    assert "exceeds max size limit" in result.error_message

def test_schema_inspector_basic():
    """
    Verifies that mixed column types are profiled correctly.
    """
    data = {
        "user_id": [1, 2, 3, 4, 5],
        "created_at": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"]),
        "score": [10.5, 20.0, 15.2, np.nan, 30.1],
        "category": ["A", "B", "A", "C", "B"],
        "notes": ["short note", "another one", "just words", "more stuff", "final"]
    }
    df = pd.DataFrame(data)
    
    profile = inspect_schema(df, "test.csv")
    
    assert profile.total_columns == 5
    assert profile.total_rows == 5
    
    # Map column names to profiles
    cols = {c.name: c for c in profile.columns}
    
    assert cols["user_id"].inferred_meaning == "id_like"
    assert cols["created_at"].inferred_meaning == "datetime"
    assert cols["score"].inferred_meaning == "numeric"
    assert cols["category"].inferred_meaning == "categorical"
    assert cols["score"].null_count == 1
    assert len(cols["category"].sample_values) <= 3

def test_quality_score_penalises_nulls():
    """
    Verifies that high null counts reduce the quality score.
    """
    data = {
        "good": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "bad": [1, 2, 3, 4, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan] # > 50% null
    }
    df = pd.DataFrame(data)
    profile = inspect_schema(df, "bad_data.csv")
    
    # Initial 100
    # No date: -5
    # One col > 50% null: -10
    # Expected score around 85
    assert profile.data_quality_score < 90
    assert any("has 66.67% missing values" in w for w in profile.quality_warnings)

def test_schema_to_prompt_string():
    """
    Verifies that the schema string is formatted correctly.
    """
    df = pd.DataFrame({"age": [20, 30], "name": ["Alice", "Bob"]})
    profile = inspect_schema(df, "users.csv")
    
    prompt_str = schema_to_prompt_string(profile)
    
    assert "=== DATASET CONTEXT ===" in prompt_str
    assert "File: users.csv" in prompt_str
    assert "age (numeric, int64)" in prompt_str
    assert "name (categorical, object)" in prompt_str
    assert prompt_str.endswith("========================")
