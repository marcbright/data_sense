import pandas as pd
from pathlib import Path
from agent.insights_engine import (
    _analyse_top_bottom,
    _analyse_correlations,
    _analyse_data_quality,
    generate_proactive_insights,
    ProactiveInsight
)


def test_top_bottom_analysis():
    df = pd.DataFrame({
        "region": ["East", "East", "West", "North", "South", "East"],
        "sales_usd": [67800, 67000, 21300, 45000, 32000, 68000]
    })

    findings = _analyse_top_bottom(df)

    assert len(findings) >= 1
    assert any("chart_suggestion" in finding and finding["chart_suggestion"] for finding in findings)


def test_correlation_detection():
    df = pd.DataFrame({
        "col_a": [1, 2, 3, 4, 5],
        "col_b": [2, 4, 6, 8, 10]
    })

    findings = _analyse_correlations(df)

    assert len(findings) >= 1
    assert any(abs(finding["correlation"]) > 0.9 for finding in findings)


def test_data_quality_nulls():
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "sales": [100.0, None, 150.0, None, None]
    })

    findings = _analyse_data_quality(df)

    assert len(findings) >= 1
    assert any(finding.get("severity") == "warning" for finding in findings)


def test_generate_insights_returns_list():
    csv_path = Path(__file__).resolve().parent.parent / "uploaded_files" / "test_sales.csv"
    df = pd.read_csv(csv_path)

    insights = generate_proactive_insights(df, "test.csv", "region, sales_usd columns")

    assert isinstance(insights, list)
    assert len(insights) >= 1
    assert all(isinstance(item, ProactiveInsight) for item in insights)


def test_insight_has_required_fields():
    csv_path = Path(__file__).resolve().parent.parent / "uploaded_files" / "test_sales.csv"
    df = pd.read_csv(csv_path)

    insights = generate_proactive_insights(df, "test.csv", "region, sales_usd columns")
    assert len(insights) >= 1

    insight = insights[0]
    assert insight.title.strip() != ""
    assert insight.category in [
        "trend", "outlier", "distribution", "correlation",
        "data_quality", "top_bottom"
    ]
    assert insight.severity in ["info", "warning", "critical"]
