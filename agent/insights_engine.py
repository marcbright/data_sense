import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from config import generate_content
from data_engine.utils import load_prompt
from tools.anomaly_detector import detect_outliers_iqr


@dataclass
class ProactiveInsight:
    insight_id: str
    category: str
    title: str
    finding: str
    severity: str
    affected_columns: list[str] = field(default_factory=list)
    supporting_stat: str = "N/A"
    chart_suggestion: dict | None = None


def _format_number(value: float, precision: int = 2) -> str:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "N/A"
        if float(value).is_integer():
            return f"{int(value):,}"
        formatted = f"{value:,.{precision}f}".rstrip("0").rstrip(".")
        return formatted
    except Exception:
        return str(value)


def _coerce_numeric_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)

    cleaned = series.astype(str).str.replace(r"[,$]", "", regex=True)
    coerced = pd.to_numeric(cleaned, errors="coerce")
    if coerced.notna().sum() >= max(1, len(coerced) * 0.8):
        return coerced.astype(float)
    return pd.Series(dtype=float)


def _get_numeric_columns(df: pd.DataFrame) -> dict[str, pd.Series]:
    numeric_map = {}
    for col in df.columns.tolist():
        series = _coerce_numeric_series(df[col])
        if series.notna().any():
            numeric_map[col] = series
    return numeric_map


def _analyse_distributions(df: pd.DataFrame) -> list[dict]:
    findings = []
    try:
        numeric_map = _get_numeric_columns(df)
        for col, series in numeric_map.items():
            series = series.dropna()
            if series.empty:
                continue

            mean = series.mean()
            std = series.std()
            skewness = series.skew()
            unique_count = series.nunique(dropna=True)

            if abs(skewness) > 1:
                findings.append({
                    "category": "distribution",
                    "column": col,
                    "stat_label": "skewness",
                    "stat_value": float(skewness),
                    "description": (
                        f"{col} has a skewed distribution with skewness of {skewness:.2f}, "
                        f"suggesting values are unevenly spread."
                    )
                })

            coef_var = abs(std / mean) if abs(mean) > 1e-9 else float("inf") if std > 0 else 0.0
            if coef_var > 0.5:
                findings.append({
                    "category": "distribution",
                    "column": col,
                    "stat_label": "cv",
                    "stat_value": float(coef_var),
                    "description": (
                        f"{col} shows high variability with a coefficient of variation of {coef_var:.2f}, "
                        f"meaning individual values are widely dispersed around the average."
                    )
                })

            if unique_count < 5:
                findings.append({
                    "category": "distribution",
                    "column": col,
                    "stat_label": "unique_values",
                    "stat_value": float(unique_count),
                    "description": (
                        f"{col} only has {unique_count} distinct values, "
                        f"which limits the variety of outcomes in this measure."
                    )
                })

        return findings
    except Exception:
        return []


def _analyse_top_bottom(df: pd.DataFrame) -> list[dict]:
    findings = []
    try:
        numeric_map = _get_numeric_columns(df)
        numeric_cols = list(numeric_map.keys())
        categorical_cols = [
            col for col in df.columns.tolist()
            if col not in numeric_cols
        ]

        candidates = []
        for cat_col in categorical_cols:
            if df[cat_col].nunique(dropna=True) > 20:
                continue
            for num_col in numeric_cols:
                numeric_series = numeric_map[num_col]
                grouped = numeric_series.groupby(df[cat_col], dropna=False).sum()
                if grouped.empty or len(grouped) < 2:
                    continue

                top_label = grouped.idxmax()
                bottom_label = grouped.idxmin()
                top_value = float(grouped.max())
                bottom_value = float(grouped.min())
                if top_value == 0:
                    continue

                gap_pct = abs(top_value - bottom_value) / abs(top_value) * 100.0
                if gap_pct <= 50:
                    continue

                candidates.append({
                    "category": "top_bottom",
                    "numeric_col": num_col,
                    "categorical_col": cat_col,
                    "top_label": str(top_label),
                    "top_value": top_value,
                    "bottom_label": str(bottom_label),
                    "bottom_value": bottom_value,
                    "gap_pct": float(gap_pct),
                    "description": (
                        f"{top_label} leads {num_col} with {top_value:,.0f}, "
                        f"while {bottom_label} has {bottom_value:,.0f}, a {gap_pct:.0f}% gap."
                    ),
                    "chart_suggestion": {
                        "chart_type": "bar",
                        "x": cat_col,
                        "y": num_col,
                        "title": f"{num_col} by {cat_col}"
                    }
                })

        candidates.sort(key=lambda x: x["gap_pct"], reverse=True)
        return candidates[:3]
    except Exception:
        return []


def _analyse_correlations(df: pd.DataFrame) -> list[dict]:
    findings = []
    try:
        numeric_map = _get_numeric_columns(df)
        numeric_cols = list(numeric_map.keys())
        if len(numeric_cols) < 2:
            return []

        corr_matrix = pd.DataFrame({col: numeric_map[col] for col in numeric_cols}).corr(method="pearson")
        pairs = []
        for i, col_a in enumerate(numeric_cols):
            for col_b in numeric_cols[i + 1:]:
                corr_value = corr_matrix.loc[col_a, col_b]
                if pd.isna(corr_value):
                    continue
                if abs(corr_value) > 0.7:
                    pairs.append({
                        "col_a": col_a,
                        "col_b": col_b,
                        "correlation": float(corr_value)
                    })

        pairs.sort(key=lambda item: abs(item["correlation"]), reverse=True)
        for item in pairs[:2]:
            direction = "positive" if item["correlation"] > 0 else "negative"
            findings.append({
                "category": "correlation",
                "col_a": item["col_a"],
                "col_b": item["col_b"],
                "correlation": item["correlation"],
                "direction": direction,
                "description": (
                    f"{item['col_a']} and {item['col_b']} move together with a {direction} correlation of {item['correlation']:.2f}."
                ),
                "chart_suggestion": {
                    "chart_type": "scatter",
                    "x": item["col_a"],
                    "y": item["col_b"],
                    "title": f"{item['col_a']} vs {item['col_b']}"
                }
            })

        return findings
    except Exception:
        return []


def _analyse_time_trends(df: pd.DataFrame) -> list[dict]:
    findings = []
    try:
        date_candidates = []
        keywords = ["date", "month", "year", "period", "quarter", "week", "time"]
        for col in df.columns.tolist():
            lower_name = col.lower()
            if pd.api.types.is_datetime64_any_dtype(df[col]) or any(keyword in lower_name for keyword in keywords):
                date_candidates.append(col)

        valid_date_cols = []
        for col in date_candidates:
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().sum() >= max(2, len(parsed) * 0.25):
                    valid_date_cols.append((col, parsed))
            except Exception:
                continue

        if not valid_date_cols:
            return []

        numeric_map = _get_numeric_columns(df)
        numeric_cols = list(numeric_map.keys())
        if not numeric_cols:
            return []

        for date_col, parsed_dates in valid_date_cols:
            ranked = df.copy()
            ranked = ranked.assign(_date_temp=parsed_dates)
            ranked = ranked.dropna(subset=["_date_temp"])
            if ranked.shape[0] < 4:
                continue

            ranked = ranked.sort_values("_date_temp")
            mid = ranked.shape[0] // 2
            first_half = ranked.iloc[:mid]
            second_half = ranked.iloc[mid:]
            for num_col in numeric_cols:
                numeric_series = numeric_map[num_col]
                first_mean = numeric_series.loc[first_half.index].dropna().mean()
                second_mean = numeric_series.loc[second_half.index].dropna().mean()
                if pd.isna(first_mean) or pd.isna(second_mean):
                    continue

                if abs(first_mean) < 1e-9:
                    change_pct = 100.0 if abs(second_mean) > 0 else 0.0
                else:
                    change_pct = (second_mean - first_mean) / abs(first_mean) * 100.0

                if abs(change_pct) <= 10:
                    continue

                direction = "upward" if change_pct > 0 else "downward"
                findings.append({
                    "category": "trend",
                    "date_col": date_col,
                    "numeric_col": num_col,
                    "direction": direction,
                    "change_pct": float(change_pct),
                    "description": (
                        f"{num_col} changed {direction} by {abs(change_pct):.0f}% between the first and second halves of {date_col}."
                    ),
                    "chart_suggestion": {
                        "chart_type": "line",
                        "x": date_col,
                        "y": num_col,
                        "title": f"{num_col} trend over {date_col}"
                    }
                })

            if findings:
                break

        return findings
    except Exception:
        return []


def _analyse_data_quality(df: pd.DataFrame) -> list[dict]:
    findings = []
    try:
        total_rows = len(df)
        if total_rows == 0:
            return []

        # Missing data
        for col in df.columns.tolist():
            null_pct = (df[col].isna().sum() / total_rows) * 100.0
            if 5 < null_pct <= 30:
                findings.append({
                    "category": "data_quality",
                    "column": col,
                    "severity": "info",
                    "description": (
                        f"{col} has {null_pct:.0f}% missing values, which may affect some summaries."
                    ),
                    "supporting_stat": f"{null_pct:.0f}% missing"
                })
            elif null_pct > 30:
                findings.append({
                    "category": "data_quality",
                    "column": col,
                    "severity": "warning",
                    "description": (
                        f"{col} is missing {null_pct:.0f}% of its values, which is likely to impact analysis reliability."
                    ),
                    "supporting_stat": f"{null_pct:.0f}% missing"
                })

        # Duplicates
        duplicate_count = int(df.duplicated().sum())
        if duplicate_count > 0:
            duplicate_pct = (duplicate_count / total_rows) * 100.0
            severity = "critical" if duplicate_pct > 30 else "warning"
            findings.append({
                "category": "data_quality",
                "column": "dataset",
                "severity": severity,
                "description": (
                    f"The dataset contains {duplicate_count} duplicate rows ({duplicate_pct:.0f}%), which may distort aggregated results."
                ),
                "supporting_stat": f"{duplicate_pct:.0f}% duplicates"
            })

        # ID-like columns with duplicates and constant columns
        for col in df.columns.tolist():
            lower_name = col.lower()
            is_id_like = any(keyword in lower_name for keyword in ["id", "code", "key", "uuid"])
            unique_ratio = df[col].nunique(dropna=True) / total_rows if total_rows > 0 else 0
            if is_id_like and unique_ratio < 0.95:
                findings.append({
                    "category": "data_quality",
                    "column": col,
                    "severity": "warning",
                    "description": (
                        f"{col} appears to be an identifier field but only {unique_ratio:.0f}% of values are unique."
                    ),
                    "supporting_stat": f"{unique_ratio:.0f}% unique"
                })

            if df[col].nunique(dropna=True) <= 1:
                findings.append({
                    "category": "data_quality",
                    "column": col,
                    "severity": "warning",
                    "description": (
                        f"{col} has a single unique value, indicating it carries no useful variance for analysis."
                    ),
                    "supporting_stat": "constant column"
                })

        # Outlier detection as proactive data issue
        numeric_map = _get_numeric_columns(df)
        for col, series in numeric_map.items():
            stats = detect_outliers_iqr(series)
            if stats["outlier_count"] > 0:
                severity = "warning" if stats["outlier_pct"] > 5 else "info"
                findings.append({
                    "category": "outlier",
                    "column": col,
                    "severity": severity,
                    "description": (
                        f"{stats['outlier_count']} outliers were detected in {col}, representing {stats['outlier_pct']:.1f}% of the values."
                    ),
                    "supporting_stat": (
                        f"{stats['outlier_count']} outliers ({stats['outlier_pct']:.1f}%)"
                    )
                })

        return findings
    except Exception:
        return []


def _make_title(finding: dict) -> str:
    title = "Insight"
    if finding["category"] == "top_bottom":
        title = f"{finding['top_label']} leads in {finding['numeric_col']}"
    elif finding["category"] == "trend":
        title = f"{finding['numeric_col']} trending {finding['direction']}"
    elif finding["category"] == "correlation":
        title = f"{finding['col_a']} correlates with {finding['col_b']}"
    elif finding["category"] == "distribution":
        title = f"High variability in {finding['column']}"
    elif finding["category"] == "data_quality":
        title = f"Data quality note: {finding.get('column', 'dataset')}"
    elif finding["category"] == "outlier":
        title = f"Outliers detected in {finding.get('column', 'data')}"

    return title[:60]


def _get_affected_cols(finding: dict) -> list[str]:
    cols = []
    for key in [
        "column", "numeric_col", "categorical_col", "col_a", "col_b", "date_col"
    ]:
        if key in finding and finding[key] not in cols and finding[key] is not None:
            cols.append(str(finding[key]))
    return cols


def _get_supporting_stat(finding: dict) -> str:
    if finding["category"] == "top_bottom":
        top = _format_number(finding.get("top_value", 0))
        bottom = _format_number(finding.get("bottom_value", 0))
        return f"{finding.get('top_label')}: {top} vs {finding.get('bottom_label')}: {bottom}"
    if finding["category"] == "correlation":
        return f"r = {finding.get('correlation', 0):.2f}"
    if finding["category"] == "trend":
        change_pct = finding.get("change_pct", 0.0)
        sign = "+" if change_pct >= 0 else "-"
        return f"{sign}{abs(change_pct):.0f}% change"
    if finding["category"] == "distribution":
        if finding.get("stat_label") == "cv":
            return f"CV = {finding.get('stat_value', 0):.2f}"
        if finding.get("stat_label") == "skewness":
            return f"Skew = {finding.get('stat_value', 0):.2f}"
        if finding.get("stat_label") == "unique_values":
            return f"{int(finding.get('stat_value', 0))} unique values"
    if finding["category"] == "data_quality" or finding["category"] == "outlier":
        stat = finding.get("supporting_stat")
        if stat:
            return stat
    return "N/A"


def generate_proactive_insights(
    df: pd.DataFrame,
    filename: str,
    schema_context: str
) -> list[ProactiveInsight]:
    try:
        if df is None or df.empty:
            return []

        distribution_findings = _analyse_distributions(df)
        top_bottom_findings = _analyse_top_bottom(df)
        correlation_findings = _analyse_correlations(df)
        trend_findings = _analyse_time_trends(df)
        quality_findings = _analyse_data_quality(df)

        all_raw = (
            distribution_findings + top_bottom_findings + correlation_findings + trend_findings + quality_findings
        )

        if not all_raw:
            return [
                ProactiveInsight(
                    insight_id="insight_001",
                    category="data_quality",
                    title="Clean data summary",
                    finding=(
                        "The dataset appears clean and did not trigger any proactive issues. "
                        "You can still ask a question to explore specific trends or comparisons."
                    ),
                    severity="info",
                    affected_columns=[],
                    supporting_stat="No issues detected",
                    chart_suggestion=None
                )
            ]

        priority_map = {
            "trend": 1,
            "top_bottom": 2,
            "outlier": 3,
            "correlation": 4,
            "distribution": 5,
            "data_quality": 6
        }

        sorted_findings = sorted(all_raw, key=lambda item: priority_map.get(item.get("category"), 99))
        selected = sorted_findings[:6]

        if any(item["category"] == "data_quality" for item in all_raw) and not any(item["category"] == "data_quality" for item in selected):
            data_quality_item = next(item for item in all_raw if item["category"] == "data_quality")
            selected.append(data_quality_item)

        try:
            system_instruction = load_prompt("insight_narrator")
        except Exception:
            system_instruction = ""

        insights = []
        for idx, finding in enumerate(selected):
            prompt = (
                f"You are a data analyst presenting a finding to a non-technical business executive.\n\n"
                f"Dataset: {filename}\n"
                f"Schema: {schema_context[:300]}\n\n"
                f"Raw finding: {finding['description']}\n"
                f"Category: {finding['category']}\n\n"
                f"Write a 2-sentence business insight from this finding.\n"
                f"Sentence 1: State the finding clearly with the number.\n"
                f"Sentence 2: State one business implication or recommended action.\n\n"
                f"Be specific. Use the actual numbers.\n"
                f"Do not use technical jargon.\n"
                f"Do not start with 'I' or 'The data shows'.\n"
                f"Max 60 words total.\n"
                f"Return only the 2 sentences, nothing else."
            )

            finding_text = finding["description"]
            try:
                llm_response = generate_content(prompt, system_instruction)
                if llm_response and llm_response.strip():
                    finding_text = " ".join(llm_response.strip().splitlines()).strip()
            except Exception:
                pass

            insights.append(ProactiveInsight(
                insight_id=f"insight_{idx + 1:03d}",
                category=finding.get("category", "data_quality"),
                title=_make_title(finding),
                finding=finding_text,
                severity=finding.get("severity", "info"),
                affected_columns=_get_affected_cols(finding),
                supporting_stat=_get_supporting_stat(finding),
                chart_suggestion=finding.get("chart_suggestion")
            ))

        return insights
    except Exception:
        return [
            ProactiveInsight(
                insight_id="insight_001",
                category="data_quality",
                title="Dataset loaded successfully",
                finding=(
                    "The dataset was loaded but automatic insight generation encountered an error. "
                    "Ask a question to begin analysis."
                ),
                severity="info",
                affected_columns=[],
                supporting_stat="N/A",
                chart_suggestion=None
            )
        ]
