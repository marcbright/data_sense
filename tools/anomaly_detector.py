"""
Anomaly Detector Tool for the AI Data Analyst Agent.
Statically detects outliers and quality issues in the dataset.
Phase 3: Tool Implementation
"""

import time
import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from tools.registry import ToolResult

def detect_outliers_iqr(series: pd.Series) -> Dict:
    """
    Uses the Interquartile Range (IQR) method to detect outliers.
    """
    if series.empty or not pd.api.types.is_numeric_dtype(series):
        return {"outlier_count": 0, "outlier_pct": 0.0, "lower_bound": 0, "upper_bound": 0, "outlier_values": []}
        
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    
    return {
        "outlier_count": len(outliers),
        "outlier_pct": (len(outliers) / len(series)) * 100 if len(series) > 0 else 0,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "outlier_values": outliers.unique()[:5].tolist()
    }

def run(session_state: dict, column_name: Optional[str] = None) -> ToolResult:
    """
    Runs anomaly detection on the active DataFrame.
    """
    start_time = time.time()
    df = session_state.get("active_dataframe")
    
    if df is None:
        execution_time = (time.time() - start_time) * 1000
        return ToolResult(
            success=False,
            tool_name="anomaly_detector",
            output=None,
            output_type="error",
            error_message="No active dataset found in session.",
            execution_time_ms=execution_time,
            code_executed=None
        )

    findings = []
    
    try:
        # 1. Target column or all numeric
        if column_name:
            if column_name not in df.columns:
                return ToolResult(
                    success=False,
                    tool_name="anomaly_detector",
                    output=None,
                    output_type="error",
                    error_message=f"Column '{column_name}' not found.",
                    execution_time_ms=(time.time() - start_time) * 1000,
                    code_executed=None
                )
            cols_to_check = [column_name]
        else:
            cols_to_check = df.select_dtypes(include=[np.number]).columns.tolist()

        # 2. Run IQR detection
        for col in cols_to_check:
            stats = detect_outliers_iqr(df[col])
            if stats["outlier_count"] > 0:
                findings.append(
                    f"Column '{col}': {stats['outlier_count']} outliers detected ({stats['outlier_pct']:.1f}% of values). "
                    f"Values above {stats['upper_bound']:.2f} or below {stats['lower_bound']:.2f} are suspect."
                )

        # 3. Overall quality checks (if no specific column)
        if not column_name:
            # Duplicates
            duplicates = df.duplicated().sum()
            if duplicates > 0:
                dup_pct = (duplicates / len(df)) * 100
                findings.append(f"{duplicates} duplicate rows found ({dup_pct:.1f}% of dataset)")
            
            # High nulls
            for col in df.columns:
                null_pct = (df[col].isna().sum() / len(df)) * 100
                if null_pct > 30:
                    findings.append(f"Column '{col}' has high missing values ({null_pct:.1f}%)")
            
            # Constant columns
            for col in df.columns:
                if df[col].nunique() == 1:
                    findings.append(f"Column '{col}' has only 1 unique value — likely contains no useful variance for analysis.")

        if not findings:
            findings = ["No significant anomalies detected in the dataset based on current statistical checks."]

        # Store in session
        session_state["anomaly_findings"] = findings
        
        execution_time = (time.time() - start_time) * 1000
        
        return ToolResult(
            success=True,
            tool_name="anomaly_detector",
            output=findings,
            output_type="string",
            error_message=None,
            execution_time_ms=execution_time,
            code_executed=None
        )

    except Exception as e:
        execution_time = (time.time() - start_time) * 1000
        return ToolResult(
            success=False,
            tool_name="anomaly_detector",
            output=None,
            output_type="error",
            error_message=f"Anomaly detection failed: {str(e)}",
            execution_time_ms=execution_time,
            code_executed=None
        )
