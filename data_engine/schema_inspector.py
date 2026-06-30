"""
Schema Inspector module for the AI Data Analyst Agent.
Profiles DataFrames to build a context-rich data dictionary for the LLM.
Phase 2: Data Engineering
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
from data_engine.utils import truncate_dataframe_for_prompt

@dataclass
class ColumnProfile:
    """
    Profile of a single column in a DataFrame.
    """
    name: str
    dtype: str              # pandas dtype as string
    inferred_meaning: str   # one of: "numeric", "categorical", "datetime", "text", "boolean", "id_like"
    null_count: int
    null_pct: float         # 0.0 to 100.0
    unique_count: int
    sample_values: List     # up to 3 representative values
    min_value: Optional[str]   # for numeric/datetime columns
    max_value: Optional[str]   # for numeric/datetime columns
    has_currency_hint: bool # True if column name contains: price, cost, revenue, etc.

@dataclass
class SchemaProfile:
    """
    Complete profile of a DataFrame schema.
    """
    filename: str
    sheet_name: str
    total_rows: int
    total_columns: int
    columns: List[ColumnProfile]
    duplicate_row_count: int
    fully_empty_columns: List[str]
    suggested_date_columns: List[str]   # columns likely to be dates
    suggested_id_columns: List[str]     # columns likely to be IDs/keys
    data_quality_score: float           # 0.0 to 100.0
    quality_warnings: List[str]         # human-readable warning strings

def infer_column_meaning(series: pd.Series, col_name: str) -> str:
    """
    Infers the semantic meaning of a column.
    """
    col_name_lower = col_name.lower()
    
    # 1. If dtype is bool -> "boolean"
    if series.dtype == 'bool' or series.dtype == 'boolean':
        return "boolean"
    
    # 2. If dtype is datetime or column name hints at date
    date_hints = ['date', 'time', 'year', 'month', 'day', 'period', 'quarter', 'week', 'timestamp']
    if pd.api.types.is_datetime64_any_dtype(series) or any(hint in col_name_lower for hint in date_hints):
        return "datetime"
    
    unique_count = series.nunique()
    total_count = len(series)
    
    if total_count == 0:
        return "categorical"

    # 3. If unique_count / len(series) < 0.05 and dtype is object -> "categorical"
    if unique_count / total_count < 0.05 and series.dtype == 'object':
        return "categorical"
    
    # 4. If dtype is numeric and high uniqueness -> "id_like"
    if pd.api.types.is_numeric_dtype(series):
        if unique_count / total_count > 0.95:
            return "id_like"
        return "numeric"
    
    # 5. If dtype is object and avg string length > 50 -> "text"
    if series.dtype == 'object':
        # Safely get mean length of non-null strings
        non_null_objs = series.dropna()
        if not non_null_objs.empty and non_null_objs.apply(lambda x: len(str(x))).mean() > 50:
            return "text"
            
    # Default
    return "categorical"

def profile_column(series: pd.Series, col_name: str) -> ColumnProfile:
    """
    Builds a full ColumnProfile for a single column.
    """
    null_count = int(series.isna().sum())
    total_count = len(series)
    null_pct = (null_count / total_count * 100) if total_count > 0 else 0.0
    unique_count = int(series.nunique())
    
    # Sample values
    non_null_series = series.dropna()
    if not non_null_series.empty:
        # Take up to 3 unique sample values
        samples = non_null_series.unique()[:3].tolist()
        # Convert numpy types to native Python types
        processed_samples = []
        for val in samples:
            if hasattr(val, 'item'):
                processed_samples.append(val.item())
            else:
                processed_samples.append(val)
    else:
        processed_samples = []
        
    inferred = infer_column_meaning(series, col_name)
    
    min_val = None
    max_val = None
    
    # Min/Max for numeric and datetime
    if inferred in ["numeric", "datetime", "id_like"] and not non_null_series.empty:
        try:
            val_min = non_null_series.min()
            val_max = non_null_series.max()
            
            if hasattr(val_min, 'strftime'): # datetime
                min_val = val_min.strftime('%Y-%m-%d %H:%M:%S')
                max_val = val_max.strftime('%Y-%m-%d %H:%M:%S')
            else:
                min_val = str(val_min)
                max_val = str(val_max)
        except:
            pass
            
    # Currency hints
    currency_keywords = [
        'price', 'cost', 'revenue', 'sales', 'amount', 'usd', 'ghs', 
        'eur', 'gbp', 'ngn', 'salary', 'fee', 'total'
    ]
    has_currency = any(keyword in col_name.lower() for keyword in currency_keywords)
    
    return ColumnProfile(
        name=col_name,
        dtype=str(series.dtype),
        inferred_meaning=inferred,
        null_count=null_count,
        null_pct=round(null_pct, 2),
        unique_count=unique_count,
        sample_values=processed_samples,
        min_value=min_val,
        max_value=max_val,
        has_currency_hint=has_currency
    )

def calculate_quality_score(df: pd.DataFrame, columns: List[ColumnProfile]) -> Tuple[float, List[str]]:
    """
    Calculates a data quality score and generates warnings.
    """
    score = 100.0
    warnings = []
    total_rows = len(df)
    
    if total_rows == 0:
        return 0.0, ["DataFrame is completely empty"]
        
    # 1. Row count deduction
    if total_rows < 10:
        score -= 20
        warnings.append("Fewer than 10 rows — too small for meaningful analysis")
        
    # 2. Null percentage deductions
    for col in columns:
        if col.null_pct > 50:
            score -= 10
            warnings.append(f"Column '{col.name}' has {col.null_pct}% missing values")
        elif col.null_pct > 20:
            score -= 5
            warnings.append(f"Column '{col.name}' has {col.null_pct}% missing values")
            
    # 3. Duplicate rows
    duplicates = int(df.duplicated().sum())
    if duplicates > 0:
        dup_pct = (duplicates / total_rows) * 100
        if dup_pct > 1.0:
            score -= 10
            warnings.append(f"{duplicates} duplicate rows detected ({dup_pct:.1f}% of data)")
            
    # 4. Empty columns
    empty_cols = [col.name for col in columns if col.null_count == total_rows]
    for col_name in empty_cols:
        score -= 5
        warnings.append(f"Column '{col_name}' is completely empty and will be ignored")
        
    # 5. No date column
    has_date = any(col.inferred_meaning == "datetime" for col in columns)
    if not has_date:
        score -= 5
        warnings.append("No date column detected — time-based analysis will not be available")
        
    return max(0.0, min(100.0, score)), warnings

def inspect_schema(df: pd.DataFrame, filename: str, sheet_name: str = "main") -> SchemaProfile:
    """
    Main entry point for profiling a DataFrame.
    """
    total_rows = len(df)
    total_cols = len(df.columns)
    
    col_profiles = [profile_column(df[col], col) for col in df.columns]
    
    quality_score, warnings = calculate_quality_score(df, col_profiles)
    
    duplicates = int(df.duplicated().sum())
    empty_cols = [col.name for col in col_profiles if col.null_count == total_rows]
    
    date_cols = [col.name for col in col_profiles if col.inferred_meaning == "datetime"]
    id_cols = [col.name for col in col_profiles if col.inferred_meaning == "id_like"]
    
    return SchemaProfile(
        filename=filename,
        sheet_name=sheet_name,
        total_rows=total_rows,
        total_columns=total_cols,
        columns=col_profiles,
        duplicate_row_count=duplicates,
        fully_empty_columns=empty_cols,
        suggested_date_columns=date_cols,
        suggested_id_columns=id_cols,
        data_quality_score=quality_score,
        quality_warnings=warnings
    )

def schema_to_prompt_string(profile: SchemaProfile) -> str:
    """
    Converts a SchemaProfile into a string for LLM prompts.
    """
    lines = [
        "=== DATASET CONTEXT ===",
        f"File: {profile.filename} | Sheet: {profile.sheet_name}",
        f"Size: {profile.total_rows} rows × {profile.total_columns} columns",
        f"Data quality score: {profile.data_quality_score}/100",
        "",
        "COLUMNS:"
    ]
    
    # Limit to 20 columns for brevity
    display_cols = profile.columns[:20]
    for col in display_cols:
        lines.append(f"- {col.name} ({col.inferred_meaning}, {col.dtype}): {col.unique_count} unique values, {col.null_pct}% null. Sample: {col.sample_values}")
        
    if len(profile.columns) > 20:
        lines.append(f"+ {len(profile.columns) - 20} more columns not shown")
        
    lines.append("")
    lines.append("WARNINGS:")
    if profile.quality_warnings:
        for warning in profile.quality_warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- None")
        
    lines.append("")
    lines.append(f"DATE COLUMNS: {', '.join(profile.suggested_date_columns) if profile.suggested_date_columns else 'None detected'}")
    
    numeric_names = [col.name for col in profile.columns if col.inferred_meaning == "numeric"]
    lines.append(f"NUMERIC COLUMNS: {', '.join(numeric_names) if numeric_names else 'None detected'}")
    
    categorical_names = [col.name for col in profile.columns if col.inferred_meaning == "categorical"]
    lines.append(f"CATEGORICAL COLUMNS: {', '.join(categorical_names) if categorical_names else 'None detected'}")
    
    lines.append("========================")
    
    return "\n".join(lines)
