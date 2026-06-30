"""
Utility module for the AI Data Analyst Agent.
Provides helper functions for prompt loading, file handling, and data formatting.
Phase 1: Foundation
"""

import os
import re
from datetime import datetime
from typing import Any
import pandas as pd

def load_prompt(prompt_name: str) -> str:
    """
    Reads a .txt file from the /prompts/ folder by name and returns its content.
    
    Args:
        prompt_name: The name of the prompt file (without .txt extension).
        
    Returns:
        str: The content of the prompt file.
        
    Raises:
        FileNotFoundError: If the prompt file does not exist.
    """
    # Use relative path from project root
    prompt_path = os.path.join("prompts", f"{prompt_name}.txt")
    
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt file not found at: {prompt_path}")
        
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read().strip()

def sanitise_filename(filename: str) -> str:
    """
    Cleans a filename by removing dangerous characters and replacing spaces.
    
    Args:
        filename: The original filename.
        
    Returns:
        str: A safe, sanitised version of the filename (max 100 chars).
    """
    # Remove file extension first to process name
    name_part, ext = os.path.splitext(filename)
    
    # Replace spaces and non-alphanumeric chars with underscores
    clean_name = re.sub(r'[^\w\s-]', '', name_part)
    clean_name = re.sub(r'[\s]+', '_', clean_name).strip('_')
    
    # Rejoin with extension and truncate
    sanitised = f"{clean_name[:95]}{ext}"
    return sanitised

def format_file_size(size_bytes: int) -> str:
    """
    Converts a battery of bytes into a human-readable string.
    
    Args:
        size_bytes: Size in bytes.
        
    Returns:
        str: Human-readable size (e.g., "2.4 MB").
    """
    if size_bytes == 0:
        return "0 B"
        
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"

def get_timestamp() -> str:
    """
    Returns the current timestamp in a safe format for filenames.
    
    Returns:
        str: Timestamp in YYYYMMDD_HHMMSS format.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def truncate_dataframe_for_prompt(df: pd.DataFrame, max_rows: int = 5) -> str:
    """
    Creates a compact string output of a DataFrame for LLM context.
    
    Args:
        df: The Pandas DataFrame to represent.
        max_rows: Number of rows to include (default 5).
        
    Returns:
        str: A string showing column names, types, and the first few rows.
    """
    if df is None:
        return "Empty DataFrame"
        
    header = f"Columns: {', '.join(df.columns.tolist())}\n"
    dtypes = f"Data Types: {df.dtypes.to_dict()}\n"
    preview = df.head(max_rows).to_string(index=False)
    
    return f"{header}{dtypes}\nData Preview:\n{preview}"
