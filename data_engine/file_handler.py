"""
File Handler module for the AI Data Analyst Agent.
Handles ingestion and validation of CSV, Excel, and PDF files.
Phase 2: Data Engineering
"""

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List
import pandas as pd
import fitz  # PyMuPDF

from config import settings
from data_engine.utils import sanitise_filename, format_file_size

@dataclass
class FileIngestionResult:
    """
    Result of the file ingestion process.
    """
    success: bool
    filename: str
    file_type: str              # "csv", "excel", "pdf"
    dataframes: Dict[str, pd.DataFrame]  # sheet_name -> DataFrame (CSV/PDF use "main")
    raw_text: Optional[str]        # PDF extracted text, None for others
    file_size_readable: str     # e.g. "2.4 MB"
    row_count: int              # total rows across all sheets
    error_message: Optional[str]   # populated only on failure

def save_uploaded_file(file_bytes: bytes, filename: str) -> Path:
    """
    Saves raw bytes (from a Streamlit file uploader) to the uploaded_files/ directory.
    Returns the full path.
    """
    upload_dir = Path("uploaded_files")
    upload_dir.mkdir(exist_ok=True)
    
    safe_name = sanitise_filename(filename)
    file_path = upload_dir / safe_name
    
    with open(file_path, "wb") as f:
        f.write(file_bytes)
    
    return file_path

def ingest_file(file_path: str | Path) -> FileIngestionResult:
    """
    Ingests a file, validates it, and parses it into DataFrames.
    
    Args:
        file_path: Path to the file to ingest.
        
    Returns:
        FileIngestionResult: Object containing the parsed data or error details.
    """
    path = Path(file_path)
    filename = path.name
    
    try:
        # 1. VALIDATE
        if not path.exists():
            return FileIngestionResult(
                success=False, filename=filename, file_type="unknown",
                dataframes={}, raw_text=None, file_size_readable="0 B",
                row_count=0, error_message=f"File not found: {path}"
            )
            
        file_size_bytes = path.stat().st_size
        file_size_readable = format_file_size(file_size_bytes)
        max_size_bytes = settings.max_file_size_mb * 1024 * 1024
        
        if file_size_bytes > max_size_bytes:
            return FileIngestionResult(
                success=False, filename=filename, file_type="unknown",
                dataframes={}, raw_text=None, file_size_readable=file_size_readable,
                row_count=0, error_message=f"File exceeds max size limit. Size: {file_size_readable}, Limit: {settings.max_file_size_mb} MB"
            )
            
        extension = path.suffix.lower()
        if extension not in ['.csv', '.xlsx', '.xls', '.pdf']:
            return FileIngestionResult(
                success=False, filename=filename, file_type="unknown",
                dataframes={}, raw_text=None, file_size_readable=file_size_readable,
                row_count=0, error_message="Unsupported file type. Accepted: CSV, Excel (.xlsx/.xls), PDF"
            )
            
        # 2. ROUTE BY FILE TYPE
        dataframes: Dict[str, pd.DataFrame] = {}
        raw_text: Optional[str] = None
        file_type = ""
        
        if extension == '.csv':
            file_type = "csv"
            try:
                df = pd.read_csv(path, encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(path, encoding="latin-1")
            dataframes = {"main": df}
            
        elif extension in ['.xlsx', '.xls']:
            file_type = "excel"
            sheets_dict = pd.read_excel(path, sheet_name=None)
            # Filter out empty sheets
            dataframes = {name: df for name, df in sheets_dict.items() if len(df) > 0}
            if not dataframes:
                return FileIngestionResult(
                    success=False, filename=filename, file_type=file_type,
                    dataframes={}, raw_text=None, file_size_readable=file_size_readable,
                    row_count=0, error_message="Excel file appears empty"
                )
                
        elif extension == '.pdf':
            file_type = "pdf"
            doc = fitz.open(path)
            all_text_list = []
            
            for page_num, page in enumerate(doc):
                text = page.get_text()
                all_text_list.append(text)
                
                # Attempt to find tables in text (simple heuristic)
                try:
                    # Look for CSV-like structure in page text
                    # We wrap in StringIO and try reading it as CSV
                    df_test = pd.read_csv(io.StringIO(text), sep=None, engine='python', on_bad_lines='skip')
                    if len(df_test.columns) >= 2 and len(df_test) >= 3:
                        dataframes[f"page_{page_num + 1}"] = df_test
                except Exception:
                    pass
            
            raw_text = "\n".join(all_text_list)
            doc.close()

        # 3. BUILD RESULT
        total_rows = sum(len(df) for df in dataframes.values())
        
        return FileIngestionResult(
            success=True,
            filename=filename,
            file_type=file_type,
            dataframes=dataframes,
            raw_text=raw_text,
            file_size_readable=file_size_readable,
            row_count=total_rows,
            error_message=None
        )

    except Exception as e:
        return FileIngestionResult(
            success=False,
            filename=filename,
            file_type="unknown",
            dataframes={},
            raw_text=None,
            file_size_readable="0 B",
            row_count=0,
            error_message=f"Ingestion failed [ {type(e).__name__} ]: {str(e)}"
        )
