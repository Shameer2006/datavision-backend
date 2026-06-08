"""
Data Processor Service
Handles reading CSV/Excel files and extracting schema metadata (headers, data types, sample rows).
"""

import io
import pandas as pd


def extract_schema(file_bytes: bytes, filename: str) -> dict:
    """
    Read a CSV or Excel file from raw bytes and extract its schema.

    Returns a dict with:
      - columns: list of column names
      - dtypes: dict mapping column name -> pandas dtype string
      - row_count: number of rows
      - sample_data: first 5 rows as list of dicts (for AI context)
    """
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext == "csv":
        df = pd.read_csv(io.BytesIO(file_bytes))
    elif ext in ("xls", "xlsx"):
        df = pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")
    else:
        raise ValueError(f"Unsupported file type: .{ext}. Supported types: .csv, .xls, .xlsx")

    # Build schema metadata
    schema = {
        "filename": filename,
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "row_count": len(df),
        "sample_data": df.head(5).fillna("").to_dict(orient="records"),
    }

    return schema, df
