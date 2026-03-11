"""
File Loader Function

Loads tape files (Excel/CSV) into DataFrames.

Supports two environments:
  1. Databricks cluster  – /Volumes/ FUSE mount is available
  2. Model Serving endpoint – no FUSE; uses Databricks SDK (REST API)
"""

import io
import os
from typing import Union

import pandas as pd


def load_tape_file(file_path: str) -> pd.DataFrame:
    """
    Load tape file (Excel or CSV) into pandas DataFrame.

    Supports:
    - Excel files (.xlsx, .xls)
    - CSV files (.csv)
    - Unity Catalog Volume paths (/Volumes/...)
    - DBFS paths (/dbfs/...)
    - Local paths
    """
    normalized_path = _normalize_path(file_path)
    file_ext = os.path.splitext(normalized_path)[1].lower()

    if file_ext not in ('.xlsx', '.xls', '.csv'):
        raise ValueError(
            f"Unsupported file format: {file_ext}. "
            f"Supported formats: .xlsx, .xls, .csv"
        )

    file_data = _resolve_file(normalized_path, file_path)

    try:
        if file_ext in ('.xlsx', '.xls'):
            df = pd.read_excel(file_data, engine='openpyxl')
        else:
            df = pd.read_csv(file_data)
        return df
    except Exception as e:
        raise ValueError(f"Error loading file {file_path}: {str(e)}")


def _resolve_file(normalized_path: str, original_path: str):
    """
    Try to resolve the file, returning either a local path (str)
    or an in-memory BytesIO obtained via REST API.
    """
    # Method 1: direct filesystem (cluster with FUSE mount)
    if os.path.exists(normalized_path):
        return normalized_path

    # Method 2: Databricks Files API via requests (serving endpoint)
    if normalized_path.startswith("/Volumes/"):
        print(f"[file_loader] Direct path not found, trying Files API for: {normalized_path}", flush=True)

        token = os.environ.get("DATABRICKS_TOKEN", "").strip()
        host = os.environ.get("DATABRICKS_HOST", "").strip().rstrip("/")

        if token and host:
            try:
                import requests

                api_url = f"{host}/api/2.0/fs/files{normalized_path}"
                resp = requests.get(
                    api_url,
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=120,
                )

                if resp.status_code == 200:
                    buf = io.BytesIO(resp.content)
                    buf.seek(0)
                    print(f"[file_loader] ✅ Downloaded via Files API ({len(resp.content):,} bytes)", flush=True)
                    return buf
                else:
                    print(f"[file_loader] Files API returned {resp.status_code}: {resp.text[:200]}", flush=True)
            except Exception as api_err:
                print(f"[file_loader] Files API download failed: {api_err}", flush=True)
        else:
            print("[file_loader] DATABRICKS_TOKEN or DATABRICKS_HOST not set — cannot download via API", flush=True)

        # Method 3: Databricks SDK fallback
        try:
            from databricks.sdk import WorkspaceClient

            w = WorkspaceClient()
            resp = w.files.download(normalized_path)
            buf = io.BytesIO(resp.contents.read())
            buf.seek(0)
            print(f"[file_loader] ✅ Downloaded via SDK ({buf.getbuffer().nbytes:,} bytes)", flush=True)
            return buf
        except ImportError:
            print("[file_loader] databricks-sdk not installed", flush=True)
        except Exception as sdk_err:
            print(f"[file_loader] SDK download also failed: {sdk_err}", flush=True)

    raise FileNotFoundError(f"File not found: {original_path}")


def _normalize_path(file_path: str) -> str:
    """Normalize file path for different Databricks path types."""
    if file_path.startswith("/Volumes/"):
        return file_path
    if file_path.startswith("/dbfs/"):
        return file_path
    elif file_path.startswith("/FileStore/") or file_path.startswith("/mnt/"):
        return f"/dbfs{file_path}"
    if file_path.startswith("/Workspace/"):
        return file_path
    return file_path

