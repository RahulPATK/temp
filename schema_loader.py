"""
Schema Loader Function

Loads schema definitions from JSON files.

Supports:
  - Direct file access (cluster with FUSE mount)
  - Files API download (model serving endpoints without FUSE)
"""

import json
import os
import tempfile
from typing import Dict, Optional


def _download_from_volumes_api(volumes_path: str) -> Optional[Dict]:
    """
    Download a JSON file from /Volumes/ via the Databricks Files API.
    Used on model serving endpoints that don't have FUSE mounts.
    """
    token = os.environ.get("DATABRICKS_TOKEN", "").strip()
    host = os.environ.get("DATABRICKS_HOST", "").strip().rstrip("/")

    if not token or not host:
        return None

    try:
        import requests

        api_url = f"{host}/api/2.0/fs/files{volumes_path}"
        resp = requests.get(
            api_url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=60,
        )

        if resp.status_code == 200:
            schema = resp.json()
            print(f"[schema_loader] Downloaded schema via Files API: {volumes_path}")
            return schema
        else:
            print(f"[schema_loader] Files API returned {resp.status_code} for {volumes_path}")
    except Exception as e:
        print(f"[schema_loader] Files API download failed: {e}")

    return None


def load_schema(schema_name: str = "nagy_clean", schema_path: Optional[str] = None) -> Dict:
    """
    Load schema from JSON file.
    
    Args:
        schema_name: Name of the schema file (without .json extension)
        schema_path: Optional path to schema directory. If None, uses TAPESMITH_SCHEMA_PATH env var.
    
    Returns:
        Dictionary containing schema definition
    
    Raises:
        FileNotFoundError: If schema file cannot be found
    """
    # Resolve schema path from arg → env var (no hardcoded fallback)
    if schema_path is None:
        schema_path = os.getenv("TAPESMITH_SCHEMA_PATH", "")

    if not schema_path:
        raise FileNotFoundError(
            f"Schema path not configured. Set TAPESMITH_SCHEMA_PATH env var "
            f"(e.g. /Volumes/<catalog>/<schema>/<volume>) or pass schema_path argument."
        )

    schema_file = f"{schema_path}/{schema_name}.json"
    
    # Try local FUSE path first (works on clusters)
    possible_paths = [
        schema_file,
        f"/dbfs/mnt{schema_file}",
    ]
    
    for path in possible_paths:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                schema = json.load(f)
                print(f"[schema_loader] Loaded {schema_name}.json from: {path}")
                return schema
        except (FileNotFoundError, OSError):
            continue
    
    # --- Fallback: Download via Files API (for serving endpoints / Apps) ---
    if schema_file.startswith("/Volumes/"):
        result = _download_from_volumes_api(schema_file)
        if result is not None:
            return result

    raise FileNotFoundError(
        f"Schema file not found: {schema_name}.json. "
        f"Tried paths: {possible_paths}. "
        f"Ensure TAPESMITH_SCHEMA_PATH is set correctly in app.yaml."
    )

