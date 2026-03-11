"""
Database utilities for TapeSmith MCP Server.

Provides:
  - execute_sql(): runs SQL on a Databricks SQL Warehouse (SDK or raw REST)
  - get_llm(): returns a ChatDatabricks LLM instance
  - get_past_corrections(): reads learned_corrections table
  - store_corrections(): writes new corrections + user_feedback
  - load_schema_safe(): loads a schema JSON from /Volumes/ (FUSE or Files API)
  - load_tape_file(): loads an Excel/CSV tape file (FUSE or Files API)
  - parse_feedback_with_regex(): extract corrections from natural-language feedback
  - generate_proposal_id(): deterministic proposal ID
"""

import io
import json
import os
import re
import time
import hashlib
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from server.config import get_config


# ═══════════════════════════════════════════════════════════════════════════
# SQL execution
# ═══════════════════════════════════════════════════════════════════════════

def execute_sql(sql: str, max_wait: int = 120) -> Optional[List[Dict[str, Any]]]:
    """
    Execute SQL on a Databricks SQL Warehouse.

    - Tries Spark first (cluster environment).
    - Falls back to Databricks SDK Statement Execution API.
    - SELECT queries poll until results arrive.
    - INSERT / UPDATE / CREATE / MERGE wait for completion (synchronous).
    """
    # --- Method 1: Spark (cluster only) ---
    try:
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()
        spark.sql("SELECT 1")
        rows = spark.sql(sql).collect()
        return [r.asDict() for r in rows]
    except Exception:
        pass

    # --- Method 2: Databricks SDK ---
    cfg = get_config()
    wid = cfg.sql_warehouse_id
    if not wid:
        raise RuntimeError(
            "No Spark and no DATABRICKS_SQL_WAREHOUSE_ID configured. "
            "Set it in app.yaml env."
        )

    from databricks.sdk import WorkspaceClient
    w = WorkspaceClient()

    stmt_upper = sql.strip().upper()
    is_write = stmt_upper.startswith(("INSERT", "UPDATE", "DELETE", "MERGE", "CREATE", "ALTER", "DROP"))

    # Writes use 10s wait (synchronous) so data is visible immediately.
    # SELECTs use 50s wait.
    response = w.statement_execution.execute_statement(
        warehouse_id=wid,
        statement=sql,
        wait_timeout="10s" if is_write else "50s",
    )

    state_val = "UNKNOWN"
    if response.status and response.status.state:
        state_val = response.status.state.value

    # Writes: ensure completion before returning
    if is_write:
        if state_val == "SUCCEEDED":
            return None
        # Still running — poll until done (writes must commit before we return)
        if state_val in ("PENDING", "RUNNING"):
            statement_id = response.statement_id
            for _ in range(6):  # poll up to 30s (6 × 5s)
                time.sleep(5)
                poll = w.statement_execution.get_statement(statement_id)
                poll_state = poll.status.state.value if poll.status and poll.status.state else "UNKNOWN"
                if poll_state == "SUCCEEDED":
                    return None
                if poll_state in ("FAILED", "CANCELED", "CLOSED"):
                    error_msg = str(poll.status.error) if poll.status and poll.status.error else ""
                    raise RuntimeError(f"SQL write failed ({poll_state}): {error_msg}")
            print(f"[execute_sql] ⚠️ Write still running after 30s, continuing (statement_id={statement_id})")
            return None
        if state_val in ("FAILED", "CANCELED", "CLOSED"):
            error_msg = str(response.status.error) if response.status and response.status.error else ""
            raise RuntimeError(f"SQL write failed ({state_val}): {error_msg}")
        return None

    # SELECTs
    if state_val == "SUCCEEDED":
        return _extract_rows_sdk(response)

    if state_val in ("PENDING", "RUNNING"):
        statement_id = response.statement_id
        for _ in range(int(max_wait / 5)):
            time.sleep(5)
            poll = w.statement_execution.get_statement(statement_id)
            poll_state = poll.status.state.value if poll.status and poll.status.state else "UNKNOWN"
            if poll_state == "SUCCEEDED":
                return _extract_rows_sdk(poll)
            if poll_state in ("FAILED", "CANCELED", "CLOSED"):
                error_msg = str(poll.status.error) if poll.status and poll.status.error else ""
                raise RuntimeError(f"SQL execution failed ({poll_state}): {error_msg}")
        raise RuntimeError(f"SQL query timed out after {max_wait}s")

    # FAILED / CANCELED / CLOSED
    error_msg = str(response.status.error) if response.status and response.status.error else ""
    raise RuntimeError(f"SQL execution failed ({state_val}): {error_msg}")


def _extract_rows_sdk(response) -> List[Dict[str, Any]]:
    """Extract list-of-dicts from a SDK Statement Execution response."""
    if (
        response.result
        and response.result.data_array
        and response.manifest
        and response.manifest.schema
    ):
        columns = [col.name for col in response.manifest.schema.columns]
        return [dict(zip(columns, row)) for row in response.result.data_array]
    return []


# ═══════════════════════════════════════════════════════════════════════════
# Table initialization
# ═══════════════════════════════════════════════════════════════════════════

def ensure_staging_tables():
    """Create staging tables if they don't exist, with the correct schema."""
    cfg = get_config()
    tables = {
        cfg.staging_header_mapping: """
            CREATE TABLE IF NOT EXISTS {table} (
                proposal_id STRING, file_path STRING, file_hash STRING,
                proposed_mapping STRING, llm_suggestions STRING,
                status STRING DEFAULT 'pending', created_by STRING, created_at TIMESTAMP,
                approved_by STRING, approved_at TIMESTAMP, comments STRING
            )""",
        cfg.staging_value_transformation: """
            CREATE TABLE IF NOT EXISTS {table} (
                proposal_id STRING, file_path STRING, header_mapping_proposal_id STRING,
                proposed_transformations STRING, unmapped_values STRING, llm_suggestions STRING,
                status STRING DEFAULT 'pending', created_by STRING, created_at TIMESTAMP,
                approved_by STRING, approved_at TIMESTAMP, comments STRING
            )""",
        cfg.staging_validation: """
            CREATE TABLE IF NOT EXISTS {table} (
                proposal_id STRING, file_path STRING, header_mapping_proposal_id STRING,
                value_transformation_proposal_id STRING, validation_results STRING,
                is_valid BOOLEAN, issues STRING,
                created_by STRING, created_at TIMESTAMP
            )""",
        cfg.staging_mcf_generation: """
            CREATE TABLE IF NOT EXISTS {table} (
                proposal_id STRING, file_path STRING, effective_date STRING,
                header_mapping_proposal_id STRING, value_transformation_proposal_id STRING,
                validation_proposal_id STRING, mcf_output STRING,
                status STRING, created_by STRING, created_at TIMESTAMP,
                approved_by STRING, approved_at TIMESTAMP, comments STRING
            )""",
    }

    for table_name, ddl_template in tables.items():
        try:
            try:
                execute_sql(f"SELECT proposal_id FROM {table_name} LIMIT 1")
                print(f"[ensure_staging_tables] ✅ {table_name} OK")
                continue
            except Exception as check_err:
                err_msg = str(check_err)
                if "UNRESOLVED_COLUMN" in err_msg:
                    print(f"[ensure_staging_tables] ⚠️ {table_name} has wrong schema, recreating...")
                    execute_sql(f"DROP TABLE IF EXISTS {table_name}")
                elif "TABLE_OR_VIEW_NOT_FOUND" in err_msg or "does not exist" in err_msg.lower():
                    print(f"[ensure_staging_tables] 📝 {table_name} does not exist, creating...")
                else:
                    print(f"[ensure_staging_tables] ⚠️ {table_name} check failed: {check_err}")

            execute_sql(ddl_template.format(table=table_name))
            print(f"[ensure_staging_tables] ✅ {table_name} created")
        except Exception as e:
            print(f"[ensure_staging_tables] ❌ {table_name}: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# LLM
# ═══════════════════════════════════════════════════════════════════════════

_llm_instance = None


def get_llm():
    """Return a ChatDatabricks LLM instance, cached as singleton."""
    global _llm_instance
    if _llm_instance is None:
        from databricks_langchain import ChatDatabricks
        cfg = get_config()
        _llm_instance = ChatDatabricks(
            endpoint=cfg.databricks_model_name,
            temperature=0.0,
        )
    return _llm_instance


# ═══════════════════════════════════════════════════════════════════════════
# Schema & file loading (FUSE first → Files API fallback)
# ═══════════════════════════════════════════════════════════════════════════

def load_schema_safe(schema_name: str = "", schema_path: str = "") -> Dict:
    """Load a schema JSON from /Volumes/ via FUSE, SDK, or raw REST."""
    cfg = get_config()
    name = schema_name or cfg.default_schema_name
    base_path = schema_path or cfg.default_schema_path

    if not name.endswith(".json"):
        name = f"{name}.json"

    errors = []

    # --- Method 1: FUSE (works on clusters) ---
    direct_path = f"{base_path}/{name}"
    if os.path.exists(direct_path):
        with open(direct_path, "r") as f:
            schema = json.load(f)
        print(f"[load_schema] Loaded {name} from FUSE: {direct_path}")
        return schema
    errors.append(f"FUSE: {direct_path} does not exist")

    # --- Method 2: Databricks SDK (works in Apps — auto-handles OAuth) ---
    try:
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
        volumes_file_path = f"{base_path}/{name}"
        print(f"[load_schema] Trying SDK download: {volumes_file_path}")
        resp = w.files.download(volumes_file_path)
        content = resp.contents.read()
        schema = json.loads(content)
        print(f"[load_schema] Loaded {name} via Databricks SDK")
        return schema
    except Exception as e:
        sdk_err = f"SDK: {type(e).__name__}: {e}"
        print(f"[load_schema] {sdk_err}")
        errors.append(sdk_err)

    # --- Method 3: Raw REST API (last resort) ---
    host = os.environ.get("DATABRICKS_HOST", cfg.databricks_host or "").rstrip("/")
    token = os.environ.get("DATABRICKS_TOKEN", cfg.databricks_token or "")
    if host and token:
        volumes_path = base_path.replace("/Volumes/", "")
        api_url = f"{host}/api/2.0/fs/files/Volumes/{volumes_path}/{name}"
        try:
            print(f"[load_schema] Trying REST API: {api_url}")
            resp = requests.get(api_url, headers={"Authorization": f"Bearer {token}"}, timeout=30)
            if resp.status_code == 200:
                schema = resp.json()
                print(f"[load_schema] Loaded {name} via REST API")
                return schema
            errors.append(f"REST: status={resp.status_code}, body={resp.text[:200]}")
        except Exception as e:
            errors.append(f"REST: {type(e).__name__}: {e}")
    else:
        errors.append(f"REST: skipped (host={'set' if host else 'EMPTY'}, token={'set' if token else 'EMPTY'})")

    all_errors = " | ".join(errors)
    raise FileNotFoundError(f"Schema file {name} not found. Errors: {all_errors}")


def load_tape_file(file_path: str) -> pd.DataFrame:
    """Load a tape file (Excel or CSV) from /Volumes/ via FUSE, SDK, or REST."""
    errors = []

    # --- Method 1: FUSE (works on clusters) ---
    if os.path.exists(file_path):
        if file_path.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file_path, engine="openpyxl")
        else:
            df = pd.read_csv(file_path)
        print(f"[load_tape_file] Loaded via FUSE: {df.shape}")
        return df
    errors.append(f"FUSE: {file_path} does not exist")

    # --- Method 2: Databricks SDK (works in Apps — auto-handles OAuth) ---
    try:
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
        print(f"[load_tape_file] Trying SDK download: {file_path}")
        resp = w.files.download(file_path)
        content = resp.contents.read()
        buf = io.BytesIO(content)
        buf.seek(0)
        if file_path.endswith((".xlsx", ".xls")):
            df = pd.read_excel(buf, engine="openpyxl")
        else:
            df = pd.read_csv(buf)
        print(f"[load_tape_file] Loaded via Databricks SDK: {df.shape}")
        return df
    except Exception as e:
        sdk_err = f"SDK: {type(e).__name__}: {e}"
        print(f"[load_tape_file] {sdk_err}")
        errors.append(sdk_err)

    # --- Method 3: Raw REST API (last resort) ---
    cfg = get_config()
    host = os.environ.get("DATABRICKS_HOST", cfg.databricks_host or "").rstrip("/")
    token = os.environ.get("DATABRICKS_TOKEN", cfg.databricks_token or "")
    if host and token and file_path.startswith("/Volumes/"):
        volumes_path = file_path.replace("/Volumes/", "")
        api_url = f"{host}/api/2.0/fs/files/Volumes/{volumes_path}"
        try:
            print(f"[load_tape_file] Trying REST API: {api_url}")
            resp = requests.get(api_url, headers={"Authorization": f"Bearer {token}"}, timeout=120, stream=True)
            if resp.status_code == 200:
                buf = io.BytesIO(resp.content)
                buf.seek(0)
                if file_path.endswith((".xlsx", ".xls")):
                    df = pd.read_excel(buf, engine="openpyxl")
                else:
                    df = pd.read_csv(buf)
                print(f"[load_tape_file] Loaded via REST API: {df.shape}")
                return df
            errors.append(f"REST: status={resp.status_code}")
        except Exception as e:
            errors.append(f"REST: {type(e).__name__}: {e}")
    else:
        errors.append(f"REST: skipped (host={'set' if host else 'EMPTY'}, token={'set' if token else 'EMPTY'})")

    all_errors = " | ".join(errors)
    raise FileNotFoundError(f"File not found: {file_path}. Errors: {all_errors}")


# ═══════════════════════════════════════════════════════════════════════════
# Corrections / feedback
# ═══════════════════════════════════════════════════════════════════════════

def get_past_corrections(stage: str) -> Dict[str, str]:
    """Read learned corrections for a stage."""
    cfg = get_config()
    table = cfg.learned_corrections_table
    try:
        rows = execute_sql(
            f"SELECT source_value, correct_value FROM {table} "
            f"WHERE stage = '{stage}' ORDER BY created_at DESC"
        )
        corrections = {}
        if rows:
            for row in rows:
                src = row.get("source_value", "").lower()
                if src and src not in corrections:
                    corrections[src] = row.get("correct_value", "")
            if corrections:
                print(f"[db_utils] Loaded {len(corrections)} learned correction(s) for stage '{stage}'")
        return corrections
    except Exception as e:
        print(f"[db_utils] Could not query learned_corrections: {e}")
        return {}


def _escape_for_sql(s: str) -> str:
    """Escape a string for safe embedding in a Databricks SQL single-quoted literal."""
    if s is None:
        return ""
    return str(s).replace("\\", "\\\\").replace("'", "''")


def store_corrections(stage: str, corrections: List[Dict[str, str]], created_by: str = "mcp_server"):
    """Store new corrections in the learned_corrections table."""
    cfg = get_config()
    table = cfg.learned_corrections_table
    for corr in corrections:
        src = _escape_for_sql(corr.get("source_value", ""))
        tgt = _escape_for_sql(corr.get("correct_value", ""))
        sql = (
            f"INSERT INTO {table} (stage, source_value, correct_value, created_by, created_at) "
            f"VALUES ('{stage}', '{src}', '{tgt}', '{_escape_for_sql(created_by)}', current_timestamp())"
        )
        try:
            execute_sql(sql)
        except Exception as e:
            print(f"[db_utils] Failed to store correction: {e}")


def parse_feedback_with_regex(feedback_text: str) -> List[Dict]:
    """Parse natural language feedback into structured corrections via regex."""
    corrections = []
    patterns = [
        r"['\"]([^'\"]+)['\"].*?(?:should (?:map|be|become)|→|->).*?['\"]([^'\"]+)['\"]",
        r"(\S+)\s+(?:should (?:map|be|become)|→|->)\s+(\S+)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, feedback_text, re.IGNORECASE):
            corrections.append({
                "source_value": match.group(1).strip(),
                "correct_value": match.group(2).strip(),
            })
        if corrections:
            break
    return corrections


# ═══════════════════════════════════════════════════════════════════════════
# Proposal ID
# ═══════════════════════════════════════════════════════════════════════════

def generate_proposal_id(file_path: str, stage: str) -> str:
    """Generate a unique-ish proposal ID."""
    raw = f"{file_path}:{stage}:{datetime.now().isoformat()}"
    return hashlib.md5(raw.encode()).hexdigest()

