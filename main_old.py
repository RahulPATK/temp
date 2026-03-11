"""
TapeSmith MCP Server — main entry point.

Exposes the 4-stage mortgage tape processing pipeline + CRUD proposal tools
as MCP tools on a Databricks App via FastMCP (streamable-http transport).

Tools:
  1. propose_header_mapping    — Stage 1: map tape headers → standard schema
  2. propose_value_transformation — Stage 2: transform column values
  3. validate_data             — Stage 3: validate processed data
  4. propose_mcf_generation    — Stage 4: generate MCF output
  5. approve_proposal          — Approve a pending proposal
  6. reject_proposal           — Reject a proposal with feedback
  7. get_proposal              — Retrieve proposal details
  8. list_proposals            — List proposals by stage / status
"""

import json
import sys
import os
import traceback
from typing import Optional

# Ensure the project root is on sys.path so "processing" and "server" are importable
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from mcp.server.fastmcp import FastMCP

from server.config import get_config
from server.db_utils import (
    execute_sql,
    get_llm,
    load_schema_safe,
    load_tape_file,
    get_past_corrections,
    store_corrections,
    generate_proposal_id,
    ensure_staging_tables,
)

# ---------------------------------------------------------------------------
# FastMCP app
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "TapeSmith",
    instructions=(
        "TapeSmith processes mortgage tape files through a 4-stage pipeline: "
        "1) Header Mapping, 2) Value Transformation, 3) Validation, 4) MCF Generation.\n\n"
        "CRITICAL RULES:\n"
        "- After calling ANY propose_* tool, you MUST STOP and show the results to the user. "
        "NEVER call approve_proposal or reject_proposal unless the user EXPLICITLY says 'approve' or 'reject'.\n"
        "- Run stages ONE AT A TIME. After each propose_* call, wait for the user to approve before moving on.\n"
        "- NEVER auto-approve. NEVER skip stages. NEVER chain multiple tool calls in one turn.\n"
        "- If a tool returns an error, report it to the user and stop. Do NOT retry the same tool.\n\n"
        "Workflow:\n"
        "1. User says 'Process <file>' → call propose_header_mapping → STOP, show results, wait for user.\n"
        "2. User says 'approve' → call approve_proposal for header_mapping → then call propose_value_transformation → STOP, wait.\n"
        "3. User says 'approve' → call approve_proposal for value_transformation → then call validate_data → STOP, wait.\n"
        "4. User says 'approve' → call approve_proposal for validation → then call propose_mcf_generation → STOP, wait.\n"
        "5. User says 'approve' → call approve_proposal for mcf_generation → DONE."
    ),
)


# ═══════════════════════════════════════════════════════════════════════════
# TOOL 1: propose_header_mapping
# ═══════════════════════════════════════════════════════════════════════════

@mcp.tool()
def propose_header_mapping(
    file_path: str,
    schema_name: str = "",
    schema_path: str = "",
    effective_date: str = "20250101",
    created_by: str = "mcp_user",
) -> str:
    """
    Stage 1 — Header Mapping.

    Loads a tape file and maps its column headers to standard schema headers.
    Returns a JSON proposal with mapped headers for review.

    Args:
        file_path: Path to the tape file on /Volumes/ (e.g. /Volumes/catalog/schema/volume/file.xlsx)
        schema_name: Name of the schema JSON (default: configured server-side)
        schema_path: Base path for schema files (default: configured server-side)
        effective_date: Effective date for the tape (YYYYMMDD)
        created_by: User or system creating the proposal
    """
    try:
        cfg = get_config()
        schema = load_schema_safe(schema_name, schema_path)
        tape_df = load_tape_file(file_path)
        llm = get_llm()

        from processing.agents import HeaderMappingAgent
        from processing.data_processor import sanitize_df

        tape_df = sanitize_df(tape_df)

        # Learned corrections
        learned = get_past_corrections("header_mapping")
        corrections_context = ""
        if learned:
            corrections_context = "Past user corrections:\n" + "\n".join(
                f"  '{k}' should map to '{v}'" for k, v in learned.items()
            )

        agent = HeaderMappingAgent(cfg, schema, llm)
        result = agent.execute({
            "tape_df": tape_df,
            "learned_corrections": learned,
            "corrections_context": corrections_context,
        })

        proposal_id = generate_proposal_id(file_path, "header_mapping")

        # Mapping preview
        mapped = result.get("mapped_headers", {})
        preview_lines = [f"  {k} → {v}" for k, v in sorted(mapped.items())[:15]]
        if len(mapped) > 15:
            preview_lines.append(f"  ... and {len(mapped) - 15} more")
        mapping_preview = "\n".join(preview_lines)

        # Complete mapping for full display
        complete_mapping = "\n".join(
            f"  {i+1}. {k} → {v}" for i, (k, v) in enumerate(sorted(mapped.items()))
        )

        # Write to staging table
        mapping_json = _escape_for_sql(json.dumps(mapped))
        suggestions_json = _escape_for_sql(json.dumps(result.get("suggestions", {})))
        sql = (
            f"INSERT INTO {cfg.staging_header_mapping} "
            f"(proposal_id, file_path, proposed_mapping, llm_suggestions, status, created_by, created_at) "
            f"VALUES ('{proposal_id}', '{_escape_for_sql(file_path)}', "
            f"'{mapping_json}', '{suggestions_json}', 'pending', '{_escape_for_sql(created_by)}', current_timestamp())"
        )
        try:
            execute_sql(sql)
            print(f"[propose_header_mapping] ✅ Staging write succeeded for {proposal_id}")
        except Exception as write_err:
            print(f"[propose_header_mapping] ❌ Staging write failed: {write_err}")
            return json.dumps({
                "error": f"Header mapping processed but staging write FAILED: {write_err}",
                "proposal_id": proposal_id,
                "stage": "header_mapping",
                "total_headers_mapped": len(mapped),
                "staging_write_error": str(write_err),
            }, indent=2)

        return json.dumps({
            "proposal_id": proposal_id,
            "file_path": file_path,
            "stage": "header_mapping",
            "total_headers_mapped": len(mapped),
            "llm_suggestions_count": len(result.get("suggestions", {})),
            "mapping_preview": mapping_preview,
            "complete_mapping": complete_mapping,
            "status": "pending",
            "message": (
                "Header mapping proposal created. Review the mapping above. "
                "Say 'approve <proposal_id> for header_mapping' to approve, or "
                "'reject <proposal_id> for header_mapping because <reason>' to reject."
            ),
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "traceback": traceback.format_exc(),
            "stage": "header_mapping",
        }, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
# TOOL 2: propose_value_transformation
# ═══════════════════════════════════════════════════════════════════════════

@mcp.tool()
def propose_value_transformation(
    file_path: str,
    header_mapping_proposal_id: str,
    schema_name: str = "",
    schema_path: str = "",
    created_by: str = "mcp_user",
) -> str:
    """
    Stage 2 — Value Transformation.

    Reads the approved header mapping, loads the tape, applies header renaming,
    then transforms column values (rule-based + LLM fallback).

    Args:
        file_path: Path to the tape file on /Volumes/
        header_mapping_proposal_id: Proposal ID of the approved header mapping
        schema_name: Schema name
        schema_path: Schema base path
        created_by: User or system creating the proposal
    """
    try:
        cfg = get_config()
        schema = load_schema_safe(schema_name, schema_path)
        tape_df = load_tape_file(file_path)
        llm = get_llm()

        from processing.agents import ValueTransformationAgent
        from processing.data_processor import sanitize_df

        tape_df = sanitize_df(tape_df)

        # Get approved header mapping
        rows = execute_sql(
            f"SELECT proposed_mapping FROM {cfg.staging_header_mapping} "
            f"WHERE proposal_id = '{header_mapping_proposal_id}' AND status = 'approved'"
        )
        if not rows:
            # Also try final table
            rows = execute_sql(
                f"SELECT mapping AS proposed_mapping FROM {cfg.final_header_mappings} "
                f"WHERE proposal_id = '{header_mapping_proposal_id}'"
            )
        if not rows:
            return json.dumps({"error": f"No approved header mapping found for {header_mapping_proposal_id}"})

        mapping = json.loads(rows[0]["proposed_mapping"])

        # Rename columns
        rename_map = {}
        for orig_col in tape_df.columns:
            mapped = mapping.get(orig_col.lower())
            if mapped:
                rename_map[orig_col] = mapped
        tape_df = tape_df.rename(columns=rename_map)

        # Learned corrections
        learned = get_past_corrections("value_transformation")
        corrections_context = ""
        if learned:
            corrections_context = "Past user corrections:\n" + "\n".join(
                f"  '{k}' should become '{v}'" for k, v in learned.items()
            )

        agent = ValueTransformationAgent(cfg, schema, llm)
        result = agent.execute({
            "tape_df": tape_df,
            "learned_corrections": learned,
            "corrections_context": corrections_context,
        })

        proposal_id = generate_proposal_id(file_path, "value_transformation")

        transformations_json = _escape_for_sql(json.dumps({"transformations_applied": result.get("transformations_applied", False)}))
        unmapped_json = _escape_for_sql(json.dumps(result.get("unmapped_values_detected", 0)))
        llm_suggestions_json = _escape_for_sql(json.dumps(result.get("llm_suggestions", {})))

        sql = (
            f"INSERT INTO {cfg.staging_value_transformation} "
            f"(proposal_id, file_path, header_mapping_proposal_id, proposed_transformations, "
            f"unmapped_values, llm_suggestions, status, created_by, created_at) "
            f"VALUES ('{proposal_id}', '{_escape_for_sql(file_path)}', '{header_mapping_proposal_id}', "
            f"'{transformations_json}', '{unmapped_json}', '{llm_suggestions_json}', "
            f"'pending', '{_escape_for_sql(created_by)}', current_timestamp())"
        )
        try:
            execute_sql(sql)
            print(f"[propose_value_transformation] ✅ Staging write succeeded for {proposal_id}")
        except Exception as write_err:
            print(f"[propose_value_transformation] ❌ Staging write failed: {write_err}")
            return json.dumps({
                "error": f"Value transformation processed but staging write FAILED: {write_err}",
                "proposal_id": proposal_id,
                "stage": "value_transformation",
                "staging_write_error": str(write_err),
            }, indent=2)

        return json.dumps({
            "proposal_id": proposal_id,
            "file_path": file_path,
            "stage": "value_transformation",
            "header_mapping_proposal_id": header_mapping_proposal_id,
            "transformations_applied": result.get("transformations_applied", False),
            "unmapped_values_detected": result.get("unmapped_values_detected", 0),
            "llm_called": result.get("llm_called", False),
            "status": "pending",
            "message": (
                "Value transformation proposal created. "
                "Say 'approve <proposal_id> for value_transformation' to approve."
            ),
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "traceback": traceback.format_exc(),
            "stage": "value_transformation",
        }, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
# TOOL 3: validate_data
# ═══════════════════════════════════════════════════════════════════════════

@mcp.tool()
def validate_data(
    file_path: str,
    header_mapping_proposal_id: str,
    schema_name: str = "",
    schema_path: str = "",
    created_by: str = "mcp_user",
) -> str:
    """
    Stage 3 — Validation.

    Loads the tape, applies approved header mapping and data type processing,
    then validates the data against the schema.

    Args:
        file_path: Path to the tape file on /Volumes/
        header_mapping_proposal_id: Proposal ID of the approved header mapping
        schema_name: Schema name
        schema_path: Schema base path
        created_by: User or system creating the proposal
    """
    try:
        cfg = get_config()
        schema = load_schema_safe(schema_name, schema_path)
        tape_df = load_tape_file(file_path)
        llm = get_llm()

        from processing.agents import ValidationAgent
        from processing.data_processor import sanitize_df, process_data_types

        tape_df = sanitize_df(tape_df)

        # Get approved header mapping
        rows = execute_sql(
            f"SELECT proposed_mapping FROM {cfg.staging_header_mapping} "
            f"WHERE proposal_id = '{header_mapping_proposal_id}' AND status = 'approved'"
        )
        if not rows:
            rows = execute_sql(
                f"SELECT mapping AS proposed_mapping FROM {cfg.final_header_mappings} "
                f"WHERE proposal_id = '{header_mapping_proposal_id}'"
            )
        if not rows:
            return json.dumps({"error": f"No approved header mapping found for {header_mapping_proposal_id}"})

        mapping = json.loads(rows[0]["proposed_mapping"])
        rename_map = {orig: mapped for orig in tape_df.columns for k, mapped in mapping.items() if orig.lower() == k}
        tape_df = tape_df.rename(columns=rename_map)

        # Process data types
        if process_data_types:
            tape_df = process_data_types(tape_df, schema)

        agent = ValidationAgent(cfg, schema, llm)
        result = agent.execute({"tape_df": tape_df})

        proposal_id = generate_proposal_id(file_path, "validation")
        validation_json = _escape_for_sql(json.dumps(result.get("validation_results", {})))

        sql = (
            f"INSERT INTO {cfg.staging_validation} "
            f"(proposal_id, file_path, header_mapping_proposal_id, "
            f"validation_results, is_valid, created_by, created_at) "
            f"VALUES ('{proposal_id}', '{_escape_for_sql(file_path)}', '{header_mapping_proposal_id}', "
            f"'{validation_json}', {result.get('is_valid', False)}, "
            f"'{_escape_for_sql(created_by)}', current_timestamp())"
        )
        try:
            execute_sql(sql)
            print(f"[validate_data] ✅ Staging write succeeded for {proposal_id}")
        except Exception as write_err:
            print(f"[validate_data] ❌ Staging write failed: {write_err}")
            return json.dumps({
                "error": f"Validation processed but staging write FAILED: {write_err}",
                "proposal_id": proposal_id,
                "stage": "validation",
                "staging_write_error": str(write_err),
            }, indent=2)

        vr = result.get("validation_results", {})
        return json.dumps({
            "proposal_id": proposal_id,
            "file_path": file_path,
            "stage": "validation",
            "header_mapping_proposal_id": header_mapping_proposal_id,
            "total_rows": vr.get("total_rows", 0),
            "total_columns": vr.get("total_columns", 0),
            "missing_required_fields": len(vr.get("missing_required_fields", [])),
            "data_quality_issues": len(vr.get("data_quality_issues", [])),
            "is_valid": result.get("is_valid", False),
            "status": "pending",
            "message": (
                "Validation complete. "
                "Say 'approve <proposal_id> for validation' to approve."
            ),
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "traceback": traceback.format_exc(),
            "stage": "validation",
        }, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
# TOOL 4: propose_mcf_generation
# ═══════════════════════════════════════════════════════════════════════════

@mcp.tool()
def propose_mcf_generation(
    file_path: str,
    header_mapping_proposal_id: str,
    validation_proposal_id: str = "",
    effective_date: str = "20250101",
    format_type: str = "COL",
    schema_name: str = "",
    schema_path: str = "",
    created_by: str = "mcp_user",
) -> str:
    """
    Stage 4 — MCF Generation.

    Loads tape, applies approved header mapping, processes data types,
    transforms values, and generates the MCF output.

    Args:
        file_path: Path to the tape file on /Volumes/
        header_mapping_proposal_id: Proposal ID of the approved header mapping
        validation_proposal_id: (Optional) Proposal ID of the validation step
        effective_date: Effective date (YYYYMMDD)
        format_type: MCF format (COL or JSON)
        schema_name: Schema name
        schema_path: Schema base path
        created_by: User or system creating the proposal
    """
    try:
        cfg = get_config()
        schema = load_schema_safe(schema_name, schema_path)
        tape_df = load_tape_file(file_path)
        llm = get_llm()

        from processing.agents import MCFGenerationAgent
        from processing.data_processor import sanitize_df, process_data_types, transform_values

        tape_df = sanitize_df(tape_df)

        # Get approved header mapping
        rows = execute_sql(
            f"SELECT proposed_mapping FROM {cfg.staging_header_mapping} "
            f"WHERE proposal_id = '{header_mapping_proposal_id}' AND status = 'approved'"
        )
        if not rows:
            rows = execute_sql(
                f"SELECT mapping AS proposed_mapping FROM {cfg.final_header_mappings} "
                f"WHERE proposal_id = '{header_mapping_proposal_id}'"
            )
        if not rows:
            return json.dumps({"error": f"No approved header mapping found for {header_mapping_proposal_id}"})

        mapping = json.loads(rows[0]["proposed_mapping"])
        rename_map = {orig: mapped for orig in tape_df.columns for k, mapped in mapping.items() if orig.lower() == k}
        tape_df = tape_df.rename(columns=rename_map)

        # Process data types & transform values
        if process_data_types:
            tape_df = process_data_types(tape_df, schema)
        if transform_values:
            tape_df = transform_values(tape_df, schema)

        # Learned corrections for penalty strings
        learned = get_past_corrections("mcf_generation")
        corrections_context = ""
        if learned:
            corrections_context = "Past user corrections:\n" + "\n".join(
                f"  '{k}' → '{v}'" for k, v in learned.items()
            )

        agent = MCFGenerationAgent(cfg, llm, schema)
        result = agent.execute({
            "tape_df": tape_df,
            "effective_date": effective_date,
            "format": format_type,
            "learned_corrections": learned,
            "corrections_context": corrections_context,
        })

        if result.get("status") == "error":
            return json.dumps({
                "error": result.get("error", "MCF generation failed"),
                "stage": "mcf_generation",
            }, indent=2)

        proposal_id = generate_proposal_id(file_path, "mcf_generation")
        mcf_output = result.get("mcf_output", "")

        mcf_escaped = _escape_for_sql(mcf_output)
        sql = (
            f"INSERT INTO {cfg.staging_mcf_generation} "
            f"(proposal_id, file_path, effective_date, header_mapping_proposal_id, "
            f"validation_proposal_id, mcf_output, status, created_by, created_at) "
            f"VALUES ('{proposal_id}', '{_escape_for_sql(file_path)}', '{effective_date}', "
            f"'{header_mapping_proposal_id}', '{validation_proposal_id}', "
            f"'{mcf_escaped}', 'pending', '{_escape_for_sql(created_by)}', current_timestamp())"
        )
        try:
            execute_sql(sql)
            print(f"[propose_mcf_generation] ✅ Staging write succeeded for {proposal_id}")
        except Exception as write_err:
            print(f"[propose_mcf_generation] ❌ Staging write failed: {write_err}")
            return json.dumps({
                "error": f"MCF generation processed but staging write FAILED: {write_err}",
                "proposal_id": proposal_id,
                "stage": "mcf_generation",
                "staging_write_error": str(write_err),
            }, indent=2)

        # MCF preview
        mcf_preview = mcf_output[:500] if mcf_output else ""

        return json.dumps({
            "proposal_id": proposal_id,
            "file_path": file_path,
            "stage": "mcf_generation",
            "effective_date": effective_date,
            "format": format_type,
            "penalty_llm_called": result.get("penalty_llm_called", False),
            "mcf_output_total_length": len(mcf_output),
            "mcf_preview": mcf_preview,
            "header_mapping_proposal_id": header_mapping_proposal_id,
            "validation_proposal_id": validation_proposal_id,
            "status": "pending",
            "message": (
                "MCF generation proposal created. Review the MCF output preview. "
                "Say 'approve <proposal_id> for mcf_generation' to persist, or "
                "'reject <proposal_id> for mcf_generation because <reason>' to reject."
            ),
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "traceback": traceback.format_exc(),
            "stage": "mcf_generation",
        }, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
# TOOL 5: approve_proposal
# ═══════════════════════════════════════════════════════════════════════════

@mcp.tool()
def approve_proposal(proposal_id: str, stage: str) -> str:
    """
    Approve a pending proposal and persist it to the final table.

    Args:
        proposal_id: The proposal ID to approve
        stage: One of: header_mapping, value_transformation, validation, mcf_generation
    """
    try:
        cfg = get_config()
        staging_table = _stage_to_staging_table(stage, cfg)
        final_table = _stage_to_final_table(stage, cfg)
        if not staging_table:
            return json.dumps({"error": f"Unknown stage: {stage}"})

        # Mark as approved in staging (validation_results table has no status column)
        if stage != "validation":
            execute_sql(
                f"UPDATE {staging_table} SET status = 'approved', "
                f"approved_by = 'mcp_user', approved_at = current_timestamp() "
                f"WHERE proposal_id = '{proposal_id}'"
            )

        # Copy to final table
        rows = execute_sql(f"SELECT * FROM {staging_table} WHERE proposal_id = '{proposal_id}'")
        if not rows:
            return json.dumps({"error": f"Proposal {proposal_id} not found in staging"})

        row = rows[0]
        # Build INSERT for final table based on stage
        _persist_to_final(stage, row, final_table, cfg)

        return json.dumps({
            "proposal_id": proposal_id,
            "stage": stage,
            "status": "approved",
            "message": f"Proposal {proposal_id} approved and persisted to {final_table}.",
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e), "traceback": traceback.format_exc()}, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
# TOOL 6: reject_proposal
# ═══════════════════════════════════════════════════════════════════════════

@mcp.tool()
def reject_proposal(proposal_id: str, stage: str, feedback: str) -> str:
    """
    Reject a proposal with feedback. Corrections are extracted and stored
    for future runs.

    Args:
        proposal_id: The proposal ID to reject
        stage: One of: header_mapping, value_transformation, validation, mcf_generation
        feedback: Natural language feedback (e.g. "'total borrowers' should map to 'number_of_borrowers'")
    """
    try:
        cfg = get_config()
        staging_table = _stage_to_staging_table(stage, cfg)
        if not staging_table:
            return json.dumps({"error": f"Unknown stage: {stage}"})

        # Mark as rejected
        escaped_feedback = _escape_for_sql(feedback)
        execute_sql(
            f"UPDATE {staging_table} SET status = 'rejected', "
            f"comments = '{escaped_feedback}' "
            f"WHERE proposal_id = '{proposal_id}'"
        )

        # Parse corrections from feedback using regex
        import re
        corrections = []
        patterns = [
            r"['\"]([^'\"]+)['\"].*?(?:should (?:map|be|become)|→|->).*?['\"]([^'\"]+)['\"]",
            r"(\S+)\s+(?:should (?:map|be|become)|→|->)\s+(\S+)",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, feedback, re.IGNORECASE)
            for src, tgt in matches:
                corrections.append({"source_value": src.strip(), "correct_value": tgt.strip()})
            if corrections:
                break

        if corrections:
            store_corrections(stage, corrections, created_by="mcp_user")

        return json.dumps({
            "proposal_id": proposal_id,
            "stage": stage,
            "status": "rejected",
            "feedback": feedback,
            "corrections_stored": len(corrections),
            "message": (
                f"Proposal {proposal_id} rejected. {len(corrections)} correction(s) stored. "
                f"Re-run the propose step to apply corrections."
            ),
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e), "traceback": traceback.format_exc()}, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
# TOOL 7: get_proposal
# ═══════════════════════════════════════════════════════════════════════════

@mcp.tool()
def get_proposal(proposal_id: str, stage: str) -> str:
    """
    Retrieve details of a specific proposal.

    Args:
        proposal_id: The proposal ID
        stage: One of: header_mapping, value_transformation, validation, mcf_generation
    """
    try:
        cfg = get_config()
        staging_table = _stage_to_staging_table(stage, cfg)
        if not staging_table:
            return json.dumps({"error": f"Unknown stage: {stage}"})

        rows = execute_sql(f"SELECT * FROM {staging_table} WHERE proposal_id = '{proposal_id}'")
        if not rows:
            return json.dumps({"error": f"Proposal {proposal_id} not found", "stage": stage})

        return json.dumps({"status": "found", "proposal": rows[0], "stage": stage}, indent=2, default=str)

    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
# TOOL 8: list_proposals
# ═══════════════════════════════════════════════════════════════════════════

@mcp.tool()
def list_proposals(stage: str, status: str = "", limit: int = 10) -> str:
    """
    List proposals for a given stage, optionally filtered by status.

    Args:
        stage: One of: header_mapping, value_transformation, validation, mcf_generation
        status: Filter by status (pending/approved/rejected). Empty = all.
        limit: Max rows to return (default 10)
    """
    try:
        cfg = get_config()
        staging_table = _stage_to_staging_table(stage, cfg)
        if not staging_table:
            return json.dumps({"error": f"Unknown stage: {stage}"})

        where = f"WHERE status = '{status}'" if status else ""
        rows = execute_sql(
            f"SELECT proposal_id, file_path, status, created_by, created_at "
            f"FROM {staging_table} {where} ORDER BY created_at DESC LIMIT {limit}"
        )

        return json.dumps({"stage": stage, "count": len(rows or []), "proposals": rows or []}, indent=2, default=str)

    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _escape_for_sql(s) -> str:
    """Escape a string for safe embedding in a Databricks SQL single-quoted literal."""
    if s is None:
        return ""
    return str(s).replace("\\", "\\\\").replace("'", "''")


def _stage_to_staging_table(stage: str, cfg) -> Optional[str]:
    return {
        "header_mapping": cfg.staging_header_mapping,
        "value_transformation": cfg.staging_value_transformation,
        "validation": cfg.staging_validation,
        "mcf_generation": cfg.staging_mcf_generation,
    }.get(stage)


def _stage_to_final_table(stage: str, cfg) -> Optional[str]:
    return {
        "header_mapping": cfg.final_header_mappings,
        "value_transformation": cfg.final_value_transformations,
        "validation": cfg.final_validations,
        "mcf_generation": cfg.final_mcf_outputs,
    }.get(stage)


def _persist_to_final(stage: str, row: dict, final_table: str, cfg):
    """Copy an approved proposal row to its final table.

    Final table schemas (from create_all_tables.sql):
      header_mappings:        proposal_id, file_path, file_hash, mapping, applied_by, applied_at
      value_transformations:  proposal_id, file_path, header_mapping_proposal_id, transformations, applied_by, applied_at
      validations:            (no separate final table — staging is the record)
      mcf_outputs:            proposal_id, file_path, effective_date, mcf_output, applied_by, applied_at
    """
    proposal_id = row.get("proposal_id", "")

    if stage == "header_mapping":
        file_path = _escape_for_sql(row.get("file_path", ""))
        file_hash = _escape_for_sql(row.get("file_hash", ""))
        mapping = _escape_for_sql(row.get("proposed_mapping", "{}"))
        execute_sql(
            f"INSERT INTO {final_table} "
            f"(proposal_id, file_path, file_hash, mapping, applied_by, applied_at) "
            f"VALUES ('{proposal_id}', '{file_path}', '{file_hash}', '{mapping}', "
            f"'mcp_user', current_timestamp())"
        )
    elif stage == "value_transformation":
        file_path = _escape_for_sql(row.get("file_path", ""))
        hm_id = row.get("header_mapping_proposal_id", "")
        transformations = _escape_for_sql(row.get("proposed_transformations", "{}"))
        execute_sql(
            f"INSERT INTO {final_table} "
            f"(proposal_id, file_path, header_mapping_proposal_id, transformations, applied_by, applied_at) "
            f"VALUES ('{proposal_id}', '{file_path}', '{hm_id}', '{transformations}', "
            f"'mcp_user', current_timestamp())"
        )
    elif stage == "validation":
        # Validation has no separate final table — the staging table IS the record.
        print(f"[_persist_to_final] Validation has no final table; staging record is sufficient.")
    elif stage == "mcf_generation":
        file_path = _escape_for_sql(row.get("file_path", ""))
        eff_date = row.get("effective_date", "")
        mcf = _escape_for_sql(row.get("mcf_output", ""))
        execute_sql(
            f"INSERT INTO {final_table} "
            f"(proposal_id, file_path, effective_date, mcf_output, applied_by, applied_at) "
            f"VALUES ('{proposal_id}', '{file_path}', '{eff_date}', '{mcf}', "
            f"'mcp_user', current_timestamp())"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Start the MCP server with streamable-http transport for Databricks Apps."""
    print("[TapeSmith] Ensuring staging tables exist...")
    try:
        ensure_staging_tables()
        print("[TapeSmith] ✅ Staging tables verified")
    except Exception as e:
        print(f"[TapeSmith] ⚠️ Staging table check failed (will retry on first tool call): {e}")
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()

