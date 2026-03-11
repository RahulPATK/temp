"""
Server-wide configuration for TapeSmith MCP Server.

Reads from environment variables set in app.yaml. No hardcoded values.
"""

import os
from dataclasses import dataclass


@dataclass
class ServerConfig:
    """Configuration for the TapeSmith MCP server."""

    # Databricks connectivity
    databricks_host: str = ""
    databricks_token: str = ""
    sql_warehouse_id: str = ""

    # LLM
    databricks_model_name: str = "databricks-gpt-5-2"

    # Default schema (configured via env vars in app.yaml)
    default_schema_name: str = "nagy_clean"
    default_schema_path: str = ""  # MUST be set via TAPESMITH_SCHEMA_PATH env var

    # Catalog / Schema for Delta tables (configured via env vars in app.yaml)
    catalog: str = ""   # MUST be set via TAPESMITH_CATALOG env var
    db_schema: str = "" # MUST be set via TAPESMITH_DB_SCHEMA env var

    # Table names (staging + final) — computed in __post_init__
    staging_header_mapping: str = ""
    staging_value_transformation: str = ""
    staging_validation: str = ""
    staging_mcf_generation: str = ""
    final_header_mappings: str = ""
    final_value_transformations: str = ""
    final_validations: str = ""
    final_mcf_outputs: str = ""
    learned_corrections_table: str = ""
    user_feedback_table: str = ""

    def __post_init__(self):
        """Load from environment and compute table FQNs."""
        self.databricks_host = os.environ.get("DATABRICKS_HOST", self.databricks_host)
        self.databricks_token = os.environ.get("DATABRICKS_TOKEN", self.databricks_token)
        self.sql_warehouse_id = os.environ.get(
            "DATABRICKS_SQL_WAREHOUSE_ID",
            os.environ.get("SQL_WAREHOUSE_ID", self.sql_warehouse_id),
        )
        self.databricks_model_name = os.environ.get("DATABRICKS_MODEL_NAME", self.databricks_model_name)
        self.default_schema_name = os.environ.get("TAPESMITH_SCHEMA_NAME", self.default_schema_name)
        self.default_schema_path = os.environ.get("TAPESMITH_SCHEMA_PATH", self.default_schema_path)
        self.catalog = os.environ.get("TAPESMITH_CATALOG", self.catalog)
        self.db_schema = os.environ.get("TAPESMITH_DB_SCHEMA", self.db_schema)

        # Validate required settings
        missing = []
        if not self.catalog:
            missing.append("TAPESMITH_CATALOG")
        if not self.db_schema:
            missing.append("TAPESMITH_DB_SCHEMA")
        if not self.default_schema_path:
            missing.append("TAPESMITH_SCHEMA_PATH")
        if missing:
            print(f"[config] ⚠️  Missing required env vars: {', '.join(missing)}. "
                  f"Set them in app.yaml or environment.")

        prefix = f"{self.catalog}.{self.db_schema}"
        self.staging_header_mapping = f"{prefix}.header_mapping_proposals"
        self.staging_value_transformation = f"{prefix}.value_transformation_proposals"
        self.staging_validation = f"{prefix}.validation_results"
        self.staging_mcf_generation = f"{prefix}.mcf_generation_proposals"
        self.final_header_mappings = f"{prefix}.header_mappings"
        self.final_value_transformations = f"{prefix}.value_transformations"
        self.final_validations = f"{prefix}.validations"
        self.final_mcf_outputs = f"{prefix}.mcf_outputs"
        self.learned_corrections_table = f"{prefix}.learned_corrections"
        self.user_feedback_table = f"{prefix}.user_feedback"


# Singleton
_config: ServerConfig | None = None


def get_config() -> ServerConfig:
    global _config
    if _config is None:
        _config = ServerConfig()
    return _config

