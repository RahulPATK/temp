# Databricks notebook source
# MAGIC %md
# MAGIC # TapeSmith MCP Server — Deploy & Test
# MAGIC
# MAGIC This notebook walks through:
# MAGIC 1. Creating all required Delta tables
# MAGIC 2. Verifying file access to schema + tape files
# MAGIC 3. Deploying the Databricks App
# MAGIC 4. Testing the MCP tools locally (on-cluster)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Customer Configuration
# MAGIC **Update all values below for your environment.**

# COMMAND ----------

# MAGIC %sql
# MAGIC GRANT USE CATALOG ON CATALOG rahul_pathak TO `db37c842-c5f5-4038-8fba-100cbb261c88`;
# MAGIC GRANT USE SCHEMA ON SCHEMA rahul_pathak.nl_example TO `db37c842-c5f5-4038-8fba-100cbb261c88`;
# MAGIC GRANT SELECT, MODIFY ON SCHEMA rahul_pathak.nl_example TO `db37c842-c5f5-4038-8fba-100cbb261c88`;
# MAGIC GRANT READ VOLUME ON VOLUME rahul_pathak.nl_example.schemafiletapesmith TO `db37c842-c5f5-4038-8fba-100cbb261c88`;
# MAGIC GRANT READ VOLUME ON VOLUME rahul_pathak.nl_example.tapesmithtapes TO `db37c842-c5f5-4038-8fba-100cbb261c88`;

# COMMAND ----------

# --- CUSTOMER CONFIGURATION ---
# Update these for your Databricks workspace

CATALOG = "<YOUR_UNITY_CATALOG>"            # e.g. "main"
DB_SCHEMA = "<YOUR_UNITY_SCHEMA>"           # e.g. "tapesmith"

SCHEMA_PATH = "/Volumes/<catalog>/<schema>/<volume>/schemafiletapesmith"  # Path to schema JSON files
SCHEMA_NAME = "nagy_clean"                   # Schema file name (without .json)

FILE_PATH = "/Volumes/<catalog>/<schema>/<volume>/tapesmithtapes/synth_tape_1.xlsx"  # Test tape file

SQL_WAREHOUSE_ID = "<YOUR_SQL_WAREHOUSE_ID>" # e.g. "148ccb90800933a1"
LLM_MODEL_NAME = "databricks-gpt-5-2"       # Foundation model endpoint

WORKSPACE_BASE = "/Workspace/Users/<your-email>/tapesmith_customer_deploy"  # Where you uploaded this folder

# --- END CUSTOMER CONFIGURATION ---

print(f"Catalog:          {CATALOG}")
print(f"Schema:           {DB_SCHEMA}")
print(f"Schema path:      {SCHEMA_PATH}")
print(f"Schema name:      {SCHEMA_NAME}")
print(f"Tape file:        {FILE_PATH}")
print(f"SQL Warehouse:    {SQL_WAREHOUSE_ID}")
print(f"LLM Model:        {LLM_MODEL_NAME}")
print(f"Workspace base:   {WORKSPACE_BASE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Create Delta Tables
# MAGIC Run the SQL script to create all required tables.

# COMMAND ----------

prefix = f"{CATALOG}.{DB_SCHEMA}"

table_ddls = [
    # Staging tables
    f"""CREATE TABLE IF NOT EXISTS {prefix}.header_mapping_proposals (
        proposal_id STRING, file_path STRING, file_hash STRING,
        proposed_mapping STRING, llm_suggestions STRING,
        status STRING DEFAULT 'pending', created_by STRING, created_at TIMESTAMP,
        approved_by STRING, approved_at TIMESTAMP, comments STRING
    )""",
    f"""CREATE TABLE IF NOT EXISTS {prefix}.value_transformation_proposals (
        proposal_id STRING, file_path STRING, header_mapping_proposal_id STRING,
        proposed_transformations STRING, unmapped_values STRING, llm_suggestions STRING,
        status STRING DEFAULT 'pending', created_by STRING, created_at TIMESTAMP,
        approved_by STRING, approved_at TIMESTAMP, comments STRING
    )""",
    f"""CREATE TABLE IF NOT EXISTS {prefix}.validation_results (
        proposal_id STRING, file_path STRING, header_mapping_proposal_id STRING,
        value_transformation_proposal_id STRING, validation_results STRING,
        is_valid BOOLEAN, issues STRING,
        created_by STRING, created_at TIMESTAMP
    )""",
    f"""CREATE TABLE IF NOT EXISTS {prefix}.mcf_generation_proposals (
        proposal_id STRING, file_path STRING, effective_date STRING,
        header_mapping_proposal_id STRING, value_transformation_proposal_id STRING,
        validation_proposal_id STRING, mcf_output STRING,
        status STRING, created_by STRING, created_at TIMESTAMP,
        approved_by STRING, approved_at TIMESTAMP, comments STRING
    )""",
    # Final tables
    f"""CREATE TABLE IF NOT EXISTS {prefix}.header_mappings (
        proposal_id STRING, file_path STRING, file_hash STRING,
        mapping STRING, applied_by STRING, applied_at TIMESTAMP
    )""",
    f"""CREATE TABLE IF NOT EXISTS {prefix}.value_transformations (
        proposal_id STRING, file_path STRING, header_mapping_proposal_id STRING,
        transformations STRING, applied_by STRING, applied_at TIMESTAMP
    )""",
    f"""CREATE TABLE IF NOT EXISTS {prefix}.validations (
        proposal_id STRING, file_path STRING, header_mapping_proposal_id STRING,
        validation_results STRING, is_valid BOOLEAN,
        applied_by STRING, applied_at TIMESTAMP
    )""",
    f"""CREATE TABLE IF NOT EXISTS {prefix}.mcf_outputs (
        proposal_id STRING, file_path STRING, effective_date STRING,
        mcf_output STRING, applied_by STRING, applied_at TIMESTAMP
    )""",
    # Support tables
    f"""CREATE TABLE IF NOT EXISTS {prefix}.learned_corrections (
        id BIGINT GENERATED ALWAYS AS IDENTITY,
        stage STRING, source_value STRING, correct_value STRING,
        created_by STRING, created_at TIMESTAMP
    )""",
    f"""CREATE TABLE IF NOT EXISTS {prefix}.user_feedback (
        id BIGINT GENERATED ALWAYS AS IDENTITY,
        proposal_id STRING, stage STRING, feedback_type STRING,
        feedback_text STRING, corrections STRING,
        created_by STRING, created_at TIMESTAMP
    )""",
]

for ddl in table_ddls:
    try:
        spark.sql(ddl)
        table_name = ddl.split("EXISTS")[1].split("(")[0].strip()
        print(f"✅ {table_name}")
    except Exception as e:
        print(f"❌ Error: {e}")

print("\n✅ All tables created!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Verify File Access

# COMMAND ----------

import os, json

# Check schema file
schema_file = f"{SCHEMA_PATH}/{SCHEMA_NAME}.json"
if os.path.exists(schema_file):
    with open(schema_file) as f:
        schema = json.load(f)
    print(f"✅ Schema loaded: {SCHEMA_NAME}.json ({len(schema.get('headers', {}))} headers)")
else:
    print(f"❌ Schema file not found: {schema_file}")
    print(f"   Make sure SCHEMA_PATH points to a Unity Catalog Volume.")

# Check tape file
if os.path.exists(FILE_PATH):
    print(f"✅ Tape file found: {FILE_PATH}")
    import pandas as pd
    if FILE_PATH.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(FILE_PATH, engine='openpyxl')
    else:
        df = pd.read_csv(FILE_PATH)
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)[:10]}...")
else:
    print(f"❌ Tape file not found: {FILE_PATH}")
    print(f"   Make sure it exists in a Unity Catalog Volume.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Test Header Mapping (On-Cluster)
# MAGIC This tests the processing logic directly — no MCP server needed.

# COMMAND ----------

import sys
sys.path.insert(0, WORKSPACE_BASE)

os.environ["TAPESMITH_CATALOG"] = CATALOG
os.environ["TAPESMITH_DB_SCHEMA"] = DB_SCHEMA
os.environ["TAPESMITH_SCHEMA_PATH"] = SCHEMA_PATH
os.environ["TAPESMITH_SCHEMA_NAME"] = SCHEMA_NAME
os.environ["DATABRICKS_SQL_WAREHOUSE_ID"] = SQL_WAREHOUSE_ID
os.environ["DATABRICKS_MODEL_NAME"] = LLM_MODEL_NAME

from processing.data_processor import sanitize_df
from processing.schema_loader import load_schema
from processing.agents import HeaderMappingAgent

schema = load_schema(SCHEMA_NAME, SCHEMA_PATH)
print(f"Schema loaded: {len(schema.get('headers', {}))} standard headers")

tape_df = pd.read_excel(FILE_PATH, engine='openpyxl') if FILE_PATH.endswith(('.xlsx', '.xls')) else pd.read_csv(FILE_PATH)
tape_df = sanitize_df(tape_df)
print(f"Tape loaded: {tape_df.shape}")

# Create a mock config for the agent
from dataclasses import dataclass
@dataclass
class MockConfig:
    databricks_model_name: str = LLM_MODEL_NAME

from databricks_langchain import ChatDatabricks
llm = ChatDatabricks(endpoint=LLM_MODEL_NAME, temperature=0.0)

agent = HeaderMappingAgent(MockConfig(), schema, llm)
result = agent.execute({"tape_df": tape_df, "learned_corrections": {}, "corrections_context": ""})

print(f"\n✅ Header mapping complete!")
print(f"   Mapped: {len(result['mapped_headers'])} headers")
print(f"   LLM suggestions: {len(result['suggestions'])}")
for k, v in sorted(result['mapped_headers'].items()):
    print(f"   {k} → {v}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Deploy as Databricks App
# MAGIC
# MAGIC Run these commands from your **local terminal** (with Databricks CLI configured):
# MAGIC
# MAGIC ```bash
# MAGIC # 1. Navigate to the deploy folder
# MAGIC cd tapesmith_customer_deploy
# MAGIC
# MAGIC # 2. Create the app (first time only)
# MAGIC databricks apps create tapesmith-mcp --app-yaml app.yaml
# MAGIC
# MAGIC # 3. Deploy code to the app
# MAGIC databricks apps deploy tapesmith-mcp --source-code-path .
# MAGIC
# MAGIC # 4. Check app status
# MAGIC databricks apps get tapesmith-mcp
# MAGIC
# MAGIC # 5. View logs if needed
# MAGIC databricks apps logs tapesmith-mcp
# MAGIC ```
# MAGIC
# MAGIC **Before deploying:** Update `app.yaml` with your actual values for:
# MAGIC - `DATABRICKS_SQL_WAREHOUSE_ID`
# MAGIC - `TAPESMITH_SCHEMA_PATH`
# MAGIC - `TAPESMITH_CATALOG`
# MAGIC - `TAPESMITH_DB_SCHEMA`
# MAGIC
# MAGIC **After deploying:**
# MAGIC 1. Grant the app's service principal **READ VOLUME** access to your schema/tape volumes
# MAGIC 2. Grant **USE CATALOG**, **USE SCHEMA**, **SELECT**, **MODIFY** on the Delta tables
# MAGIC 3. Go to **AI Playground** → Select your LLM → Add MCP connection → Enter the app URL

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Verify Tables Have Data (After Running Pipeline)

# COMMAND ----------

for table in [
    "header_mapping_proposals",
    "value_transformation_proposals",
    "validation_results",
    "mcf_generation_proposals",
    "header_mappings",
    "value_transformations",
    "mcf_outputs",
    "learned_corrections",
    "user_feedback",
]:
    try:
        count = spark.sql(f"SELECT COUNT(*) as cnt FROM {CATALOG}.{DB_SCHEMA}.{table}").collect()[0]['cnt']
        print(f"  {table}: {count} rows")
    except Exception as e:
        print(f"  {table}: ❌ {e}")
