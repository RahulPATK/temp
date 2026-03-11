-- ══════════════════════════════════════════════════════════════════════════
-- TapeSmith MCP Server — Create All Tables
--
-- Replace __CATALOG__ and __SCHEMA__ with your Unity Catalog values.
-- Example: my_catalog.tapesmith
--
-- Run this ONCE per environment (dev, staging, prod) before first deploy.
-- All statements are idempotent (CREATE TABLE IF NOT EXISTS).
-- ══════════════════════════════════════════════════════════════════════════


-- ──────────────────────────────────────────────────────────────────────────
-- STAGING TABLES  (proposals awaiting user approval)
-- ──────────────────────────────
-- SET TBLPROPERTIES('delta.feature.allowColumnDefaults' = 'supported')

-- Stage 1: Header Mapping Proposals
CREATE TABLE IF NOT EXISTS rahul_pathak.nl_example.header_mapping_proposals (
    proposal_id STRING,
    file_path STRING,
    file_hash STRING,
    proposed_mapping STRING,
    llm_suggestions STRING,
    status STRING,
    created_by STRING,
    created_at TIMESTAMP,
    approved_by STRING,
    approved_at TIMESTAMP,
    comments STRING
);

ALTER TABLE rahul_pathak.nl_example.header_mapping_proposals
SET TBLPROPERTIES('delta.feature.allowColumnDefaults' = 'supported');

ALTER TABLE rahul_pathak.nl_example.header_mapping_proposals
ALTER COLUMN status SET DEFAULT 'pending';

-- Stage 2: Value Transformation Proposals
CREATE TABLE IF NOT EXISTS rahul_pathak.nl_example.value_transformation_proposals (
    proposal_id STRING,
    file_path STRING,
    header_mapping_proposal_id STRING,
    proposed_transformations STRING,
    unmapped_values STRING,
    llm_suggestions STRING,
    status STRING,
    created_by STRING,
    created_at TIMESTAMP,
    approved_by STRING,
    approved_at TIMESTAMP,
    comments STRING
);

ALTER TABLE rahul_pathak.nl_example.value_transformation_proposals
SET TBLPROPERTIES('delta.feature.allowColumnDefaults' = 'supported');

ALTER TABLE rahul_pathak.nl_example.value_transformation_proposals
ALTER COLUMN status SET DEFAULT 'pending';

-- Stage 3: Validation Results
CREATE TABLE IF NOT EXISTS rahul_pathak.nl_example.validation_results (
    proposal_id STRING,
    file_path STRING,
    header_mapping_proposal_id STRING,
    value_transformation_proposal_id STRING,
    validation_results STRING,
    is_valid BOOLEAN,
    issues STRING,
    created_by STRING,
    created_at TIMESTAMP
);

-- Stage 4: MCF Generation Proposals
CREATE TABLE IF NOT EXISTS rahul_pathak.nl_example.mcf_generation_proposals (
    proposal_id STRING,
    file_path STRING,
    effective_date STRING,
    header_mapping_proposal_id STRING,
    value_transformation_proposal_id STRING,
    validation_proposal_id STRING,
    mcf_output STRING,
    status STRING,
    created_by STRING,
    created_at TIMESTAMP,
    approved_by STRING,
    approved_at TIMESTAMP,
    comments STRING
);


-- ──────────────────────────────────────────────────────────────────────────
-- FINAL TABLES  (approved / persisted records)
-- ──────────────────────────────────────────────────────────────────────────

-- Final: Approved Header Mappings
CREATE TABLE IF NOT EXISTS rahul_pathak.nl_example.header_mappings (
    proposal_id STRING,
    file_path STRING,
    file_hash STRING,
    mapping STRING,
    applied_by STRING,
    applied_at TIMESTAMP
);

-- Final: Approved Value Transformations
CREATE TABLE IF NOT EXISTS rahul_pathak.nl_example.value_transformations (
    proposal_id STRING,
    file_path STRING,
    header_mapping_proposal_id STRING,
    transformations STRING,
    applied_by STRING,
    applied_at TIMESTAMP
);

-- Final: Approved Validations
CREATE TABLE IF NOT EXISTS rahul_pathak.nl_example.validations (
    proposal_id STRING,
    file_path STRING,
    header_mapping_proposal_id STRING,
    validation_results STRING,
    is_valid BOOLEAN,
    applied_by STRING,
    applied_at TIMESTAMP
);

-- Final: Approved MCF Outputs
CREATE TABLE IF NOT EXISTS rahul_pathak.nl_example.mcf_outputs (
    proposal_id STRING,
    file_path STRING,
    effective_date STRING,
    mcf_output STRING,
    applied_by STRING,
    applied_at TIMESTAMP
);


-- ──────────────────────────────────────────────────────────────────────────
-- SUPPORT TABLES  (feedback loop)
-- ──────────────────────────────────────────────────────────────────────────

-- Learned Corrections (auto-applied on subsequent runs)
CREATE TABLE IF NOT EXISTS rahul_pathak.nl_example.learned_corrections (
    id BIGINT GENERATED ALWAYS AS IDENTITY,
    stage STRING,
    source_value STRING,
    correct_value STRING,
    created_by STRING,
    created_at TIMESTAMP
);

-- User Feedback (audit trail)
CREATE TABLE IF NOT EXISTS rahul_pathak.nl_example.user_feedback (
    id BIGINT GENERATED ALWAYS AS IDENTITY,
    proposal_id STRING,
    stage STRING,
    feedback_type STRING,
    feedback_text STRING,
    corrections STRING,
    created_by STRING,
    created_at TIMESTAMP
);

