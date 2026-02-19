-- =============================================================================
-- ML Image Analysis Database Schema - FINAL CORRECTED VERSION
-- Created: January 10, 2026
-- Reviewed: January 13, 2026
-- Finalized: January 13, 2026
-- Version: 2.1 FINAL
-- =============================================================================
--
-- PURPOSE: This file defines 7 tables for storing ML image analysis data
-- EXECUTION ORDER: Must be run BEFORE create_indexes.sql and create_views.sql
--
-- CHANGES FROM ORIGINAL:
-- 1. REMOVED invalid WHERE clause from ground_truth_history table constraint
--    (moved to partial unique index in create_indexes.sql instead)
-- 2. CHANGED images table to use SERIAL primary key + source tracking (Option C)
-- 3. ADDED python_script column to analysis_runs table
-- 4. UPDATED clustering_results comment (removed "autoencoder" reference)
-- =============================================================================

-- Optional: Enable case-insensitive text extension (commented out by default)
-- Uncomment if you need case-insensitive text comparisons
-- CREATE EXTENSION IF NOT EXISTS citext;

-- =============================================================================
-- TABLE 1: images
-- Core entity table storing base image information
-- =============================================================================
--
-- EXPLANATION:
-- - Stores metadata about each image in the analysis
-- - image_id: Auto-incrementing internal database ID (SERIAL)
-- - source: Which dataset this image comes from (e.g., 'giub', 'vg')
-- - source_image_id: The original ID from the source dataset (from filename)
-- - filename: Must be unique across all images
-- - file_path: Can be NULL for portability (different environments have different paths)
-- - date_added: Auto-populated with current timestamp when row is inserted
--
-- DESIGN CHOICE (Option C):
-- - Database manages internal IDs (image_id) automatically
-- - User's original IDs preserved in source_image_id
-- - Allows multiple datasets without ID collisions
-- - UNIQUE constraint on (source, source_image_id) prevents duplicates within each source
--
-- POSTGRESQL FEATURES USED:
-- - SERIAL PRIMARY KEY: Auto-incrementing integer (shorthand for INTEGER + sequence)
-- - UNIQUE constraint: Prevents duplicate filenames
-- - COMPOSITE UNIQUE constraint: Ensures (source, source_image_id) combination is unique
-- - DEFAULT NOW(): Automatically sets timestamp to insertion time
-- - DEFAULT 'giub': Sets default source for existing data
-- =============================================================================

CREATE TABLE IF NOT EXISTS images (
    image_id SERIAL PRIMARY KEY,            -- Auto-increment internal ID
    source TEXT NOT NULL DEFAULT 'giub',    -- Dataset source identifier
    source_image_id INTEGER NOT NULL,       -- Original ID from source (e.g., from filename)
    filename TEXT UNIQUE NOT NULL,          -- Unique filename across all sources
    file_path TEXT,                         -- Nullable - environment-specific
    date_added TIMESTAMP DEFAULT NOW(),     -- Auto-populated on insert
    total_size_bytes INTEGER,               -- File size in bytes
    
    -- Ensure no duplicate (source, source_image_id) combinations
    CONSTRAINT unique_source_image UNIQUE (source, source_image_id)
);

-- Table-level comments (visible in PostgreSQL metadata)
COMMENT ON TABLE images IS 
    'Core table storing image metadata with auto-increment ID and source tracking';
COMMENT ON COLUMN images.image_id IS 
    'Auto-incrementing internal database ID (managed by PostgreSQL)';
COMMENT ON COLUMN images.source IS 
    'Dataset source identifier (e.g., giub, vg) - allows multiple datasets without ID collision';
COMMENT ON COLUMN images.source_image_id IS 
    'Original ID from source dataset (extracted from filename) - preserved for reference';
COMMENT ON COLUMN images.file_path IS 
    'Environment-specific path - can be null for portability';

-- =============================================================================
-- TABLE 2: ground_truth_history
-- Single source of truth for all ground truth labels (current + historical)
-- =============================================================================
--
-- EXPLANATION:
-- - Stores ALL ground truth labels including historical changes
-- - is_current flag: TRUE = current value, FALSE = historical value
-- - Only ONE row per (image_id, label_name) can have is_current = TRUE
-- - history_id: Auto-incrementing primary key (SERIAL)
-- - ON DELETE CASCADE: If an image is deleted, all its ground truth rows are deleted too
--
-- IMPORTANT NOTE ABOUT UNIQUENESS CONSTRAINT:
-- - We CANNOT enforce "only one is_current=TRUE per (image_id, label_name)" 
--   using a table constraint with WHERE clause (syntax not supported)
-- - Instead, this is enforced by a PARTIAL UNIQUE INDEX in create_indexes.sql
-- - See: CREATE UNIQUE INDEX unique_current_label ... WHERE is_current = TRUE
--
-- POSTGRESQL FEATURES USED:
-- - SERIAL: Auto-incrementing integer (shorthand for INTEGER with sequence)
-- - FOREIGN KEY with ON DELETE CASCADE: Maintains referential integrity
-- - BOOLEAN type: TRUE/FALSE/NULL values
-- - DEFAULT values: Auto-populate fields on insert
-- =============================================================================

CREATE TABLE IF NOT EXISTS ground_truth_history (
    history_id SERIAL PRIMARY KEY,                                      -- Auto-incrementing ID
    image_id INTEGER NOT NULL REFERENCES images(image_id) ON DELETE CASCADE,  -- FK to images
    label_name TEXT NOT NULL,                                           -- e.g., 'with_person'
    value TEXT NOT NULL,                                                -- e.g., 'true', 'false'
    changed_at TIMESTAMP DEFAULT NOW(),                                 -- When this value was set
    reason TEXT,                                                        -- Why it was changed (optional)
    is_current BOOLEAN DEFAULT TRUE                                     -- TRUE = current, FALSE = historical
    
    -- NOTE: The unique constraint for is_current = TRUE is enforced via 
    -- partial index in create_indexes.sql, not here
);

-- Table-level comments
COMMENT ON TABLE ground_truth_history IS 
    'Complete audit trail of ground truth labels with history';
COMMENT ON COLUMN ground_truth_history.is_current IS 
    'Only one TRUE per (image_id, label_name) - enforced by partial unique index';
COMMENT ON COLUMN ground_truth_history.history_id IS 
    'Auto-incrementing surrogate key';

-- =============================================================================
-- TABLE 3: prompts
-- Reusable prompt library (independent of runs/images)
-- =============================================================================
--
-- EXPLANATION:
-- - Stores LLM prompt templates that can be reused across multiple analysis runs
-- - prompt_name: Human-readable identifier (must be unique)
-- - prompt_text: The actual prompt text sent to the LLM
-- - Independent table: Not tied to specific images or runs
--
-- POSTGRESQL FEATURES USED:
-- - SERIAL PRIMARY KEY: Auto-incrementing ID
-- - UNIQUE constraint on prompt_name: Prevents duplicate prompt names
-- =============================================================================

CREATE TABLE IF NOT EXISTS prompts (
    prompt_id SERIAL PRIMARY KEY,           -- Auto-incrementing ID
    prompt_name TEXT UNIQUE NOT NULL,       -- Unique name (e.g., 'basic_person_detection')
    prompt_text TEXT NOT NULL,              -- The actual prompt content
    created_at TIMESTAMP DEFAULT NOW()      -- When prompt was created
);

COMMENT ON TABLE prompts IS 
    'Library of reusable LLM prompt templates';

-- =============================================================================
-- TABLE 4: analysis_runs
-- Metadata for each analysis execution
-- =============================================================================
--
-- EXPLANATION:
-- - Each row represents one execution of a model on a set of images
-- - run_timestamp: Used as identifier (format: YYYYMMDD_HHMMSS)
-- - model_name: The primary model used (ONE model per run - best practice)
-- - python_script: Optional filename of the script that executed the analysis
-- - hyperparameters: Stored as JSONB for flexible structure
-- - duration_seconds: How long the analysis took
--
-- DESIGN PRINCIPLE:
-- - ONE model per analysis run (best practice for tracking and comparison)
-- - If multiple models needed, create separate analysis runs
-- - python_script is nullable and not unique (same script can run multiple times)
--
-- POSTGRESQL FEATURES USED:
-- - JSONB: Binary JSON storage with indexing support (more efficient than JSON)
-- - REAL: 4-byte floating point (vs DOUBLE PRECISION which is 8-byte)
-- - SERIAL: Auto-incrementing primary key
-- =============================================================================

CREATE TABLE IF NOT EXISTS analysis_runs (
    analysis_run_id SERIAL PRIMARY KEY,     -- Auto-incrementing ID
    run_timestamp TIMESTAMP NOT NULL,       -- When this analysis was run
    analysis_type TEXT NOT NULL,            -- e.g., 'llm_classification', 'yolo_detection'
    model_name TEXT NOT NULL,               -- Primary model used (ONE per run)
    python_script TEXT,                     -- Script filename (optional, can be reused)
    model_version TEXT,                     -- Model version string (optional)
    hyperparameters JSONB,                  -- Model parameters as JSON
    notes TEXT,                             -- Human notes about the run
    images_processed INTEGER,               -- How many images were analyzed
    start_time TIMESTAMP,                   -- When processing started
    duration_seconds REAL                   -- How long it took (using REAL not FLOAT)
);

COMMENT ON TABLE analysis_runs IS 
    'Metadata for each analysis execution - one model per run (best practice)';
COMMENT ON COLUMN analysis_runs.run_timestamp IS 
    'Run identifier in YYYYMMDD_HHMMSS format';
COMMENT ON COLUMN analysis_runs.model_name IS 
    'Primary model used in this run - one model per run recommended';
COMMENT ON COLUMN analysis_runs.python_script IS 
    'Script filename that executed this analysis (optional, can be reused across runs)';
COMMENT ON COLUMN analysis_runs.hyperparameters IS 
    'JSONB format allows flexible structure and indexing';

-- =============================================================================
-- TABLE 5: llm_responses
-- LLM responses with direct prompt reference
-- =============================================================================
--
-- EXPLANATION:
-- - Stores raw and parsed LLM responses
-- - Links to: analysis_run, image, and prompt
-- - parsed_response: Structured JSON output from LLM
-- - raw_response_text: Original unprocessed response
-- - parse_success: Did we successfully parse the JSON?
--
-- POSTGRESQL FEATURES USED:
-- - Multiple FOREIGN KEYS with different ON DELETE behaviors:
--   * CASCADE: Delete responses when run/image deleted
--   * RESTRICT: Prevent prompt deletion if responses reference it
-- - JSONB: For structured parsed responses
-- =============================================================================

CREATE TABLE IF NOT EXISTS llm_responses (
    response_id SERIAL PRIMARY KEY,                                     -- Auto-incrementing ID
    analysis_run_id INTEGER NOT NULL REFERENCES analysis_runs(analysis_run_id) ON DELETE CASCADE,
    image_id INTEGER NOT NULL REFERENCES images(image_id) ON DELETE CASCADE,
    prompt_id INTEGER NOT NULL REFERENCES prompts(prompt_id) ON DELETE RESTRICT,
    parsed_response JSONB,                                              -- Structured JSON output
    raw_response_text TEXT,                                             -- Original raw text
    parse_success BOOLEAN DEFAULT FALSE,                                -- Did JSON parsing work?
    tokens_used INTEGER,                                                -- Token count for cost tracking
    received_at TIMESTAMP DEFAULT NOW()                                 -- When response received
);

COMMENT ON TABLE llm_responses IS 
    'LLM outputs with both structured (JSONB) and raw text';
COMMENT ON COLUMN llm_responses.parse_success IS 
    'TRUE if JSON parsing succeeded';
COMMENT ON COLUMN llm_responses.prompt_id IS 
    'ON DELETE RESTRICT prevents accidental prompt deletion';

-- =============================================================================
-- TABLE 6: predictions
-- Model predictions (flexible EAV format)
-- =============================================================================
--
-- EXPLANATION:
-- - Entity-Attribute-Value (EAV) structure for flexible predictions
-- - Can store any type of prediction: binary, multi-class, etc.
-- - prompt_id: NULL for non-LLM predictions (e.g., YOLO, clustering)
-- - predicted_value: TEXT type supports both '0'/'1' and class names
--
-- UNIQUENESS CONSTRAINT:
-- - One prediction per (analysis_run, image, label, prompt) combination
-- - prompt_id is part of UNIQUE constraint so NULL values are treated specially
--
-- FOREIGN KEY BEHAVIOR:
-- - ON DELETE SET NULL for prompt_id: If prompt deleted, prediction remains but loses prompt link
-- - This preserves prediction data even if prompt is removed
--
-- POSTGRESQL FEATURES USED:
-- - UNIQUE constraint with multiple columns including nullable column
-- - ON DELETE SET NULL: Keep predictions even if prompt is deleted
-- - REAL for confidence scores (0.0 to 1.0 range)
-- =============================================================================

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id SERIAL PRIMARY KEY,                                   -- Auto-incrementing ID
    analysis_run_id INTEGER NOT NULL REFERENCES analysis_runs(analysis_run_id) ON DELETE CASCADE,
    image_id INTEGER NOT NULL REFERENCES images(image_id) ON DELETE CASCADE,
    prompt_id INTEGER REFERENCES prompts(prompt_id) ON DELETE SET NULL, -- Nullable for non-LLM
    label_name TEXT NOT NULL,                                           -- What we're predicting
    predicted_value TEXT NOT NULL,                                      -- The prediction (0/1/class_name)
    confidence_score REAL,                                              -- Model confidence (0.0-1.0)
    created_at TIMESTAMP DEFAULT NOW(),                                 -- When prediction was made
    
    -- One prediction per label per image per run (per prompt if applicable)
    -- NOTE: Including prompt_id allows multiple predictions for same label using different prompts
    CONSTRAINT unique_prediction UNIQUE (analysis_run_id, image_id, label_name, prompt_id)
);

COMMENT ON TABLE predictions IS 
    'Flexible EAV table supporting binary and multi-class predictions';
COMMENT ON COLUMN predictions.prompt_id IS 
    'NULL for non-LLM predictions (YOLO, clustering). ON DELETE SET NULL preserves predictions if prompt deleted.';
COMMENT ON COLUMN predictions.predicted_value IS 
    'TEXT type supports both 0/1 and class names';

-- =============================================================================
-- TABLE 7: clustering_results
-- Clustering algorithm outputs
-- =============================================================================
--
-- EXPLANATION:
-- - Stores results from clustering algorithms (e.g., K-means, hierarchical)
-- - cluster_id: Which cluster this image belongs to
-- - distance_to_centroid: How far from cluster center
-- - embedding: Stored as TEXT (can be parsed as JSON/array in application)
--
-- DESIGN CHOICE:
-- - embedding stored as TEXT rather than ARRAY or JSONB for flexibility
-- - Application layer handles parsing/deserialization
--
-- NOTE: Updated to support any clustering algorithm, not just autoencoders
-- =============================================================================

CREATE TABLE IF NOT EXISTS clustering_results (
    clustering_id SERIAL PRIMARY KEY,                                   -- Auto-incrementing ID
    analysis_run_id INTEGER NOT NULL REFERENCES analysis_runs(analysis_run_id) ON DELETE CASCADE,
    image_id INTEGER NOT NULL REFERENCES images(image_id) ON DELETE CASCADE,
    cluster_id INTEGER NOT NULL,                                        -- Which cluster (0, 1, 2, ...)
    distance_to_centroid REAL,                                          -- Distance from cluster center
    embedding TEXT                                                      -- Vector embedding as text
);

COMMENT ON TABLE clustering_results IS 
    'Results from clustering analyses';
COMMENT ON COLUMN clustering_results.embedding IS 
    'Stored as TEXT - parse in application as needed';

-- =============================================================================
-- Add CHECK constraints for data validation
-- =============================================================================
--
-- EXPLANATION:
-- - CHECK constraints validate data at insertion/update time
-- - This ensures confidence_score is always between 0 and 1 (or NULL)
--
-- POSTGRESQL FEATURE:
-- - CHECK constraint can reference the same row's columns
-- - Evaluated before INSERT/UPDATE completes
-- =============================================================================

ALTER TABLE predictions
    ADD CONSTRAINT check_confidence_range
    CHECK (confidence_score IS NULL OR (confidence_score >= 0 AND confidence_score <= 1));

-- Optional: Constrain label_name to known values
-- UNCOMMENT if you want to restrict labels to a specific set
-- This is commented out by default for flexibility (dynamic labels)
-- 
-- ALTER TABLE ground_truth_history
--     ADD CONSTRAINT check_label_names
--     CHECK (label_name IN ('with_person', 'with_person_recognisable', 'is_photo', 
--                           'with_church', 'in_high_alpine_environment'));

-- =============================================================================
-- Schema completion message
-- =============================================================================
--
-- EXPLANATION:
-- - Uses anonymous code block (DO $$) to execute PostgreSQL code
-- - RAISE NOTICE: Prints message to output (not an error)
-- - Useful for confirming script completed successfully
-- =============================================================================

DO $$
BEGIN
    RAISE NOTICE '=======================================================';
    RAISE NOTICE 'Schema created successfully!';
    RAISE NOTICE '  Tables: 7';
    RAISE NOTICE '  Foreign Keys: 10';
    RAISE NOTICE '  Constraints: Multiple';
    RAISE NOTICE '';
    RAISE NOTICE 'Changes in this version:';
    RAISE NOTICE '  - Images table uses SERIAL ID + source tracking';
    RAISE NOTICE '  - Analysis runs includes python_script column';
    RAISE NOTICE '  - Clustering results supports any algorithm';
    RAISE NOTICE '=======================================================';
END $$;

-- =============================================================================
-- END OF create_tables.sql
-- =============================================================================
