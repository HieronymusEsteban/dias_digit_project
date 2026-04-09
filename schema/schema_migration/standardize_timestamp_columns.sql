-- =============================================================================
-- Schema Migration: Standardize Timestamp Columns to 'created_at'
-- Date: 2026-03-22
-- Purpose: Consistent timestamp naming across all tables
-- =============================================================================

-- IMPORTANT: This migration is designed for FRESH database setup
-- If you have existing data you want to preserve, use the alternative 
-- migration script that copies old timestamp values to new columns

BEGIN;

-- -----------------------------------------------------------------------------
-- 1. Add created_at to analysis_runs (keep run_timestamp - different meaning)
-- -----------------------------------------------------------------------------

ALTER TABLE analysis_runs 
ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;

COMMENT ON COLUMN analysis_runs.created_at IS 
    'When this record was inserted into the database';

COMMENT ON COLUMN analysis_runs.run_timestamp IS 
    'When the analysis was actually executed (different from record creation)';


-- -----------------------------------------------------------------------------
-- 2. Add created_at to clustering_results
-- -----------------------------------------------------------------------------

ALTER TABLE clustering_results 
ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;

COMMENT ON COLUMN clustering_results.created_at IS 
    'When this record was inserted into the database';


-- -----------------------------------------------------------------------------
-- 3. Rename changed_at to created_at in ground_truth_history
-- -----------------------------------------------------------------------------

ALTER TABLE ground_truth_history 
RENAME COLUMN changed_at TO created_at;

-- Update comment
COMMENT ON COLUMN ground_truth_history.created_at IS 
    'When this ground truth record was created (for historical tracking)';


-- -----------------------------------------------------------------------------
-- 4. Rename received_at to created_at in llm_responses
-- -----------------------------------------------------------------------------

ALTER TABLE llm_responses 
RENAME COLUMN received_at TO created_at;

-- Update comment
COMMENT ON COLUMN llm_responses.created_at IS 
    'When the LLM response was received and recorded in the database';


-- -----------------------------------------------------------------------------
-- Summary of changes
-- -----------------------------------------------------------------------------

DO $$
BEGIN
    RAISE NOTICE '========================================================';
    RAISE NOTICE 'Timestamp Standardization Complete!';
    RAISE NOTICE '';
    RAISE NOTICE 'Tables modified:';
    RAISE NOTICE '  1. analysis_runs - ADDED created_at';
    RAISE NOTICE '  2. clustering_results - ADDED created_at';
    RAISE NOTICE '  3. ground_truth_history - RENAMED changed_at → created_at';
    RAISE NOTICE '  4. llm_responses - RENAMED received_at → created_at';
    RAISE NOTICE '';
    RAISE NOTICE 'Already correct (no changes):';
    RAISE NOTICE '  - predictions (has created_at)';
    RAISE NOTICE '  - training_validation_splits (has created_at)';
    RAISE NOTICE '';
    RAISE NOTICE 'All tables now use consistent "created_at" naming';
    RAISE NOTICE '========================================================';
END $$;

COMMIT;


-- -----------------------------------------------------------------------------
-- Verification Queries
-- -----------------------------------------------------------------------------

-- Run these to verify the changes

-- Check all timestamp columns across tables
SELECT 
    table_name,
    column_name,
    data_type,
    column_default
FROM information_schema.columns
WHERE table_schema = 'public'
    AND column_name LIKE '%created_at%'
    OR column_name LIKE '%timestamp%'
ORDER BY table_name, column_name;


-- Verify each table has created_at
SELECT 
    'analysis_runs' as table_name,
    COUNT(*) FILTER (WHERE created_at IS NOT NULL) as has_created_at
FROM analysis_runs
UNION ALL
SELECT 
    'clustering_results',
    COUNT(*) FILTER (WHERE created_at IS NOT NULL)
FROM clustering_results
UNION ALL
SELECT 
    'ground_truth_history',
    COUNT(*) FILTER (WHERE created_at IS NOT NULL)
FROM ground_truth_history
UNION ALL
SELECT 
    'llm_responses',
    COUNT(*) FILTER (WHERE created_at IS NOT NULL)
FROM llm_responses
UNION ALL
SELECT 
    'predictions',
    COUNT(*) FILTER (WHERE created_at IS NOT NULL)
FROM predictions
UNION ALL
SELECT 
    'training_validation_splits',
    COUNT(*) FILTER (WHERE created_at IS NOT NULL)
FROM training_validation_splits;
