-- =============================================================================
-- ML Image Analysis Database - Indexes - FINAL CORRECTED VERSION
-- Created: January 10, 2026
-- Reviewed: January 13, 2026
-- Finalized: January 13, 2026
-- Version: 2.1 FINAL
-- =============================================================================
--
-- PURPOSE: Create indexes for query performance and enforce uniqueness constraints
-- EXECUTION ORDER: Must be run AFTER create_tables.sql, BEFORE loading data
--
-- WHY INDEXES MATTER:
-- - Foreign keys: PostgreSQL does NOT auto-create indexes on FK columns
-- - Without indexes, FK lookups are slow (full table scans)
-- - Query optimization: Indexes speed up WHERE, JOIN, and ORDER BY operations
--
-- CHANGES FROM ORIGINAL:
-- - ADDED: unique_current_label partial unique index (enforces business rule)
-- - This replaces the invalid WHERE clause that was in create_tables.sql
-- - Updated to work with new images table structure (SERIAL ID)
-- =============================================================================

-- =============================================================================
-- CRITICAL: Partial Unique Index for ground_truth_history
-- =============================================================================
--
-- EXPLANATION:
-- - Business rule: Only ONE row can have is_current=TRUE for each (image_id, label_name)
-- - This allows multiple historical records (is_current=FALSE) for the same label
-- - We use a PARTIAL index with WHERE clause to enforce this
--
-- WHY NOT A TABLE CONSTRAINT?
-- - Table-level UNIQUE constraints don't support WHERE clauses in PostgreSQL
-- - Must use CREATE UNIQUE INDEX with WHERE instead
--
-- POSTGRESQL FEATURE: PARTIAL UNIQUE INDEX
-- - Index only applies to rows matching the WHERE condition
-- - Much more efficient than indexing all rows
-- - Enforces uniqueness only among "current" records
--
-- EXAMPLE:
-- These are ALLOWED (multiple historical records):
--   image_id=1, label='person', is_current=FALSE
--   image_id=1, label='person', is_current=FALSE
--   image_id=1, label='person', is_current=TRUE   ← only ONE current allowed
--
-- This would be REJECTED (duplicate current record):
--   image_id=1, label='person', is_current=TRUE   ← ERROR: violates unique constraint
-- =============================================================================

CREATE UNIQUE INDEX IF NOT EXISTS unique_current_label 
    ON ground_truth_history(image_id, label_name)
    WHERE is_current = TRUE;

COMMENT ON INDEX unique_current_label IS 
    'Enforces: only one is_current=TRUE per (image_id, label_name) combination';

-- =============================================================================
-- Foreign Key Indexes (CRITICAL - PostgreSQL doesn't auto-create these!)
-- =============================================================================
--
-- EXPLANATION:
-- - PostgreSQL creates indexes for PRIMARY KEY columns automatically
-- - But does NOT create indexes for FOREIGN KEY columns
-- - Without these indexes, FK lookups require full table scans (very slow!)
--
-- WHEN THESE MATTER:
-- - JOINs on foreign key columns
-- - ON DELETE CASCADE operations
-- - Checking referential integrity
--
-- INDEX TYPE: B-tree (default)
-- - Good for equality and range queries
-- - Supports sorting (ORDER BY)
--
-- NOTE: images.image_id already has index (PRIMARY KEY auto-creates it)
-- =============================================================================

-- Index for ground_truth_history.image_id (FK to images)
-- Used when: Joining ground_truth to images, deleting images (CASCADE)
CREATE INDEX IF NOT EXISTS idx_gt_history_image_id 
    ON ground_truth_history(image_id);

-- Indexes for llm_responses foreign keys
-- Used when: Joining to analysis_runs, images, or prompts
CREATE INDEX IF NOT EXISTS idx_llm_resp_analysis_run 
    ON llm_responses(analysis_run_id);

CREATE INDEX IF NOT EXISTS idx_llm_resp_image_id 
    ON llm_responses(image_id);

CREATE INDEX IF NOT EXISTS idx_llm_resp_prompt_id 
    ON llm_responses(prompt_id);

-- Indexes for predictions foreign keys
-- Used when: Joining predictions to runs/images/prompts
CREATE INDEX IF NOT EXISTS idx_predictions_analysis_run 
    ON predictions(analysis_run_id);

CREATE INDEX IF NOT EXISTS idx_predictions_image_id 
    ON predictions(image_id);

CREATE INDEX IF NOT EXISTS idx_predictions_prompt_id 
    ON predictions(prompt_id);

-- Indexes for clustering_results foreign keys
-- Used when: Joining clustering results to runs/images
CREATE INDEX IF NOT EXISTS idx_clustering_analysis_run 
    ON clustering_results(analysis_run_id);

CREATE INDEX IF NOT EXISTS idx_clustering_image_id 
    ON clustering_results(image_id);

-- =============================================================================
-- Additional Index for New Images Table Structure
-- =============================================================================
--
-- EXPLANATION:
-- - With the new design, you'll often query by (source, source_image_id)
-- - This is your original ID that you use in Python code
-- - A composite index speeds up lookups using both columns
--
-- NOTE: The UNIQUE constraint already creates an index, but this is documented
-- here for clarity. PostgreSQL will use the UNIQUE index automatically.
-- =============================================================================

-- This index is automatically created by the UNIQUE constraint in create_tables.sql:
-- CONSTRAINT unique_source_image UNIQUE (source, source_image_id)
-- 
-- But documenting here for reference:
COMMENT ON INDEX unique_source_image IS 
    'Composite index on (source, source_image_id) - automatically created by UNIQUE constraint';

-- =============================================================================
-- Query Optimization Indexes
-- =============================================================================
--
-- EXPLANATION:
-- - These indexes speed up common query patterns
-- - Designed based on expected use cases
-- - Trade-off: Indexes speed up reads but slow down writes slightly
--
-- DESIGN PRINCIPLE:
-- - Index columns used in WHERE clauses
-- - Index columns used in JOINs
-- - Consider multi-column indexes for common query combinations
-- =============================================================================

-- Common query: Get CURRENT ground truth for an image and label
-- Used in: Comparing predictions to ground truth
-- NOTE: This is a PARTIAL index (only indexes rows where is_current=TRUE)
CREATE INDEX IF NOT EXISTS idx_gt_current_lookup 
    ON ground_truth_history(image_id, label_name, is_current)
    WHERE is_current = TRUE;

COMMENT ON INDEX idx_gt_current_lookup IS 
    'Optimizes lookups for current ground truth values';

-- Common query: Find all predictions for a specific label
-- Used in: Analyzing performance per label type
CREATE INDEX IF NOT EXISTS idx_predictions_label 
    ON predictions(label_name);

-- Common query: Look up predictions for specific run+image+label combination
-- Multi-column index: Most efficient for queries using all three columns
CREATE INDEX IF NOT EXISTS idx_predictions_lookup 
    ON predictions(analysis_run_id, image_id, label_name);

COMMENT ON INDEX idx_predictions_lookup IS 
    'Composite index for run+image+label lookups (used in performance analysis)';

-- Time-based queries on analysis runs
-- Used when: Finding runs within a date range, ordering by time
CREATE INDEX IF NOT EXISTS idx_analysis_runs_timestamp 
    ON analysis_runs(run_timestamp);

-- Model performance queries
-- Used when: Grouping/filtering by model name and analysis type
CREATE INDEX IF NOT EXISTS idx_analysis_runs_model 
    ON analysis_runs(model_name, analysis_type);

-- =============================================================================
-- JSONB Indexes (for hyperparameters and parsed_response)
-- =============================================================================
--
-- EXPLANATION:
-- - JSONB columns support special index types
-- - GIN (Generalized Inverted Index): For JSONB searching
--
-- POSTGRESQL FEATURE: GIN Index
-- - Supports operators: ?, ?&, ?|, @>, <@
-- - Example: WHERE hyperparameters @> '{"learning_rate": 0.01}'
-- - Example: WHERE hyperparameters ? 'temperature'
--
-- TRADE-OFF:
-- - GIN indexes are larger than B-tree indexes
-- - Much faster for JSONB queries
-- - Slower to update (but read-optimized for our use case)
-- =============================================================================

-- GIN index for hyperparameters (supports ? and @> operators)
-- Used when: Searching for runs with specific hyperparameter values
-- Example query: WHERE hyperparameters @> '{"model": "gpt-4"}'
CREATE INDEX IF NOT EXISTS idx_analysis_runs_hyperparams 
    ON analysis_runs USING GIN (hyperparameters);

COMMENT ON INDEX idx_analysis_runs_hyperparams IS 
    'GIN index enables fast JSONB queries on hyperparameters (supports @> and ? operators)';

-- GIN index for parsed_response
-- Used when: Searching LLM responses for specific content
-- Example query: WHERE parsed_response @> '{"detected": "person"}'
CREATE INDEX IF NOT EXISTS idx_llm_parsed_response 
    ON llm_responses USING GIN (parsed_response);

COMMENT ON INDEX idx_llm_parsed_response IS 
    'GIN index enables fast JSONB queries on parsed LLM responses';

-- =============================================================================
-- Text Search Indexes (Optional - commented out by default)
-- =============================================================================
--
-- EXPLANATION:
-- - Full-text search indexes for TEXT columns
-- - Uses tsvector (text search vector) with GIN index
-- - VERY useful for searching through prompt text or raw responses
--
-- POSTGRESQL FEATURE: Full-Text Search (FTS)
-- - to_tsvector(): Converts text to searchable vector
-- - Supports: Stemming, stop words, ranking
-- - Example query: WHERE to_tsvector('english', prompt_text) @@ to_tsquery('person & detection')
--
-- WHY COMMENTED OUT:
-- - Only needed if you'll search through text frequently
-- - Adds storage overhead and write penalty
-- - Uncomment if you need full-text search capabilities
-- =============================================================================

-- Full-text search on prompt text
-- UNCOMMENT if you need to search prompts by keywords
-- CREATE INDEX IF NOT EXISTS idx_prompts_text_search 
--     ON prompts USING GIN (to_tsvector('english', prompt_text));

-- Full-text search on raw LLM responses
-- UNCOMMENT if you need to search raw response text
-- CREATE INDEX IF NOT EXISTS idx_llm_raw_text_search 
--     ON llm_responses USING GIN (to_tsvector('english', raw_response_text));

-- =============================================================================
-- Completion Message
-- =============================================================================

DO $$
BEGIN
    RAISE NOTICE '=======================================================';
    RAISE NOTICE 'Indexes created successfully!';
    RAISE NOTICE '  - 1 partial unique index (ground_truth uniqueness)';
    RAISE NOTICE '  - 9 B-tree indexes (foreign keys)';
    RAISE NOTICE '  - 5 B-tree indexes (query optimization)';
    RAISE NOTICE '  - 2 GIN indexes (JSONB queries)';
    RAISE NOTICE '  - 2 full-text search indexes (optional - commented out)';
    RAISE NOTICE '';
    RAISE NOTICE 'Total active indexes: 17';
    RAISE NOTICE '  (plus auto-created PRIMARY KEY and UNIQUE indexes)';
    RAISE NOTICE '=======================================================';
END $$;

-- =============================================================================
-- END OF create_indexes.sql
-- =============================================================================
