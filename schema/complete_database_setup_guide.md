# Complete Database Setup Guide
**Version:** 2.1  
**Date:** 2026-03-22  
**Purpose:** Fresh database setup with all schema migrations applied in correct order

---

## Overview

This guide provides step-by-step instructions for setting up a complete, production-ready database from scratch. It consolidates all schema files and migrations into a single workflow.

**What this covers:**
- Fresh database creation
- Base schema installation
- All schema migrations (clustering columns, train/val splits, timestamp standardization)
- Index and view creation
- Verification at each step

**NOT covered in this guide:**
- Data loading (use ETL notebooks after schema setup)
- Backup/restore procedures
- Performance tuning

---

## Prerequisites

**Required:**
- PostgreSQL 16 installed and running
- Database created (e.g., `image_analysis_test`)
- Command-line access (`psql`)
- Schema files in `schema/` directory

**Verify PostgreSQL is running:**
```bash
psql --version
# Should show: psql (PostgreSQL) 16.x
```

**Verify database exists:**
```bash
psql -l | grep image_analysis
# Should show your target database
```

---

## File Structure

Ensure you have these files in your project directory:

```
schema/
├── create_tables.sql                               # Base schema (7 tables)
├── create_indexes.sql                              # Indexes for performance
├── create_views.sql                                # 6 analytical views
└── schema_migration/
    ├── alter_analysis_runs_clustering.sql          # Migration 1: Clustering columns
    ├── add_training_validation_splits.sql          # Migration 2: Train/val splits table
    └── standardize_timestamp_columns.sql           # Migration 3: Timestamp standardization
```

---

## Migration Order (CRITICAL)

**Execute in this exact order:**

1. `create_tables.sql` - Base schema (7 tables)
2. `create_indexes.sql` - Performance indexes
3. `alter_analysis_runs_clustering.sql` - Add clustering pipeline columns
4. `add_training_validation_splits.sql` - Add train/val splits table
5. `standardize_timestamp_columns.sql` - Standardize timestamps
6. `create_views.sql` - Create analytical views (MUST be last)

**Why this order matters:**
- Views must come LAST (they reference the final column names including standardized timestamps)
- Training/validation splits table must come AFTER base tables (uses foreign keys)
- Timestamp standardization must come BEFORE views (views reference `created_at` columns)

---

## Complete Setup Procedure

### Step 1: Prepare Environment

```bash
# Navigate to project directory
cd ~/Documents/CAS_AML/dias_digit_project

# Set database name (change for dev/test/prod)
export DB_NAME=image_analysis_test
```

**Optional but recommended:** Backup existing database
```bash
pg_dump $DB_NAME > backups/${DB_NAME}_backup_$(date +%Y%m%d_%H%M%S).sql
```

---

### Step 2: Drop Existing Schema (Fresh Start)

**⚠️ WARNING:** This deletes ALL tables and data!

```bash
psql $DB_NAME
```

```sql
-- Drop all tables (CASCADE removes dependencies)
DROP TABLE IF EXISTS llm_responses CASCADE;
DROP TABLE IF EXISTS predictions CASCADE;
DROP TABLE IF EXISTS clustering_results CASCADE;
DROP TABLE IF EXISTS training_validation_splits CASCADE;
DROP TABLE IF EXISTS ground_truth_history CASCADE;
DROP TABLE IF EXISTS analysis_runs CASCADE;
DROP TABLE IF EXISTS prompts CASCADE;
DROP TABLE IF EXISTS images CASCADE;

-- Drop all views
DROP VIEW IF EXISTS ground_truth_labels CASCADE;
DROP VIEW IF EXISTS ground_truth_wide CASCADE;
DROP VIEW IF EXISTS model_performance CASCADE;
DROP VIEW IF EXISTS prompt_performance CASCADE;
DROP VIEW IF EXISTS recent_analysis_runs CASCADE;
DROP VIEW IF EXISTS image_summary CASCADE;
DROP VIEW IF EXISTS llm_response_summary CASCADE;

-- Verify everything is gone
\dt
\dv

-- Exit
\q
```

**Expected output:** "Did not find any relations" (empty database)

---

### Step 3: Create Base Schema

```bash
psql $DB_NAME -f schema/create_tables.sql
```

**Expected output:**
```
NOTICE:  =======================================================
NOTICE:  Schema created successfully!
NOTICE:    Tables: 7
NOTICE:    Foreign Keys: 10
...
```

**Verify:**
```bash
psql $DB_NAME -c "\dt"
```

**Should show 7 tables:**
- `images`
- `ground_truth_history`
- `prompts`
- `analysis_runs`
- `llm_responses`
- `predictions`
- `clustering_results`

---

### Step 4: Create Indexes

```bash
psql $DB_NAME -f schema/create_indexes.sql
```

**Expected output:**
```
NOTICE:  =======================================================
NOTICE:  Indexes created successfully!
NOTICE:    - 1 partial unique index (ground_truth uniqueness)
NOTICE:    - 9 B-tree indexes (foreign keys)
...
```

**Verify:**
```bash
psql $DB_NAME -c "\di"
```

**Should show 17+ indexes** (including auto-created PRIMARY KEY indexes)

---

### Step 5: Migration 1 - Add Clustering Columns

**Purpose:** Add autoencoder, dimensionality reduction, and clustering columns to `analysis_runs`

```bash
psql $DB_NAME -f schema/schema_migration/alter_analysis_runs_clustering.sql
```

**Verify:**
```bash
psql $DB_NAME -c "\d analysis_runs"
```

**Should see new columns:**
- `autoencoder_name`, `autoencoder_implementation`, `autoencoder_file`, `autoencoder_params`
- `dim_reduction_name`, `dim_reduction_implementation`, `dim_reduction_params`
- `clustering_name`, `clustering_implementation`, `clustering_params`

**📖 Detailed guide:** `schema_migration_clustering_columns.md`

---

### Step 6: Migration 2 - Add Training/Validation Splits Table

**Purpose:** Track which images were used for training vs validation in clustering runs

```bash
psql $DB_NAME -f schema/schema_migration/add_training_validation_splits.sql
```

**Verify:**
```bash
psql $DB_NAME -c "\d training_validation_splits"
```

**Should show table with 5 columns:**
- `split_id` (PRIMARY KEY)
- `analysis_run_id` (FK to analysis_runs)
- `image_id` (FK to images)
- `split_type` (TEXT)
- `created_at` (TIMESTAMP)

**📖 Detailed guide:** `schema_migration_train_val_split.md`

---

### Step 7: Migration 3 - Standardize Timestamp Columns

**Purpose:** Rename all timestamp columns to consistent `created_at` naming

```bash
psql $DB_NAME -f schema/schema_migration/standardize_timestamp_columns.sql
```

**Expected output:**
```
NOTICE:  ========================================================
NOTICE:  Timestamp Standardization Complete!
NOTICE:    1. analysis_runs - ADDED created_at
NOTICE:    2. clustering_results - ADDED created_at
NOTICE:    3. ground_truth_history - RENAMED changed_at → created_at
NOTICE:    4. llm_responses - RENAMED received_at → created_at
...
```

**Verify:**
```bash
psql $DB_NAME
```

```sql
-- Check timestamp columns
SELECT 
    table_name,
    column_name,
    data_type
FROM information_schema.columns
WHERE table_schema = 'public'
    AND (column_name LIKE '%created_at%' OR column_name = 'run_timestamp')
ORDER BY table_name, column_name;
```

**Should show:**
- All tables have `created_at` column
- NO columns named `changed_at` or `received_at`
- `analysis_runs` has BOTH `created_at` and `run_timestamp` (different meanings)

```sql
\q
```

**📖 Detailed guide:** `timestamp_standardization_setup_guide.md`

---

### Step 8: Create Views (MUST BE LAST)

**Purpose:** Create analytical views for common queries

```bash
psql $DB_NAME -f schema/create_views.sql
```

**Expected output:**
```
NOTICE:  =======================================================
NOTICE:  Views created successfully!
NOTICE:    1. ground_truth_labels - Current labels for all images
NOTICE:    2. model_performance - Accuracy metrics by model
...
```

**Verify:**
```bash
psql $DB_NAME -c "\dv"
```

**Should show 6 views:**
- `ground_truth_labels`
- `ground_truth_wide`
- `model_performance`
- `prompt_performance`
- `recent_analysis_runs`
- `image_summary`
- `llm_response_summary`

---

## Final Verification

Run this comprehensive check to verify complete setup:

```bash
psql $DB_NAME
```

```sql
-- =============================================================================
-- Complete Schema Verification
-- =============================================================================

-- 1. Check table count
SELECT COUNT(*) as table_count 
FROM information_schema.tables 
WHERE table_schema = 'public' AND table_type = 'BASE TABLE';
-- Expected: 8 (7 base + 1 from migration)

-- 2. Check view count
SELECT COUNT(*) as view_count 
FROM information_schema.views 
WHERE table_schema = 'public';
-- Expected: 6

-- 3. Verify all tables have created_at
SELECT 
    t.table_name,
    CASE WHEN c.column_name IS NOT NULL THEN '✓' ELSE '✗' END as has_created_at
FROM information_schema.tables t
LEFT JOIN information_schema.columns c 
    ON t.table_name = c.table_name 
    AND c.column_name = 'created_at'
WHERE t.table_schema = 'public'
    AND t.table_type = 'BASE TABLE'
ORDER BY t.table_name;
-- Expected: All tables show ✓

-- 4. Check foreign keys
SELECT 
    tc.table_name,
    tc.constraint_name,
    tc.constraint_type,
    kcu.column_name,
    ccu.table_name AS foreign_table_name,
    ccu.column_name AS foreign_column_name,
    rc.delete_rule
FROM information_schema.table_constraints tc
JOIN information_schema.key_column_usage kcu 
    ON tc.constraint_name = kcu.constraint_name
JOIN information_schema.constraint_column_usage ccu 
    ON ccu.constraint_name = tc.constraint_name
LEFT JOIN information_schema.referential_constraints rc
    ON rc.constraint_name = tc.constraint_name
WHERE tc.constraint_type = 'FOREIGN KEY'
ORDER BY tc.table_name, tc.constraint_name;
-- Expected: 13 foreign keys with appropriate CASCADE/RESTRICT/SET NULL rules

-- 5. Verify clustering columns exist
\d analysis_runs
-- Should see: autoencoder_*, dim_reduction_*, clustering_* columns

-- 6. Verify training_validation_splits table exists
\d training_validation_splits
-- Should show table structure with foreign keys

-- 7. Test a view
SELECT COUNT(*) FROM ground_truth_labels;
-- Should return 0 (no data loaded yet, but view should work)

\q
```

**If all checks pass:** ✅ Schema setup complete!

---

## Summary of What Was Created

### Tables (8 total):

| Table | Purpose | Migration |
|-------|---------|-----------|
| `images` | Core image metadata | Base schema |
| `ground_truth_history` | Labels with history tracking | Base schema |
| `prompts` | Reusable LLM prompts | Base schema |
| `analysis_runs` | Analysis execution metadata | Base schema + clustering migration |
| `llm_responses` | LLM outputs (parsed + raw) | Base schema |
| `predictions` | Model predictions (EAV format) | Base schema |
| `clustering_results` | Clustering algorithm outputs | Base schema |
| `training_validation_splits` | Train/val data splits | Migration 2 |

### Key Features:

✅ **Consistent timestamps** - All tables use `created_at`  
✅ **Clustering pipeline tracking** - Autoencoder + dim reduction + clustering columns  
✅ **Train/val split tracking** - Separate table for data split management  
✅ **Foreign keys with CASCADE** - Automatic cleanup on delete  
✅ **Indexes for performance** - Foreign keys, JSONB, and query optimization  
✅ **Analytical views** - Pre-built queries for common analysis tasks  

---

## Next Steps: Data Loading

Now that schema is set up, load data using ETL notebooks in this order:

### 1. Load Images
```python
# Notebook: db_etl_images.ipynb (or your equivalent)
loader = MLDataLoader(db_name='image_analysis_test')
loader.load_images_safe(image_ids, filenames, source='giub', file_paths=paths)
```

### 2. Load Ground Truth Labels
```python
# Notebook: db_etl_ground_truth.ipynb
loader.load_ground_truth(image_id, label_name, value)
```

### 3. Load Analysis Runs & Results

**For Clustering:**
```python
# Notebook: db_etl_clustering_applied.ipynb
analysis_run_id = loader.load_analysis_run(
    run_timestamp=datetime.now(),
    analysis_type='clustering',
    # Include clustering pipeline details
    autoencoder_name='...',
    dim_reduction_name='...',
    clustering_name='...',
    # ... etc
)

# Load clustering results
loader.load_clustering_results(analysis_run_id, image_ids, cluster_ids)
```

**For Classification (LLM/YOLO):**
```python
# Your classification ETL notebooks
analysis_run_id = loader.load_analysis_run(
    analysis_type='llm_classification',  # or 'yolo_classification'
    # clustering columns will be NULL
)

loader.load_predictions(analysis_run_id, predictions_data)
```

### 4. Load Training/Validation Splits
```python
# Notebook: db_etl_train_val_splits.ipynb
train_filepaths = pd.read_csv('train_data_file_paths.csv')['filepaths'].tolist()
val_filepaths = pd.read_csv('val_data_file_paths.csv')['filepaths'].tolist()

loader.load_training_validation_splits(train_filepaths, 'train', analysis_run_id)
loader.load_training_validation_splits(val_filepaths, 'validation', analysis_run_id)
```

---

## Troubleshooting

### Error: "relation already exists"

**Cause:** Schema already partially created  
**Solution:** Run Step 2 (Drop Existing Schema) to clean slate

### Error: "column does not exist" in views

**Cause:** Views created before timestamp standardization  
**Solution:** Recreate views:
```bash
psql $DB_NAME -f schema/create_views.sql
```

### Error: "foreign key constraint violation"

**Cause:** Trying to load data in wrong order  
**Solution:** Always load in order: images → ground_truth → analysis_runs → results

### Error: Migration already applied

**Cause:** Running migration twice  
**Solution:** Check which migrations have been applied:
```sql
-- Check if clustering columns exist
\d analysis_runs

-- Check if training_validation_splits exists
\dt training_validation_splits

-- Check if timestamps are standardized
\d ground_truth_history  -- should show created_at, not changed_at
```

### Schema doesn't match expected structure

**Cause:** Migrations run out of order or skipped  
**Solution:** Drop everything (Step 2) and start over from Step 3

---

## Alternative: Using pg_dump

**After all migrations are complete**, you can export the final schema:

```bash
# Export schema only (no data)
pg_dump -s -d image_analysis_dev -f schema/complete_schema_v2.1.sql
```

**Then use this file for future fresh setups:**

```bash
# One-step schema creation (replaces Steps 3-8)
psql image_analysis_test -f schema/complete_schema_v2.1.sql
```

**Advantages:**
- Single file
- Guaranteed correct order
- Exact replica of working schema

**Disadvantages:**
- Harder to see individual migrations
- Loses migration history
- Must regenerate if schema changes

---

## Files and Documentation Reference

### Schema Files (Execution Order):
1. `schema/create_tables.sql`
2. `schema/create_indexes.sql`
3. `schema/schema_migration/alter_analysis_runs_clustering.sql`
4. `schema/schema_migration/add_training_validation_splits.sql`
5. `schema/schema_migration/standardize_timestamp_columns.sql`
6. `schema/create_views.sql`

### Documentation:
- `schema_migration_clustering_columns.md` - Details on clustering pipeline columns
- `schema_migration_train_val_split.md` - Details on train/val splits table
- `timestamp_standardization_setup_guide.md` - Details on timestamp standardization

### ETL Notebooks (Data Loading):
- `db_etl_images.ipynb` - Load images
- `db_etl_ground_truth.ipynb` - Load labels
- `db_etl_clustering_applied.ipynb` - Load clustering results
- `db_etl_train_val_splits.ipynb` - Load train/val splits
- Your classification notebooks - Load classification results

---

## Schema Version History

**v2.1 (Current)** - 2026-03-22
- ✅ Base schema with 7 tables
- ✅ Clustering pipeline columns (autoencoder, dim reduction, clustering)
- ✅ Training/validation splits table
- ✅ Standardized timestamp columns (created_at)
- ✅ 6 analytical views
- ✅ 17+ performance indexes

**Previous versions:** See individual migration guides

---

## Support and Feedback

**Issues with this guide?**
- Check individual migration guides for detailed troubleshooting
- Verify PostgreSQL version (16.x required)
- Ensure all SQL files are present and uncorrupted

**Schema modifications needed?**
- Create new migration file in `schema/schema_migration/`
- Update this guide with new migration step
- Document in separate migration guide

---

*End of Complete Database Setup Guide v2.1*
