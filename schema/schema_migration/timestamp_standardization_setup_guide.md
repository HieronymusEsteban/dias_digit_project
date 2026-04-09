# Timestamp Standardization - Setup Guide

**Migration:** `standardize_timestamp_columns.sql`  
**Date:** 2026-03-22  
**Purpose:** Standardize all timestamp columns to use `created_at` naming convention

---

## What This Migration Does

Makes timestamp column naming consistent across all database tables by:

1. **Adding `created_at`** to tables missing it
2. **Renaming** inconsistent timestamp columns to `created_at`
3. **Preserving** semantically different columns (like `run_timestamp`)

---

## Tables Affected

### Tables Modified:

| Table | Change | Old Column | New Column |
|-------|--------|------------|------------|
| `analysis_runs` | ADD | - | `created_at` |
| `clustering_results` | ADD | - | `created_at` |
| `ground_truth_history` | RENAME | `changed_at` | `created_at` |
| `llm_responses` | RENAME | `received_at` | `created_at` |

### Tables Already Correct (No Changes):

- `predictions` - already has `created_at`
- `training_validation_splits` - already has `created_at`
- `images` - timestamp column already correct
- `prompts` - timestamp column already correct

### Special Cases:

- **`analysis_runs.run_timestamp`** - NOT changed (different semantic meaning: when analysis executed, not when record created)

---

## When to Run This Migration

### Scenario A: Fresh Database Setup (RECOMMENDED)

**When:** Setting up test database from scratch or reloading all data

**Steps:**

1. **Create base schema**
   ```bash
   psql image_analysis_test -f schema/create_tables.sql
   ```

2. **Run timestamp standardization**
   ```bash
   psql image_analysis_test -f schema/schema_migration/standardize_timestamp_columns.sql
   ```

3. **Create views**
   ```bash
   psql image_analysis_test -f schema/create_views.sql
   ```

4. **Load data via ETL notebooks**
   - Images
   - Ground truth
   - Analysis runs
   - Results

5. **Verify**
   ```bash
   psql image_analysis_test -f schema/schema_migration/verify_timestamps.sql
   ```

**Result:** Clean database with consistent naming from the start

---

### Scenario B: Existing Database with Data to Preserve

**When:** Running on dev database with historical data you want to keep

**⚠️ WARNING:** This migration will:
- Set new `created_at` columns to CURRENT_TIMESTAMP (now)
- You'll LOSE original timestamp information for existing records

**Alternative Options:**

1. **Accept new timestamps** - Run migration as-is, all existing records get current timestamp
2. **Preserve timestamps** - Create custom migration that copies old values before renaming
3. **Fresh reload** - Export data, recreate schema, reimport (cleanest)

**If you choose Option 1:**
```bash
# Backup first!
pg_dump image_analysis_dev > backup_before_timestamp_migration.sql

# Run migration
psql image_analysis_dev -f schema/schema_migration/standardize_timestamp_columns.sql
```

---

## Complete Fresh Setup Process (Test Database)

This is the recommended path for moving from dev to test database.

### Step 1: Prepare

```bash
cd ~/Documents/CAS_AML/dias_digit_project
```

**Optional:** Backup existing test database
```bash
pg_dump image_analysis_test > backups/test_db_backup_$(date +%Y%m%d).sql
```

### Step 2: Drop Existing Tables (if any)

```bash
psql image_analysis_test
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

\q
```

### Step 3: Create Fresh Schema

```bash
# Base tables
psql image_analysis_test -f schema/create_tables.sql

# Add training/validation splits table
psql image_analysis_test -f schema/schema_migration/add_training_validation_splits.sql

# Standardize timestamps
psql image_analysis_test -f schema/schema_migration/standardize_timestamp_columns.sql

# Create views
psql image_analysis_test -f schema/create_views.sql
```

### Step 4: Verify Schema

```bash
psql image_analysis_test
```

```sql
-- Check all tables exist
\dt

-- Check timestamp columns
SELECT 
    table_name,
    column_name,
    data_type
FROM information_schema.columns
WHERE table_schema = 'public'
    AND (column_name LIKE '%created_at%' OR column_name = 'run_timestamp')
ORDER BY table_name, column_name;

\q
```

**Expected output:** All tables should have `created_at` column

### Step 5: Load Data

Use your ETL notebooks in this order:

1. **Load images**
   - Notebook: `db_etl_images.ipynb` (or equivalent)
   - Loads: `images` table

2. **Load ground truth**
   - Notebook: `db_etl_ground_truth.ipynb` (or equivalent)
   - Loads: `ground_truth_history` table

3. **Load analysis runs & results**
   - Notebook: Varies by analysis type
   - Clustering: `db_etl_clustering_applied.ipynb`
   - Classification: Your classification ETL notebook
   - Loads: `analysis_runs`, `clustering_results`, `predictions`, etc.

4. **Load training/validation splits**
   - Notebook: `db_etl_train_val_splits.ipynb`
   - Loads: `training_validation_splits` table

### Step 6: Verify Data Load

```bash
psql image_analysis_test
```

```sql
-- Check record counts
SELECT 'images' as table_name, COUNT(*) FROM images
UNION ALL
SELECT 'ground_truth_history', COUNT(*) FROM ground_truth_history
UNION ALL
SELECT 'analysis_runs', COUNT(*) FROM analysis_runs
UNION ALL
SELECT 'clustering_results', COUNT(*) FROM clustering_results
UNION ALL
SELECT 'predictions', COUNT(*) FROM predictions
UNION ALL
SELECT 'training_validation_splits', COUNT(*) FROM training_validation_splits;

-- Verify timestamps are populated
SELECT 
    'analysis_runs' as table_name,
    MIN(created_at) as earliest,
    MAX(created_at) as latest,
    COUNT(*) as total
FROM analysis_runs
UNION ALL
SELECT 
    'clustering_results',
    MIN(created_at),
    MAX(created_at),
    COUNT(*)
FROM clustering_results;

\q
```

---

## Rollback (If Needed)

If you need to undo this migration:

```sql
BEGIN;

-- Remove added columns
ALTER TABLE analysis_runs DROP COLUMN IF EXISTS created_at;
ALTER TABLE clustering_results DROP COLUMN IF EXISTS created_at;

-- Rename back to original names
ALTER TABLE ground_truth_history RENAME COLUMN created_at TO changed_at;
ALTER TABLE llm_responses RENAME COLUMN created_at TO received_at;

COMMIT;
```

**Note:** Rollback only works if you haven't reloaded data yet

---

## Impact on ETL Code

### Code That Needs Updates:

**None!** 

All tables now have `created_at` with `DEFAULT CURRENT_TIMESTAMP`, so:
- Existing INSERT statements work without modification
- New records automatically get current timestamp
- No ETL code changes required

### Optional: Explicit Timestamps

If you want to set specific timestamps during load:

```python
# Old code (still works)
loader.load_analysis_run(...)

# New code (if you want custom timestamp)
loader.load_analysis_run(..., created_at='2026-03-20 14:30:00')
```

---

## Verification Queries

After migration, run these to confirm success:

### Check Column Names

```sql
SELECT 
    table_name,
    column_name,
    data_type,
    column_default
FROM information_schema.columns
WHERE table_schema = 'public'
    AND (column_name LIKE '%created_at%' 
         OR column_name LIKE '%timestamp%'
         OR column_name LIKE '%received%'
         OR column_name LIKE '%changed%')
ORDER BY table_name, column_name;
```

**Expected:** Only `created_at` and `run_timestamp` columns (no `changed_at`, `received_at`)

### Check All Tables Have created_at

```sql
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
```

**Expected:** All tables show ✓

---

## Files Modified

**Schema files updated:**
- `schema/schema_migration/standardize_timestamp_columns.sql` (NEW)

**No changes needed to:**
- ETL notebooks (timestamps auto-populated)
- Views (column renames handled automatically)
- Existing queries (if you weren't using old column names)

---

## Common Issues

### Issue 1: Column already exists

**Error:** `column "created_at" of relation "X" already exists`

**Solution:** You've already run this migration. Check if columns were renamed correctly.

### Issue 2: Migration runs but timestamps are NULL

**Cause:** DEFAULT wasn't set or data loaded before migration

**Solution:** 
```sql
UPDATE table_name SET created_at = CURRENT_TIMESTAMP WHERE created_at IS NULL;
```

### Issue 3: Views broken after migration

**Cause:** Views reference old column names (`changed_at`, `received_at`)

**Solution:** Recreate views:
```bash
psql image_analysis_test -f schema/create_views.sql
```

---

## Summary

✅ **Consistent naming** - All tables use `created_at`  
✅ **Preserved semantics** - `run_timestamp` kept for its distinct meaning  
✅ **Zero ETL impact** - Auto-populated timestamps, no code changes needed  
✅ **Clean fresh start** - Perfect for dev → test migration  

**Recommendation:** Run this migration as part of fresh test database setup.

---

*End of Setup Guide*
