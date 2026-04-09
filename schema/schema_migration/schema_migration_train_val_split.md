# Schema Migration Guide: Training/Validation Splits Table

**Migration:** `add_training_validation_splits.sql`  
**Date:** 2026-03-18  
**Purpose:** Add table to track which images were used for training vs validation in clustering runs

---

## What This Migration Does

Adds a new table `training_validation_splits` to track training/validation data splits for clustering analysis runs.

**New Table:**
- `training_validation_splits` - Records which images belong to train vs validation sets for each analysis run

**No changes to existing tables** - this is a pure addition.

---

## Prerequisites

- PostgreSQL 16 installed and running
- Database `image_analysis_dev` (or target database) exists
- Existing tables: `analysis_runs`, `images`
- No existing data needs to be reloaded

---

## Migration Steps

### 1. Backup Database (Recommended)

```bash
cd ~/Documents/CAS_AML/dias_digit_project
pg_dump -d image_analysis_dev -F c -f db_backups/image_analysis_dev_backup_$(date +%Y%m%d_%H%M%S).backup
```

### 2. Run Migration

```bash
psql image_analysis_dev -f schema/schema_migration/add_training_validation_splits.sql
```

### 3. Verify Migration

```bash
psql image_analysis_dev
```

Then run:

```sql
-- Check table exists
\dt training_validation_splits

-- Check table structure
\d training_validation_splits

-- Check indexes
\di idx_training_validation_*

-- Exit
\q
```

Expected output: Table with 5 columns (split_id, analysis_run_id, image_id, split_type, created_at) and 2 indexes.

---

## Important Notes

### Can I Run This Without Reloading Data?

**YES!** This migration only adds a NEW table. It does NOT:
- Modify existing tables
- Change existing data
- Require reloading any existing data

**Why this works:**
- Foreign keys are **one-directional**: training_validation_splits references existing tables
- Existing tables (analysis_runs, images) don't need to know about the new table
- No changes to their structure or data

### Foreign Key Behavior

**ON DELETE CASCADE** is enabled:
- If an analysis_run is deleted → corresponding split assignments are automatically deleted
- If an image is deleted → corresponding split assignments are automatically deleted

This prevents orphaned records and maintains referential integrity.

---

## After Migration

### Load Training/Validation Split Data

Use the ETL notebook: `db_etl_train_val_splits.ipynb`

Or use the MLDataLoader method directly:

```python
from db_loader import MLDataLoader

loader = MLDataLoader(db_name='image_analysis_dev')

# Load training splits
train_filepaths = ['path/to/image1.jpg', 'path/to/image2.jpg']
loader.load_training_validation_splits(
    filepaths=train_filepaths,
    split_type='train',
    analysis_run_id=4
)

# Load validation splits
val_filepaths = ['path/to/image3.jpg', 'path/to/image4.jpg']
loader.load_training_validation_splits(
    filepaths=val_filepaths,
    split_type='validation',
    analysis_run_id=4
)

loader.close()
```

---

## Rollback (If Needed)

If you need to undo this migration:

```sql
-- Drop indexes
DROP INDEX IF EXISTS idx_training_validation_analysis_run_split;
DROP INDEX IF EXISTS idx_training_validation_image;

-- Drop table (CASCADE removes foreign key constraints automatically)
DROP TABLE IF EXISTS training_validation_splits CASCADE;
```

---

## Verification Queries

### Check split counts per analysis run

```sql
SELECT 
    analysis_run_id,
    split_type,
    COUNT(*) as count
FROM training_validation_splits
GROUP BY analysis_run_id, split_type
ORDER BY analysis_run_id, split_type;
```

### Calculate split ratios

```sql
SELECT 
    analysis_run_id,
    COUNT(*) FILTER (WHERE split_type = 'train') as train_count,
    COUNT(*) FILTER (WHERE split_type = 'validation') as val_count,
    ROUND(
        COUNT(*) FILTER (WHERE split_type = 'train')::NUMERIC / COUNT(*) * 100, 
        1
    ) as train_pct,
    ROUND(
        COUNT(*) FILTER (WHERE split_type = 'validation')::NUMERIC / COUNT(*) * 100, 
        1
    ) as val_pct
FROM training_validation_splits
GROUP BY analysis_run_id
ORDER BY analysis_run_id;
```

### View sample data

```sql
SELECT 
    tv.split_id,
    tv.analysis_run_id,
    tv.split_type,
    i.source,
    i.filename,
    ar.model_name
FROM training_validation_splits tv
JOIN images i ON tv.image_id = i.image_id
JOIN analysis_runs ar ON tv.analysis_run_id = ar.analysis_run_id
LIMIT 10;
```

---

## Troubleshooting

### Error: relation "analysis_runs" does not exist

**Cause:** Core schema not installed  
**Solution:** Run base schema files first:
```bash
psql image_analysis_dev -f schema/create_tables.sql
```

### Error: duplicate key value violates unique constraint

**Cause:** Trying to insert same split assignment twice  
**Solution:** Check if data already loaded for this analysis_run_id:
```sql
SELECT COUNT(*) FROM training_validation_splits WHERE analysis_run_id = YOUR_ID;
```

### Error: insert or update on table violates foreign key constraint

**Cause:** Referenced analysis_run_id or image_id doesn't exist  
**Solution:** 
- Verify analysis run exists: `SELECT * FROM analysis_runs WHERE analysis_run_id = YOUR_ID;`
- Verify images are loaded: `SELECT COUNT(*) FROM images;`

---

## Files Modified/Created

**Created:**
- `schema/schema_migration/add_training_validation_splits.sql` - Migration SQL
- `db_loader.py` - Added `load_training_validation_splits()` method
- `db_etl_train_val_splits.ipynb` - ETL notebook for loading split data

**No files modified** - all changes are additions.

---

## Summary

✅ Safe to run without reloading existing data  
✅ Only adds new table, doesn't modify existing tables  
✅ Foreign keys point TO existing tables (one-directional)  
✅ Indexes added for query performance  
✅ CASCADE delete prevents orphaned records  
✅ Rollback available if needed
