# Schema Migration: Clustering Pipeline Columns

**Date:** 2026-02-07  
**Type:** ALTER TABLE (non-breaking change)  
**Status:** Pending execution

---

## Purpose

Add columns to the `analysis_runs` table to track details of clustering analysis pipelines, which typically involve three stages:
1. **Autoencoder** - Feature extraction using trained neural network
2. **Dimensionality Reduction** - Reduce feature space (e.g., TSNE, PCA, UMAP)
3. **Clustering** - Group similar items (e.g., K-Means, DBSCAN)

---

## New Columns Added

### Autoencoder Columns
| Column | Type | Description |
|--------|------|-------------|
| `autoencoder_name` | TEXT | Name/version identifier (e.g., "CNN_Autoencoder_v1") |
| `autoencoder_implementation` | TEXT | Implementation source (e.g., "pytorch.custom.ConvAutoencoder") |
| `autoencoder_file` | TEXT | Path to saved model file |
| `autoencoder_params` | JSONB | Training hyperparameters and architecture details |

### Dimensionality Reduction Columns
| Column | Type | Description |
|--------|------|-------------|
| `dim_reduction_name` | TEXT | Algorithm name (e.g., "TSNE", "PCA", "UMAP") |
| `dim_reduction_implementation` | TEXT | Library source (e.g., "sklearn.manifold.TSNE") |
| `dim_reduction_params` | JSONB | Algorithm parameters |

### Clustering Columns
| Column | Type | Description |
|--------|------|-------------|
| `clustering_name` | TEXT | Algorithm name (e.g., "K-Means", "DBSCAN") |
| `clustering_implementation` | TEXT | Library source (e.g., "sklearn.cluster.KMeans") |
| `clustering_params` | JSONB | Algorithm parameters |

---

## All Columns Are Nullable

These columns are only populated when relevant to the analysis type:
- **LLM classification** (MiniCPM): All clustering columns = NULL
- **Object detection** (YOLO): All clustering columns = NULL
- **Clustering analysis**: Populate relevant columns

---

## Example Usage

### Example 1: Clustering Analysis Run

```python
analysis_run_id = loader.load_analysis_run(
    run_timestamp=datetime.now(),
    analysis_type='clustering',
    model_name='clustering_pipeline_v1',
    
    # Autoencoder details
    autoencoder_name='CNN_Autoencoder_v1',
    autoencoder_implementation='pytorch.custom.ConvAutoencoder',
    autoencoder_file='/models/autoencoder_20260107.h5',
    autoencoder_params={
        'training_set_size': 4282,
        'validation_set_size': 1072,
        'code_size': 100,
        'noise_rate': 0.1,
        'input_size': [320, 320],
        'batch_size': 32,
        'feature_maps': 64,
        'architecture': {
            'type': 'CNN',
            'stride_1_layers': 1,
            'stride_2_layers': 6,
            'final_feature_map_size': [5, 5]
        },
        'loss_function': 'MSE',
        'learning_rate': 0.0009,
        'epochs': 60
    },
    
    # Dimensionality reduction details
    dim_reduction_name='TSNE',
    dim_reduction_implementation='sklearn.manifold.TSNE',
    dim_reduction_params={
        'perplexity': 1,
        'init': 'pca',
        'learning_rate': 'auto',
        'random_state': 42
    },
    
    # Clustering details
    clustering_name='K-Means',
    clustering_implementation='sklearn.cluster.KMeans',
    clustering_params={
        'n_clusters': 5,
        'random_state': 42
    }
)
```

### Example 2: Non-Clustering Analysis (YOLO)

```python
analysis_run_id = loader.load_analysis_run(
    run_timestamp=datetime.now(),
    analysis_type='object_detection',
    model_name='YOLOv8',
    model_version='8.0',
    
    # All clustering columns remain NULL
)
```

---

## Migration Steps

1. **Backup database** (optional but recommended)
   ```bash
   pg_dump image_analysis_dev > backup_before_migration.sql
   ```

2. **Execute ALTER TABLE script**
   ```bash
   psql -d image_analysis_dev -f alter_analysis_runs_clustering.sql
   ```

3. **Verify columns added**
   ```sql
   \d analysis_runs
   ```

4. **Update Python code**
   - Modify `MLDataLoader.load_analysis_run()` to accept new parameters
   - All new parameters should be optional (default to None)

---

## Impact Assessment

- ✅ **Non-breaking change**: Existing data unaffected
- ✅ **Backward compatible**: Existing code continues to work
- ✅ **No data loss**: All existing rows remain intact with NULL in new columns
- ✅ **No performance impact**: Columns are nullable, no indexes required

---

## Rollback Plan

If needed, remove columns:

```sql
ALTER TABLE analysis_runs
DROP COLUMN autoencoder_name,
DROP COLUMN autoencoder_implementation,
DROP COLUMN autoencoder_file,
DROP COLUMN autoencoder_params,
DROP COLUMN dim_reduction_name,
DROP COLUMN dim_reduction_implementation,
DROP COLUMN dim_reduction_params,
DROP COLUMN clustering_name,
DROP COLUMN clustering_implementation,
DROP COLUMN clustering_params;
```

---

## Future Improvements

- Create comprehensive schema documentation with all migrations tracked
- Consider version control for schema changes
- Add validation constraints if patterns emerge
- Consider indexing JSONB columns if frequent queries on parameters
