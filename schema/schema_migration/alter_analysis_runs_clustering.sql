-- ============================================================================
-- ALTER TABLE: Add Clustering Pipeline Columns to analysis_runs
-- ============================================================================
-- Date: 2026-02-07
-- Purpose: Add columns to track autoencoder, dimensionality reduction, 
--          and clustering algorithm details for clustering analysis workflows
-- 
-- All columns are nullable (used only when relevant to the analysis type)
-- ============================================================================

ALTER TABLE analysis_runs 

-- Autoencoder Model (for feature extraction)
ADD COLUMN autoencoder_name TEXT,
ADD COLUMN autoencoder_implementation TEXT,
ADD COLUMN autoencoder_file TEXT,
ADD COLUMN autoencoder_params JSONB,

-- Dimensionality Reduction Algorithm
ADD COLUMN dim_reduction_name TEXT,
ADD COLUMN dim_reduction_implementation TEXT,
ADD COLUMN dim_reduction_params JSONB,

-- Clustering Algorithm
ADD COLUMN clustering_name TEXT,
ADD COLUMN clustering_implementation TEXT,
ADD COLUMN clustering_params JSONB;

-- ============================================================================
-- Add comments to document column purposes
-- ============================================================================

COMMENT ON COLUMN analysis_runs.autoencoder_name IS 'Name/version of trained autoencoder model (e.g., "CNN_Autoencoder_v1")';
COMMENT ON COLUMN analysis_runs.autoencoder_implementation IS 'Implementation details (e.g., "pytorch.custom.ConvAutoencoder")';
COMMENT ON COLUMN analysis_runs.autoencoder_file IS 'Path to saved model file (e.g., "/models/autoencoder_20260107.h5")';
COMMENT ON COLUMN analysis_runs.autoencoder_params IS 'Training hyperparameters and architecture details as JSON';

COMMENT ON COLUMN analysis_runs.dim_reduction_name IS 'Dimensionality reduction method (e.g., "TSNE", "PCA", "UMAP")';
COMMENT ON COLUMN analysis_runs.dim_reduction_implementation IS 'Implementation source (e.g., "sklearn.manifold.TSNE")';
COMMENT ON COLUMN analysis_runs.dim_reduction_params IS 'Algorithm parameters as JSON (e.g., perplexity, learning_rate)';

COMMENT ON COLUMN analysis_runs.clustering_name IS 'Clustering algorithm (e.g., "K-Means", "DBSCAN", "Hierarchical")';
COMMENT ON COLUMN analysis_runs.clustering_implementation IS 'Implementation source (e.g., "sklearn.cluster.KMeans")';
COMMENT ON COLUMN analysis_runs.clustering_params IS 'Algorithm parameters as JSON (e.g., n_clusters, random_state)';

-- ============================================================================
-- Verify changes
-- ============================================================================
-- Run this to verify columns were added:
-- \d analysis_runs
-- ============================================================================
