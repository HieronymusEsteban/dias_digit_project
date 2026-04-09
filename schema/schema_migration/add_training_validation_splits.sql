-- Schema Migration: Add training_validation_splits table
-- Purpose: Track which images were used for training vs validation in clustering runs
-- Author: Claude
-- Date: 2026-03-18

-- Create training_validation_splits table
CREATE TABLE training_validation_splits (
    split_id SERIAL PRIMARY KEY,
    analysis_run_id INTEGER NOT NULL,
    image_id INTEGER NOT NULL,
    split_type TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign key constraints
    CONSTRAINT fk_training_validation_analysis_run
        FOREIGN KEY (analysis_run_id) 
        REFERENCES analysis_runs(analysis_run_id)
        ON DELETE CASCADE,
    
    CONSTRAINT fk_training_validation_image
        FOREIGN KEY (image_id)
        REFERENCES images(image_id)
        ON DELETE CASCADE
);

-- Create indexes for performance
CREATE INDEX idx_training_validation_analysis_run_split 
    ON training_validation_splits(analysis_run_id, split_type);

CREATE INDEX idx_training_validation_image 
    ON training_validation_splits(image_id);

-- Add comment for documentation
COMMENT ON TABLE training_validation_splits IS 
    'Tracks training/validation split assignments for clustering analysis runs';

COMMENT ON COLUMN training_validation_splits.split_type IS 
    'Type of split: typically "train" or "validation", but flexible for future use';
