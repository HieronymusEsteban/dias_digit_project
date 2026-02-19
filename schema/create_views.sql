-- =============================================================================
-- Image Analysis Database - Views
-- Created: 2026-01-24
-- =============================================================================

CREATE OR REPLACE VIEW ground_truth_labels AS
SELECT 
    i.image_id,
    i.filename,
    gt.label_name,
    gt.value,
    gt.changed_at
FROM images i
LEFT JOIN ground_truth_history gt 
    ON i.image_id = gt.image_id
WHERE gt.is_current = TRUE OR gt.is_current IS NULL;  -- Current values or no ground truth

COMMENT ON VIEW ground_truth_labels IS 
    'Current ground truth labels for all images (excludes historical changes)';



CREATE OR REPLACE VIEW ground_truth_wide AS
SELECT 
    i.image_id,
    i.source,
    i.source_image_id,
    i.filename,
    MAX(CASE WHEN gt.label_name = 'with_person' THEN gt.value END) as with_person,
    MAX(CASE WHEN gt.label_name = 'person_recognisable' THEN gt.value END) as person_recognisable,
    MAX(CASE WHEN gt.label_name = 'is_photo' THEN gt.value END) as is_photo,
    MAX(CASE WHEN gt.label_name = 'with_church' THEN gt.value END) as with_church,
    MAX(CASE WHEN gt.label_name = 'in_high_alpine_environment' THEN gt.value END) as in_high_alpine_environment
FROM images i
LEFT JOIN ground_truth_history gt 
    ON i.image_id = gt.image_id 
    AND gt.is_current = TRUE
GROUP BY i.image_id;

COMMENT ON VIEW ground_truth_wide IS 
    'Ground truth in wide format - one row per image with labels as columns';


CREATE OR REPLACE VIEW model_performance AS
SELECT 
    ar.model_name,
    ar.analysis_type,
    p.label_name,
    COUNT(*) as total_predictions,
    SUM(CASE WHEN p.predicted_value = gt.value THEN 1 ELSE 0 END) as correct_predictions,
    ROUND(
        100.0 * SUM(CASE WHEN p.predicted_value = gt.value THEN 1 ELSE 0 END) / COUNT(*),
        2
    ) as accuracy_percentage,
    AVG(p.confidence_score) as avg_confidence
FROM predictions p
JOIN analysis_runs ar ON p.analysis_run_id = ar.analysis_run_id
JOIN ground_truth_history gt 
    ON p.image_id = gt.image_id 
    AND p.label_name = gt.label_name 
    AND gt.is_current = TRUE  -- Only compare to current ground truth
GROUP BY ar.model_name, ar.analysis_type, p.label_name;

COMMENT ON VIEW model_performance IS 
    'Performance metrics (accuracy, confidence) by model and label';


CREATE OR REPLACE VIEW prompt_performance AS
SELECT 
    pr.prompt_name,
    p.label_name,
    ar.model_name,
    COUNT(*) as total_predictions,
    SUM(CASE WHEN p.predicted_value = gt.value THEN 1 ELSE 0 END) as correct_predictions,
    ROUND(
        100.0 * SUM(CASE WHEN p.predicted_value = gt.value THEN 1 ELSE 0 END) / COUNT(*),
        2
    ) as accuracy_percentage
FROM predictions p
JOIN prompts pr ON p.prompt_id = pr.prompt_id  -- Only includes predictions with prompts
JOIN analysis_runs ar ON p.analysis_run_id = ar.analysis_run_id
JOIN ground_truth_history gt 
    ON p.image_id = gt.image_id 
    AND p.label_name = gt.label_name 
    AND gt.is_current = TRUE
GROUP BY pr.prompt_name, p.label_name, ar.model_name;

COMMENT ON VIEW prompt_performance IS 
    'Compare prediction accuracy across different prompts';


CREATE OR REPLACE VIEW recent_analysis_runs AS
SELECT 
    ar.analysis_run_id,
    ar.run_timestamp,
    ar.model_name,
    ar.analysis_type,
    ar.images_processed,
    ar.duration_seconds,
    COUNT(DISTINCT p.image_id) as images_with_predictions,
    COUNT(p.prediction_id) as total_predictions
FROM analysis_runs ar
LEFT JOIN predictions p ON ar.analysis_run_id = p.analysis_run_id
GROUP BY 
    ar.analysis_run_id,
    ar.run_timestamp,
    ar.model_name,
    ar.analysis_type,
    ar.images_processed,
    ar.duration_seconds
ORDER BY ar.run_timestamp DESC
LIMIT 10;

COMMENT ON VIEW recent_analysis_runs IS 
    'Summary of the 10 most recent analysis runs';


CREATE OR REPLACE VIEW image_summary AS
SELECT 
    i.image_id,
    i.filename,
    COUNT(DISTINCT gt.label_name) as ground_truth_labels_count,
    COUNT(DISTINCT p.prediction_id) as total_predictions,
    COUNT(DISTINCT p.analysis_run_id) as analysis_runs_count
FROM images i
LEFT JOIN ground_truth_history gt 
    ON i.image_id = gt.image_id AND gt.is_current = TRUE
LEFT JOIN predictions p 
    ON i.image_id = p.image_id
GROUP BY i.image_id, i.filename
ORDER BY i.image_id;

COMMENT ON VIEW image_summary IS 
    'High-level summary of ground truth and predictions per image';


CREATE OR REPLACE VIEW llm_response_summary AS
SELECT 
    ar.model_name,
    pr.prompt_name,
    COUNT(*) as total_responses,
    SUM(CASE WHEN lr.parse_success THEN 1 ELSE 0 END) as successful_parses,
    ROUND(
        100.0 * SUM(CASE WHEN lr.parse_success THEN 1 ELSE 0 END) / COUNT(*),
        2
    ) as parse_success_rate,
    AVG(lr.tokens_used) as avg_tokens,
    SUM(lr.tokens_used) as total_tokens
FROM llm_responses lr
JOIN analysis_runs ar ON lr.analysis_run_id = ar.analysis_run_id
JOIN prompts pr ON lr.prompt_id = pr.prompt_id
GROUP BY ar.model_name, pr.prompt_name;

COMMENT ON VIEW llm_response_summary IS 
    'LLM performance metrics including parsing success and token usage';


DO $$
BEGIN
    RAISE NOTICE '=======================================================';
    RAISE NOTICE 'Views created successfully!';
    RAISE NOTICE '  1. ground_truth_labels - Current labels for all images';
    RAISE NOTICE '  2. model_performance - Accuracy metrics by model';
    RAISE NOTICE '  3. prompt_performance - Accuracy metrics by prompt';
    RAISE NOTICE '  4. recent_analysis_runs - Last 10 runs summary';
    RAISE NOTICE '  5. image_summary - Per-image statistics';
    RAISE NOTICE '  6. llm_response_summary - LLM usage and parsing stats';
    RAISE NOTICE '';
    RAISE NOTICE 'Total views: 6';
    RAISE NOTICE '';
    RAISE NOTICE 'NOTE: Views work with new images table structure';
    RAISE NOTICE '      (auto-increment ID + source tracking)';
    RAISE NOTICE '=======================================================';
END $$;
