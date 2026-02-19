"""
Database loader module for ML image analysis project.
Contains MLDataLoader class for loading images, predictions, and ground truth.
"""

# Standard library imports
import os
import sys
from datetime import datetime
from pathlib import Path

# Database connection library
# psycopg2: PostgreSQL adapter for Python - handles all DB communication
import psycopg2
from psycopg2 import extras  # extras provides advanced features like Json adapter

# Data manipulation
import pandas as pd  # For handling CSV files and DataFrames

# Environment variable management
# python-dotenv: Loads database credentials from .env file (keeps passwords out of code)
from dotenv import load_dotenv

# Load environment variables from .env file
# This reads DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT from your .env file
load_dotenv()


class MLDataLoader:
    """
    Handles loading ML image analysis data into PostgreSQL database.
    Manages ID mapping between source IDs and database auto-increment IDs.
    """
    
    def __init__(self, db_name=None, source='giub'):
        """
        Initialize database connection and ID mapping dictionary.
        
        Parameters:
        -----------
        db_name : str, optional
            Database name (defaults to DB_NAME from .env file)
        source : str, default='giub'
            Dataset source identifier (e.g., 'giub', 'vg')
        """
        # Use provided db_name or fall back to environment variable
        self.db_name = db_name or os.getenv('DB_NAME')
        self.source = source
        
        # ID mapping: tracks relationship between your source IDs and database IDs
        # Format: {(source, source_image_id): database_image_id}
        # Example: {('giub', 1): 1, ('giub', 2): 2, ...}
        self.id_mapping = {}
        
        # Establish database connection
        # Connection parameters come from .env file
        self.conn = psycopg2.connect(
            dbname=self.db_name,
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            host=os.getenv('DB_HOST'),
            port=os.getenv('DB_PORT')
        )
        
        # Cursor: object used to execute SQL commands
        self.cur = self.conn.cursor()
        
        print(f"✅ Connected to database: {self.db_name}")
        print(f"📊 Source dataset: {self.source}")
    
    def load_images(self, image_ids, filenames, file_paths=None, source=None):
        """
        Load images into database and create ID mapping.
        
        Parameters:
        -----------
        image_ids : list of int
            Your original source image IDs (e.g., [1, 2, 3, ...])
        filenames : list of str
            Image filenames (e.g., ['BernerOberland001.tif', ...])
        file_paths : list of str, optional
            Full file paths (can be None for portability)
        source : str, optional
            Dataset source (defaults to self.source from __init__)
        
        Returns:
        --------
        dict : ID mapping {(source, source_image_id): database_image_id}
        
        How it works:
        -------------
        1. Inserts each image into the 'images' table
        2. PostgreSQL auto-assigns image_id (SERIAL primary key)
        3. We capture that assigned ID using RETURNING clause
        4. Build mapping: your_id -> database_id
        """
        source = source or self.source  # Use provided source or default
        
        # file_paths can be None - if not provided, create list of Nones
        if file_paths is None:
            file_paths = [None] * len(image_ids)
        
        print(f"📸 Loading {len(image_ids)} images (source: {source})...")
        
        # SQL INSERT statement
        # RETURNING image_id: PostgreSQL returns the auto-generated ID after insert
        insert_query = """
            INSERT INTO images (source, source_image_id, filename, file_path)
            VALUES (%s, %s, %s, %s)
            RETURNING image_id;
        """
        
        # Loop through all images and insert them one by one
        for src_id, fname, fpath in zip(image_ids, filenames, file_paths):
            # Execute insert with parameters (prevents SQL injection)
            # %s placeholders are safely replaced by psycopg2
            self.cur.execute(insert_query, (source, src_id, fname, fpath))
            
            # Fetch the returned database ID
            # fetchone() returns tuple, [0] gets first element (the image_id)
            db_id = self.cur.fetchone()[0]
            
            # Store mapping: (source, your_id) -> database_id
            # Example: ('giub', 1) -> 1, ('giub', 2) -> 2, etc.
            self.id_mapping[(source, src_id)] = db_id
        
        # Commit the transaction (make changes permanent)
        self.conn.commit()
        
        print(f"   ✅ {len(image_ids)} images loaded")
        print(f"   ✅ ID mapping created: {len(self.id_mapping)} entries")
        
        return self.id_mapping

    def load_images_safe(self, image_ids, filenames, file_paths=None, source=None):
        """
        Load images into database, skipping those that already exist.
        Handles both new and existing images, building ID mapping for all.
        
        Parameters:
        -----------
        image_ids : list of int
            Your original source image IDs (e.g., [1, 2, 3, ...])
        filenames : list of str
            Image filenames (e.g., ['BernerOberland001.tif', ...])
        file_paths : list of str, optional
            Full file paths (can be None for portability)
        source : str, optional
            Dataset source (defaults to self.source from __init__)
        
        Returns:
        --------
        dict : ID mapping {(source, source_image_id): database_image_id}
        
        Differences from load_images():
        -------------------------------
        - Skips images that already exist (instead of failing)
        - Reports how many new vs existing
        - Builds ID mapping for both new and existing images
        """
        source = source or self.source
        
        if file_paths is None:
            file_paths = [None] * len(image_ids)
        
        print(f"📸 Loading {len(image_ids)} images (source: {source}, safe mode)...")
        
        inserted_count = 0
        existing_count = 0
        inserted_files = []
        existing_files = []
        
        for src_id, fname, fpath in zip(image_ids, filenames, file_paths):
            # Try to insert - if duplicate, do nothing
            self.cur.execute("""
                INSERT INTO images (source, source_image_id, filename, file_path)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (filename) DO NOTHING
                RETURNING image_id
            """, (source, src_id, fname, fpath))
            
            result = self.cur.fetchone()
            
            if result:
                # Image was newly inserted
                db_id = result[0]
                inserted_count += 1
                inserted_files.append(fname)
            else:
                # Image already existed - fetch its existing ID
                self.cur.execute("""
                    SELECT image_id FROM images WHERE filename = %s
                """, (fname,))
                db_id = self.cur.fetchone()[0]
                existing_count += 1
                existing_files.append(fname)
            
            # Store mapping regardless of new/existing
            self.id_mapping[(source, src_id)] = db_id
        
        self.conn.commit()
        
        print(f"   ✅ {inserted_count} new images loaded")
        if existing_count > 0:
            print(f"   ℹ️  {existing_count} images already existed (reused)")
        print(f"   ✅ ID mapping created: {len(self.id_mapping)} entries")
        
        return {
            'id_mapping': self.id_mapping,
            'inserted_files': inserted_files,
            'existing_files': existing_files
        }

    def load_ground_truth(self, ground_truth_data, source=None):
        """
        Load ground truth labels into database.
        
        Parameters:
        -----------
        ground_truth_data : str or pd.DataFrame
            Either a CSV file path or a pandas DataFrame
            Expected columns: image_id, label_name, value
            
        source : str, optional
            Dataset source (defaults to self.source)
        
        How it works:
        -------------
        1. Reads ground truth data (CSV or DataFrame)
        2. Uses ID mapping to convert source_image_id -> database image_id
        3. Inserts into ground_truth_history table with is_current=TRUE
        """
        source = source or self.source
        
        # Load data: handle both CSV files and DataFrames
        if isinstance(ground_truth_data, str):
            # It's a file path - read CSV
            df = pd.read_csv(ground_truth_data)
            print(f"📋 Loading ground truth from: {ground_truth_data}")
        else:
            # It's already a DataFrame
            df = ground_truth_data
            print(f"📋 Loading ground truth from DataFrame")
        
        print(f"   Found {len(df)} ground truth entries")
        
        # SQL INSERT statement
        # Sets is_current=TRUE by default (enforced by partial unique index)
        insert_query = """
            INSERT INTO ground_truth_history 
                (image_id, label_name, value, is_current)
            VALUES (%s, %s, %s, TRUE);
        """
        
        inserted_count = 0
        skipped_count = 0
        
        # Process each row in the ground truth data
        for _, row in df.iterrows():
            src_image_id = row['image_id']  # Your original image ID
            label_name = row['label_name']
            value = str(row['value'])  # Convert to string (supports 'true'/'false'/class names)
            
            # Look up database ID using the mapping
            # Key: (source, source_image_id) -> Value: database_image_id
            db_image_id = self.id_mapping.get((source, src_image_id))
            
            if db_image_id is None:
                # Image not found in mapping - skip this entry
                skipped_count += 1
                continue
            
            # Insert ground truth with mapped database ID
            self.cur.execute(insert_query, (db_image_id, label_name, value))
            inserted_count += 1
        
        # Commit all inserts
        self.conn.commit()
        
        print(f"   ✅ {inserted_count} ground truth entries loaded")
        if skipped_count > 0:
            print(f"   ⚠️  {skipped_count} entries skipped (image not found in mapping)")


    def load_ground_truth_safe(self, ground_truth_data, source=None):
        """
        Load ground truth labels, skipping entries that already exist.
        Handles both new and existing ground truth entries.
        
        Parameters:
        -----------
        ground_truth_data : str or pd.DataFrame
            Either a CSV file path or a pandas DataFrame
            Expected columns: image_id, label_name, value
            
        source : str, optional
            Dataset source (defaults to self.source)
        
        Returns:
        --------
        dict : {'inserted': int, 'existing': int, 'skipped': int}
        
        Differences from load_ground_truth():
        ------------------------------------
        - Skips ground truth that already exists (instead of failing)
        - Returns dict with counts of inserted/existing/skipped
        """
        source = source or self.source
        
        # Load data: handle both CSV files and DataFrames
        if isinstance(ground_truth_data, str):
            df = pd.read_csv(ground_truth_data)
            print(f"📋 Loading ground truth from: {ground_truth_data}")
        else:
            df = ground_truth_data
            print(f"📋 Loading ground truth from DataFrame (safe mode)")
        
        print(f"   Found {len(df)} ground truth entries")
        
        # CHANGE 1: Modified INSERT statement with ON CONFLICT
        insert_query = """
            INSERT INTO ground_truth_history 
                (image_id, label_name, value, is_current)
            VALUES (%s, %s, %s, TRUE)
            ON CONFLICT (image_id, label_name) WHERE is_current = TRUE
            DO NOTHING
            RETURNING history_id;
        """
        
        # CHANGE 2: Track inserted vs existing
        inserted_count = 0
        existing_count = 0
        skipped_count = 0
        
        # Process each row in the ground truth data
        for _, row in df.iterrows():
            src_image_id = row['image_id']
            label_name = row['label_name']
            value = str(row['value'])
            
            db_image_id = self.id_mapping.get((source, src_image_id))
            
            if db_image_id is None:
                skipped_count += 1
                continue
            
            # CHANGE 3: Check if insert succeeded or conflicted
            self.cur.execute(insert_query, (db_image_id, label_name, value))
            result = self.cur.fetchone()
            
            if result:
                # New entry inserted
                inserted_count += 1
            else:
                # Entry already existed (conflict)
                existing_count += 1
        
        self.conn.commit()
        
        print(f"   ✅ {inserted_count} new ground truth entries loaded")
        if existing_count > 0:
            print(f"   ℹ️  {existing_count} entries already existed (reused)")
        if skipped_count > 0:
            print(f"   ⚠️  {skipped_count} entries skipped (image not found in mapping)")
        
        # CHANGE 4: Return dict with results
        return {
            'inserted': inserted_count,
            'existing': existing_count,
            'skipped': skipped_count
        }

    def load_predictions(self, predictions_dict, analysis_run_id, prompt_id=None, source=None):
        """
        Load predictions into database.
        
        Parameters:
        -----------
        predictions_dict : dict
            Nested dictionary: {image_id: {label_name: {'predicted_value': val, 'confidence_score': score}}}
            Example: {1: {'with_person': {'predicted_value': 'true', 'confidence_score': 0.95}}}
            
        analysis_run_id : int
            The database ID of the analysis run these predictions belong to
            
        prompt_id : int, optional
            The prompt ID if these are LLM predictions (None for non-LLM like YOLO)
            
        source : str, optional
            Dataset source (defaults to self.source)
        
        How it works:
        -------------
        1. Iterates through predictions dictionary
        2. Uses ID mapping to convert source_image_id -> database image_id
        3. Inserts each prediction with proper foreign key references
        """
        source = source or self.source
        
        print(f"🎯 Loading predictions for analysis_run_id={analysis_run_id}...")
        
        # SQL INSERT statement
        # prompt_id can be NULL for non-LLM predictions
        insert_query = """
            INSERT INTO predictions 
                (analysis_run_id, image_id, prompt_id, label_name, predicted_value, confidence_score)
            VALUES (%s, %s, %s, %s, %s, %s);
        """
        inserted_count = 0
        skipped_count = 0
        
        # Iterate through predictions: {image_id: {label: {pred, conf}}}
        for src_image_id, labels_dict in predictions_dict.items():
            # Look up database ID
            db_image_id = self.id_mapping.get((source, src_image_id))
            
            if db_image_id is None:
                # Image not in mapping - skip all predictions for this image
                skipped_count += len(labels_dict)
                continue
            
            # Process each label's prediction for this image
            for label_name, pred_data in labels_dict.items():
                predicted_value = str(pred_data['predicted_value'])
                confidence_score = pred_data.get('confidence_score')  # May be None
                
                # Insert prediction
                self.cur.execute(insert_query, (
                    analysis_run_id,
                    db_image_id,
                    prompt_id,  # NULL for non-LLM predictions
                    label_name,
                    predicted_value,
                    confidence_score
                ))
                inserted_count += 1
        
        # Commit all inserts
        self.conn.commit()
        
        print(f"   ✅ {inserted_count} predictions loaded")
        if skipped_count > 0:
            print(f"   ⚠️  {skipped_count} predictions skipped (image not found in mapping)")


    def load_llm_responses(self, responses_dict, analysis_run_id, prompt_id, source=None):
        """
        Load LLM responses into database.
        
        Parameters:
        -----------
        responses_dict : dict
            Dictionary: {image_id: {'parsed_response': dict, 'raw_response_text': str, 
                                    'parse_success': bool, 'tokens_used': int}}
            
        analysis_run_id : int
            The database ID of the analysis run
            
        prompt_id : int
            The prompt ID used for these LLM calls
            
        source : str, optional
            Dataset source (defaults to self.source)
        
        How it works:
        -------------
        1. Iterates through LLM response data
        2. Uses ID mapping to convert source_image_id -> database image_id
        3. Stores parsed JSON (JSONB) and raw text
        4. Tracks parsing success and token usage
        """
        source = source or self.source
        
        print(f"🤖 Loading LLM responses for analysis_run_id={analysis_run_id}...")
        
        # SQL INSERT statement
        # parsed_response is JSONB type - psycopg2 handles JSON conversion
        insert_query = """
            INSERT INTO llm_responses 
                (analysis_run_id, image_id, prompt_id, parsed_response, 
                raw_response_text, parse_success, tokens_used)
            VALUES (%s, %s, %s, %s, %s, %s, %s);
        """
        
        inserted_count = 0
        skipped_count = 0
        
        # Process each LLM response
        for src_image_id, response_data in responses_dict.items():
            # Look up database ID
            db_image_id = self.id_mapping.get((source, src_image_id))
            
            if db_image_id is None:
                # Image not in mapping - skip
                skipped_count += 1
                continue
            
            # Extract response components
            parsed_response = response_data.get('parsed_response')  # dict/JSON
            raw_response_text = response_data.get('raw_response_text')  # str
            parse_success = response_data.get('parse_success', False)  # bool
            tokens_used = response_data.get('tokens_used')  # int or None
            
            # Insert LLM response
            # extras.Json() converts Python dict to PostgreSQL JSONB
            self.cur.execute(insert_query, (
                analysis_run_id,
                db_image_id,
                prompt_id,
                extras.Json(parsed_response) if parsed_response else None,  # Convert to JSONB
                raw_response_text,
                parse_success,
                tokens_used
            ))
            inserted_count += 1
        
        # Commit all inserts
        self.conn.commit()
        
        print(f"   ✅ {inserted_count} LLM responses loaded")
        if skipped_count > 0:
            print(f"   ⚠️  {skipped_count} responses skipped (image not found in mapping)")


    def close(self):
        """
        Close database cursor and connection.
        
        Why this matters:
        ----------------
        - Releases database resources
        - Ensures all transactions are finalized
        - Prevents connection leaks
        - Good practice: always close connections when done
        """
        self.cur.close()
        self.conn.close()
        print("🔌 Database connection closed")


    def get_database_image_id(self, source, source_image_id):
        """
        Map from (source, source_image_id) to database image_id
        Uses cache (self.id_mapping) first, then queries database
        """
        # Check cache first
        cache_key = (source, source_image_id)
        if cache_key in self.id_mapping:
            return self.id_mapping[cache_key]
        
        # Query database if not in cache
        self.cur.execute("""
            SELECT image_id FROM images 
            WHERE source = %s AND source_image_id = %s
        """, (source, source_image_id))
        
        result = self.cur.fetchone()
        if not result:
            raise ValueError(f"Image not found: source={source}, source_image_id={source_image_id}")
        
        db_image_id = result[0]
        self.id_mapping[cache_key] = db_image_id
        return db_image_id


    def get_or_create_prompt(self, prompt_name, prompt_text):
        """
        Get existing prompt ID or create new prompt
        Checks if prompt_name already exists to avoid duplicates
        """
        # Check if exists
        self.cur.execute("""
            SELECT prompt_id FROM prompts WHERE prompt_name = %s
        """, (prompt_name,))
        
        result = self.cur.fetchone()
        if result:
            return result[0]
        
        # Create new if doesn't exist
        self.cur.execute("""
            INSERT INTO prompts (prompt_name, prompt_text)
            VALUES (%s, %s)
            RETURNING prompt_id
        """, (prompt_name, prompt_text))
        
        return self.cur.fetchone()[0]


    # def load_analysis_run(self, run_timestamp, analysis_type, model_name,
    #                     python_script=None, model_version=None, hyperparameters=None, 
    #                     notes=None, start_time=None, duration_seconds=None, images_processed=None):
    #     """
    #     Create analysis run entry
    #     Returns the analysis_run_id
    #     """
    #     self.cur.execute("""
    #         INSERT INTO analysis_runs (
    #             run_timestamp, analysis_type, model_name, python_script, model_version,
    #             hyperparameters, notes, start_time, duration_seconds, images_processed
    #         )
    #         VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    #         RETURNING analysis_run_id
    #     """, (run_timestamp, analysis_type, model_name, python_script, model_version,
    #         extras.Json(hyperparameters) if hyperparameters else None, 
    #         notes, start_time, duration_seconds, images_processed))
        
    #     return self.cur.fetchone()[0]
    


    def load_analysis_run(self, run_timestamp, analysis_type, model_name,
                    python_script=None, model_version=None, hyperparameters=None, 
                    notes=None, start_time=None, duration_seconds=None, images_processed=None,
                    # Clustering pipeline parameters (all optional)
                    autoencoder_name=None, autoencoder_implementation=None, 
                    autoencoder_file=None, autoencoder_params=None,
                    dim_reduction_name=None, dim_reduction_implementation=None, 
                    dim_reduction_params=None,
                    clustering_name=None, clustering_implementation=None, 
                    clustering_params=None):
        """
        Create analysis run entry
        
        Parameters:
        -----------
        run_timestamp : datetime
            When the analysis was run
        analysis_type : str
            Type of analysis (e.g., 'llm_classification', 'object_detection', 'clustering')
        model_name : str
            Main model identifier (e.g., 'MiniCPM-V-2.6', 'YOLOv8', 'Clustering Pipeline')
        python_script : str, optional
            Script filename used for analysis
        model_version : str, optional
            Model version identifier
        hyperparameters : dict, optional
            Model hyperparameters (stored as JSONB)
        notes : str, optional
            Additional notes about the run
        start_time : datetime, optional
            Analysis start time
        duration_seconds : float, optional
            How long the analysis took
        images_processed : int, optional
            Number of images analyzed
        
        Clustering Pipeline Parameters (only for clustering analysis):
        --------------------------------------------------------------
        autoencoder_name : str, optional
            Autoencoder model name (e.g., 'CNN_Autoencoder_v1')
        autoencoder_implementation : str, optional
            Implementation source (e.g., 'pytorch.custom.ConvAutoencoder')
        autoencoder_file : str, optional
            Path to saved model file
        autoencoder_params : dict, optional
            Autoencoder hyperparameters (stored as JSONB)
        dim_reduction_name : str, optional
            Dimensionality reduction method (e.g., 'TSNE', 'PCA', 'UMAP')
        dim_reduction_implementation : str, optional
            Implementation source (e.g., 'sklearn.manifold.TSNE')
        dim_reduction_params : dict, optional
            Dim reduction parameters (stored as JSONB)
        clustering_name : str, optional
            Clustering algorithm (e.g., 'K-Means', 'DBSCAN')
        clustering_implementation : str, optional
            Implementation source (e.g., 'sklearn.cluster.KMeans')
        clustering_params : dict, optional
            Clustering parameters (stored as JSONB)
        
        Returns:
        --------
        int : analysis_run_id
        """
        self.cur.execute("""
            INSERT INTO analysis_runs (
                run_timestamp, analysis_type, model_name, python_script, model_version,
                hyperparameters, notes, start_time, duration_seconds, images_processed,
                autoencoder_name, autoencoder_implementation, autoencoder_file, autoencoder_params,
                dim_reduction_name, dim_reduction_implementation, dim_reduction_params,
                clustering_name, clustering_implementation, clustering_params
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING analysis_run_id
        """, (
            run_timestamp, analysis_type, model_name, python_script, model_version,
            extras.Json(hyperparameters) if hyperparameters else None, 
            notes, start_time, duration_seconds, images_processed,
            autoencoder_name, autoencoder_implementation, autoencoder_file,
            extras.Json(autoencoder_params) if autoencoder_params else None,
            dim_reduction_name, dim_reduction_implementation,
            extras.Json(dim_reduction_params) if dim_reduction_params else None,
            clustering_name, clustering_implementation,
            extras.Json(clustering_params) if clustering_params else None
        ))
        
        self.conn.commit()
        
        return self.cur.fetchone()[0]

    

    def load_predictions_from_dict(self, analysis_run_id, results_dict, source=None):
    #def load_predictions_from_dict(self, analysis_run_id=None, results_dict, times_dict=None, source=None):
        """
        Load predictions from your results dictionary format

        Args:
            analysis_run_id: The database ID of the analysis run (create separately first)
            results_dict: Dict with structure:
                {timestamp_id: {
                    'prompt_id': str,
                    'prompt_text': str,
                    'predictions': {
                        label_name: DataFrame(image_id, label_pred)
                    }
                }}
            source: Dataset source (default: self.source)
        """
        if source is None:
            source = self.source
        
        print(f"\n🔄 Loading results dictionary (source: {source})...")
        
        for timestamp_id, run_data in results_dict.items():
            print(f"\n   Processing run: {timestamp_id}")
            
            # Get or create prompt
            prompt_name = run_data['prompt_id']
            prompt_text = run_data.get('prompt_text', '')
            prompt_id = self.get_or_create_prompt(prompt_name, prompt_text)
            print(f"      Prompt: {prompt_name} (ID: {prompt_id})")
            
            # Get timing info if available
            # if not analysis_run_id: 
            #     start_time = None
            #     duration_seconds = None
            #     if times_dict:
            #         # Find matching entry in times_dict
            #         for i, name in enumerate(times_dict.get('analysis_name', [])):
            #             if name == prompt_name:
            #                 start_time = times_dict['time_stamp_start'][i]
            #                 duration_seconds = times_dict['duration_seconds'][i]
            #                 break
                
            #     # Create analysis run
            #     run_timestamp = datetime.strptime(timestamp_id, '%Y%m%d_%H%M%S')
            #     analysis_run_id = self.load_analysis_run(
            #         run_timestamp=run_timestamp,
            #         analysis_type='llm_classification',
            #         model_name='MiniCPM',
            #         python_script='your_analysis_script.py',
            #         notes=f'Prompt: {prompt_name}',
            #         start_time=start_time,
            #         duration_seconds=duration_seconds
            #     )
            #     print(f"      Analysis run ID: {analysis_run_id}")
            
            # Load predictions
            predictions_dict = run_data.get('predictions', {})
            total_predictions = 0
            
            for dict_key, df in predictions_dict.items():
                # Get the prediction column (not image_id)
                pred_column = [col for col in df.columns if col != 'image_id'][0]
                
                # Extract label name (remove '_pred' suffix)
                label_name = pred_column.replace('_pred', '')
                
                # Insert predictions
                for _, row in df.iterrows():
                    try:
                        # CRITICAL: Map source_image_id to database image_id
                        source_img_id = int(row['image_id'])
                        db_image_id = self.get_database_image_id(source, source_img_id)
                        
                        # Convert prediction value: 0/1 → 'false'/'true'
                        pred_value = 'true' if str(row[pred_column]) == '1' else 'false'
                        
                        self.cur.execute("""
                            INSERT INTO predictions (
                                analysis_run_id, image_id, prompt_id, label_name, predicted_value
                            )
                            VALUES (%s, %s, %s, %s, %s)
                            ON CONFLICT (analysis_run_id, image_id, label_name, prompt_id) 
                            DO NOTHING
                        """, (analysis_run_id, db_image_id, prompt_id, label_name, pred_value))
                        total_predictions += 1
                    except ValueError as e:
                        print(f"      ⚠️  Skipping prediction for image_id={row['image_id']}: {e}")
                        continue
            
            print(f"      ✅ Loaded {total_predictions} predictions")
            self.conn.commit()


    def load_yolo_predictions(self, analysis_run_id, csv_file_or_dataframe, source=None):
        """
        Load YOLO predictions from CSV file
        
        Args:
            analysis_run_id: The database ID of the analysis run (create separately first)
            csv_file_or_dataframe: Path to CSV file or pandas DataFrame
            source: Dataset source (default: self.source)
        
        CSV Structure Expected:
            - image_id: Source image IDs (e.g., '002', '003')
            - with_person_pred: YOLO predictions (0 or 1)
            - Other columns ignored
        """
        if source is None:
            source = self.source
        
        # Load CSV
        if isinstance(csv_file_or_dataframe, str):
            df = pd.read_csv(csv_file_or_dataframe)
            print(f"📊 Loading YOLO predictions from: {csv_file_or_dataframe}")
        else:
            df = csv_file_or_dataframe
            print(f"📊 Loading YOLO predictions from DataFrame")
        
        print(f"   Found {len(df)} predictions")
        
        label_name = 'with_person'  # Fixed label for YOLO
        total_predictions = 0
        skipped_count = 0
        
        for _, row in df.iterrows():
            try:
                # Map source_image_id to database image_id
                source_img_id = int(row['image_id'])
                print('source_img_id:')
                print(source_img_id)
                db_image_id = self.get_database_image_id(source, source_img_id)
                
                # Convert prediction value: 0/1 → 'false'/'true'
                pred_value = 'true' if str(row['with_person_pred']) == '1' else 'false'
                
                # Insert prediction with NULL prompt_id
                self.cur.execute("""
                    INSERT INTO predictions (
                        analysis_run_id, image_id, prompt_id, label_name, predicted_value
                    )
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (analysis_run_id, image_id, label_name, prompt_id) 
                    DO NOTHING
                """, (analysis_run_id, db_image_id, None, label_name, pred_value))
                # Note: None becomes NULL in PostgreSQL ↑
                
                total_predictions += 1
                
            except ValueError as e:
                print(f"   ⚠️  Skipping prediction for image_id={row['image_id']}: {e}")
                skipped_count += 1
                continue
        
        self.conn.commit()
        
        print(f"   ✅ Loaded {total_predictions} YOLO predictions")
        if skipped_count > 0:
            print(f"   ⚠️  {skipped_count} predictions skipped (image not found)")

    def load_clustering_results(self, analysis_run_id, clustering_dataframe, source=None):
        """
        Load clustering results into database
        
        Parameters:
        -----------
        analysis_run_id : int
            The database ID of the analysis run (create separately first)
        clustering_dataframe : pd.DataFrame
            DataFrame with columns:
            - image_id: Source image IDs (e.g., 1, 2, 3...)
            - cluster_label: Cluster assignments (e.g., 0, 1, 2...)
            - feature_1, feature_2 (optional): For distance calculation
        source : str, optional
            Dataset source (defaults to self.source)
        
        Returns:
        --------
        dict : {'inserted': int, 'skipped': int}
        """
        if source is None:
            source = self.source
        
        print(f"📊 Loading clustering results (source: {source})...")
        print(f"   Found {len(clustering_dataframe)} cluster assignments")
        
        inserted_count = 0
        skipped_count = 0
        
        for _, row in clustering_dataframe.iterrows():
            try:
                # Map source_image_id to database image_id
                source_img_id = int(row['image_id'])
                db_image_id = self.get_database_image_id(source, source_img_id)
                
                # Get cluster assignment
                cluster_id = int(row['cluster_label'])
                
                # Insert clustering result
                self.cur.execute("""
                    INSERT INTO clustering_results (
                        analysis_run_id, image_id, cluster_id
                    )
                    VALUES (%s, %s, %s)
                """, (analysis_run_id, db_image_id, cluster_id))
                
                inserted_count += 1
                
            except ValueError as e:
                print(f"   ⚠️  Skipping image_id={row['image_id']}: {e}")
                skipped_count += 1
                continue
        
        self.conn.commit()
        
        print(f"   ✅ {inserted_count} clustering results loaded")
        if skipped_count > 0:
            print(f"   ⚠️  {skipped_count} results skipped (image not found)")
        
        return {
            'inserted': inserted_count,
            'skipped': skipped_count
        }


def delete_images(image_ids=None, source=None, filenames=None, conn=None, cur=None):
    """
    Delete specific images and all their dependencies from database.
    
    Parameters:
    -----------
    image_ids : list of int, optional
        Source image IDs to delete (e.g., [2, 3, 8])
    source : str, optional
        Dataset source (e.g., 'giub')
    filenames : list of str, optional
        Filenames to delete (e.g., ['BernerOberland002.tif'])
    conn : connection object, optional
        Existing database connection (if None, creates new one)
    cur : cursor object, optional
        Existing cursor (if None, creates new one)
    
    Note: Provide at least one of: image_ids, source, or filenames
    """
    import psycopg2
    from dotenv import load_dotenv
    import os
    
    # Use provided connection or create new one
    close_after = False
    if conn is None:
        load_dotenv()
        conn = psycopg2.connect(
            dbname=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            host=os.getenv('DB_HOST'),
            port=os.getenv('DB_PORT')
        )
        cur = conn.cursor()
        close_after = True
    elif cur is None:
        cur = conn.cursor()
    
    # Build WHERE clause
    conditions = []
    params = []
    
    if source:
        conditions.append("source = %s")
        params.append(source)
    
    if image_ids:
        conditions.append("source_image_id = ANY(%s)")
        params.append(image_ids)
    
    if filenames:
        conditions.append("filename = ANY(%s)")
        params.append(filenames)
    
    if not conditions:
        print("❌ Error: Must provide at least one parameter")
        if close_after:
            cur.close()
            conn.close()
        return
    
    where_clause = " AND ".join(conditions)
    query = f"DELETE FROM images WHERE {where_clause} RETURNING image_id, filename;"
    
    print(f"🗑️  Deleting images...")
    cur.execute(query, params)
    
    deleted = cur.fetchall()
    conn.commit()
    
    print(f"   ✅ Deleted {len(deleted)} images (CASCADE deleted related data)")
    for img_id, filename in deleted:
        print(f"      - {filename} (image_id: {img_id})")
    
    # Only close if we created the connection
    if close_after:
        cur.close()
        conn.close()