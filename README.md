# Project:
This project is a collaboration with the University Library of Bern and the Institute of Geography of the university of Bern.

The goal of this project is to support the subject cataloging of old slide films from a collection of the Institute of Geography. 
The data set obtained from the University Library is referred to as "giub" (Geography Institute University Bern).
##------------------------------------------------------------------------------------------------------------------------------------------
##------------------------------------------------------------------------------------------------------------------------------------------
# Project workflows:
##------------------------------------------------------------------------------------------------------------------------------------------
##------------------------------------------------------------------------------------------------------------------------------------------
## Label preparation and people recognition with yolo and with MiniCPM-v (VLM: vision language model, in this project sometimes unprecisely referred to as LLM: large language model):
##------------------------------------------------------------------------------------------------------------------------------------------
### 1. load_transform_labels.ipynb:  
Loads a manually created label file and saves a modified version ("labels_mod.csv"). This only needs to be done once after the creation or manual modification of the original label file, after which the modified file is used by the following notebooks.
### 2. filter_out_people_multi_approach.ipynb (or as script: filter_out_people_multi_approach.py):
Recognises images with people first by using yolo then by using an LLM (MiniCPM-v). 
The output from the yolo analysis is stored in files (integers at the end of filename identify analysis run):
- 'people_detect_multi_approach_ml_metrics_yolo_20260309_214342.csv',
- 'people_detect_multi_approach_labels_results_yolo_20260309_214342.csv',
- 'times_people_detect_multi_approach_yolo_20260309_214342.pkl',
- 'conf_matrix_metrics_pers_recognisable_yolo.pdf',
- 'conf_matrix_metrics_pers_yolo.pdf'

The output from the llm analysis is stored in files (integers at the end of filename identify analysis run):
- 'people_detect_multi_approach_labels_results_llm_20260309_214410.csv',
- 'people_detect_multi_approach_ml_metrics_llm_20260309_214410.csv',
- 'minicpm_v_model_info.txt',
- 'responses_llm_people_detect_multi_approach_20260309_214410.pkl',
- 'results_llm_people_detect_multi_approach_20260309_214410.pkl',
- 'times_people_detect_multi_approach_llm_20260309_214410.pkl'
### 3. Load results into database:
db_etl_yolo.ipynb reads the results files ('people_detect_multi_approach_labels_results_yolo_20260309_214342.csv',  	'times_people_detect_multi_approach_yolo_20260309_214342.pkl') and loads the data into the database.
db_etl_llm.ipynb reads the results files ('responses_llm_people_detect_multi_approach_20260309_214410.pkl', 	'results_llm_people_detect_multi_approach_20260309_214410.pkl', 'times_people_detect_multi_approach_llm_20260309_214410.pkl') and 	loads the data into the database.
##------------------------------------------------------------------------------------------------------------------------------------------
##------------------------------------------------------------------------------------------------------------------------------------------
## Label preparation and recognition of image type, people, and objects with MiniCPM-v:
##------------------------------------------------------------------------------------------------------------------------------------------
### 1. load_transform_labels.ipynb:  
Loads a manually created label file and saves a modified version ("labels_mod.csv"). This only needs to be done once after the creation or manual modification of the original label file, after which the modified file is used by the following notebooks.
### 2. rec_multi_object_MiniCPM.ipynb (or as script rec_multi_object_MiniCPM.py):
Classifies images by image type, content of people or other objects, performance is measured by comparing results with labels in a modified label file ("labels_mod.csv"). The output is stored in (integers at the end of filename identify analysis run):
- 'responses_multi_object_struct_minicpm_20260309_174756.pkl',
- 'img_closer_insp_multi_object_struct_minicpm_20260309_174756.pkl',
- 'ml_metrics_multi_object_struct_minicpm_20260309_174756.csv',
- 'results_multi_object_struct_minicpm_20260309_174756.pkl',
- 'times_multi_object_struct_minicpm_20260309_174756.pkl'
### 3. db_etl_llm.ipynb:
Reads the results files ('responses_multi_object_struct_minicpm_20260309_174756.pkl', 'img_closer_insp_multi_object_struct_minicpm_20260309_174756.pkl', 'ml_metrics_multi_object_struct_minicpm_20260309_174756.csv', 'results_multi_object_struct_minicpm_20260309_174756.pkl', 'times_multi_object_struct_minicpm_20260309_174756.pkl') and loads the data into the database.
##------------------------------------------------------------------------------------------------------------------------------------------
##------------------------------------------------------------------------------------------------------------------------------------------
## Label preparation and recognition of image type, people, and objects with Qwen3-VL-8B-Instruct:
##------------------------------------------------------------------------------------------------------------------------------------------
### 1. load_transform_labels.ipynb:  
Loads a manually created label file and saves a modified version ("labels_mod.csv"). This only needs to be done once after the creation or manual modification of the original label file, after which the modified file is used by the following notebooks.
### 2. rec_multi_object_qwen3vl.ipynb (or as script rec_multi_object_qwen3vl.py):
Classifies images by image type, content of people or other objects, performance is measured by comparing results with labels in a modified label file ("labels_mod.csv"). The output is stored in (integers at the end of filename identify analysis run):
- 'responses_multi_object_struct_qwen3vl_20260328_185249.pkl',
- 'img_closer_insp_multi_object_struct_qwen3vl_20260328_185249.pkl',
- 'ml_metrics_multi_object_struct_qwen3vl_20260328_185249.csv',
- 'results_multi_object_struct_qwen3vl_20260328_185249.pkl',
- 'times_multi_object_struct_qwen3vl_20260328_185249.pkl'
### 3. db_etl_llm.ipynb:
Reads the results files ('responses_multi_object_struct_qwen3vl_20260328_185249.pkl', 'img_closer_insp_multi_object_struct_qwen3vl_20260328_185249.pkl', 'ml_metrics_multi_object_struct_qwen3vl_20260328_185249.csv', 'results_multi_object_struct_qwen3vl_20260328_185249.pkl', 'times_multi_object_struct_qwen3vl_20260328_185249.pkl') and loads the data into the database.
##------------------------------------------------------------------------------------------------------------------------------------------
##------------------------------------------------------------------------------------------------------------------------------------------
## Training autoencoder on combined data (visual genome and swisstopo) data for clustering:
##------------------------------------------------------------------------------------------------------------------------------------------
### 1. download_maps_swisstopo.ipynb: 
Downloads maps from swisstopo, extracts and saves patches as .jpg and saves a labels.csv file
containing image_ids, filepaths, and labels (with value 1 indicating all images are category 'map').
This notebook only needs to be executed once.
### 2. Explore_VisualGenome_Dataset.ipynb: 
Is used to explore visual genome metadata and to download and explore desired parts of the visual genome data. Once the desired data has been downloaded this notebook is not necessary for any analysis of the downloaded data anymore. 
### 3. visual_genome_to_pytorch.ipynb: 
Selects image paths of images with minimum_side_length or bigger, downsize them if necessary, and save images in data_proc_path. Select image_file_paths of images with desired object class based on visual genome meta data and save image_file_paths along with image_ids and class label in a labels.csv file.
### 4. combined_data_labels.ipynb: 
Takes the labels.csv file from both the visual genome and the swisstopo merges 
them with class label 'is_map' = 1 for all swisstopo images and = 0 for visual genome images.
### 5. img_to_pytorch_vae.ipynb: 
Takes the image_file_paths from labels.csv file, separates the file paths into training and validation set, and loads data into torchvision.dataset format and train variational convolutional autoencoder; then executes clustering.
The output is stored in: 
- combined_data/
	- 'results_clustering_pipeline_20260325_110746.pkl', 
	- 'times_clustering_pipeline_20260325_110746.pkl',
	- 'train_data_file_paths_20260325_110746.csv', 
	- 'val_data_file_paths_20260325_110746.csv'
	- 'var_conv_ae_20260325_110746/
		model_000.pth
		.
		.
		.
		model_059.pth'
IMPORTANT: Delete from early training epochs to save storage space!
### 6. db_etl_clustering_comb.ipynb: 
Reads the files 'results_clustering_pipeline_20260325_110746.pkl', 'times_clustering_pipeline_20260325_110746.pkl'
and loads data into database.
### 7. db_etl_train_val_splits.ipynb
Reads the files 'train_data_file_paths_20260325_110746.csv', 'val_data_file_paths_20260325_110746.csv'
and loads data (train and validation metadata) into the database.
##------------------------------------------------------------------------------------------------------------------------------------------
##------------------------------------------------------------------------------------------------------------------------------------------
## Apply pre-trained autoencoder on data for clustering:
##------------------------------------------------------------------------------------------------------------------------------------------
### 1. img_to_pytorch_apply.ipynb: 
Loads a pretrained autoencoder model and labels_new.csv (new labels added otherwise normal labels file),
adds the filepaths and data source (e.g. giub) to the labels data, and applies the pretrained autoencoder 
as well as a clustering procedure to the image files defined in the labels data. 
The output is stored in: 
- data/
	- times_clustering_20260324_223646.pkl
	- results_clustering_20260324_223646.pkl
### 2. db_etl_clustering_applied.ipynb:
Reads the files times_clustering_20260324_223646.pkl, results_clustering_20260324_223646.pkl and loads
the data into the database. 
##------------------------------------------------------------------------------------------------------------------------------------------
##------------------------------------------------------------------------------------------------------------------------------------------
## Training and applying autoencoder on the same data set 
(autoencoder not generalisable, only for clustering the data set in question):
##------------------------------------------------------------------------------------------------------------------------------------------
CAVE! Require labels_new.csv file with file paths and data source already added to it!
##------------------------------------------------------------------------------------------------------------------------------------------
### 1. img_to_pytorch_vae_nv.ipynb: 
Loads labels data from labels_new.csv file, trains an autoencoder on the image files defined in the labels data (training data = validation data), then applies trained autoencoder and clustering procedure to the same data set.
The output is stored in: 
combined_data/
- 'results_clustering_pipeline_20260325_201946.pkl', 
	- 'times_clustering_pipeline_20260325_201946.pkl',
	- 'train_data_file_paths_20260325_201946.csv', 
	- 'val_data_file_paths_20260325_201946.csv'
	- 'var_conv_ae_20260325_201946/
		- model_000.pth
		.
		.
		.
		model_059.pth'
IMPORTANT: Delete from early training epochs to save storage space!
### 2. db_etl_clustering_nv.ipynb:
Reads the files 'results_clustering_pipeline_20260325_201946.pkl', 
'times_clustering_pipeline_20260325_201946.pkl' and loads the data into the data base. Puts out
the analysis_run_id to be used with the next notebook!
### 3. db_etl_train_val_splits.ipynb:
Reads the files 'train_data_file_paths_20260325_201946.csv', 
'val_data_file_paths_20260325_201946.csv' and loads the training and validation metadata into the database.
IMPORTANT: Requires setting the analysis_run_id according to the results from db_etl_clustering_nv.ipynb.
##------------------------------------------------------------------------------------------------------------------------------------------
##------------------------------------------------------------------------------------------------------------------------------------------
## Training and using Yolo for object detection:
##------------------------------------------------------------------------------------------------------------------------------------------
### 1. Explore_VisualGenome_Dataset.ipynb: 
Is used to explore visual genome metadata and to download and explore desired parts of the visual genome data. Once the desired data has been downloaded this notebook is not necessary for any analysis of the downloaded data anymore. 
### 2. visual_genome_to_yolo.ipynb:
creating yolo compatible meta data files based on desired object classes. This process also selects the images to be used by creating meta data text files for them. IMPORTANT: meta data text files from previous analyses must be deleted first. The images downloaded in step one do not get changed, only the meta data files change and determine which images get used. A file structure for training and testing yolo is created and images to be used are copied to there and further processed: from coloured to black and white (grayscale), if desired aged image effects can be added. /
If necessary 
### vg_to_yolo_annotate_clean.ipynb:
can be used instead of  visual_genome_to_yolo.ipynb: This notebook also offers the possibility to add or change some annotations manually and save them in meta data text files.
### 3. TrainYolo_TryOut.ipynb:
Use the file structure prepared in the previous step, train yolo on visual genome data and test it on the data provided by the institute of Geography. 
##------------------------------------------------------------------------------------------------------------------------------------------
##------------------------------------------------------------------------------------------------------------------------------------------
## DEPRECATED WORKFLOW: Training autoencoder on visual genome data for clustering:
### 1. visual_genome_to_pytorch.ipynb: 
Selects image paths of images with minimum_side_length or bigger, downsize them if necessary, and save images in data_proc_path. Select image_file_paths of images with desired object class based on visual genome meta data and save image_file_paths along with image_ids and class label in a labels.csv file.
### 2. img_to_pytorch.ipynb: 
Takes the image_file_paths from labels.csv file, separates the file paths into training and validation set, and loads data into torchvision.dataset format and train convolutional autoencoder.
IMPORTANT: Delete from early training epochs to save storage space!
##------------------------------------------------------------------------------------------------------------------------------------------
## DEPRECATED WORKFLOW: Label preparation and recognition of image type, people, and objects with an LLM:
### 1. load_transform_labels.ipynb:  
Loads a manually created label file and saves a modified version ("labels_mod.csv"). This only needs to be done once after the creation or manual modification of the original label file, after which the modified file is used by the following notebooks.
### 2. rec_people_image_type_MiniCPM.ipynb: 
Classifies images by image type, content of people or other objects, performance is measured by comparing results with labels in a modified label file ("labels_mod.csv"). 
##------------------------------------------------------------------------------------------------------------------------------------------
## DEPRECATED WORKFLOW: Label preparation and people recognition with yolo:
### 1. load_transform_person_labels.ipynb: 
Loads a manually created person label file and saves a modified version ("with_without_person_mod.csv"). This only needs to be done once after the creation or manual modification of the original label file, after which the modified file is used by the following notebooks.
### 2. filter_out_people_try.ipynb: 
Separates images with people from images without people by using yolo, performance is measured by comparing results with labels in a modified label file ("with_without_person_mod.csv"). 
##------------------------------------------------------------------------------------------------------------------------------------------
## DEPRECATED WORKFLOW: Training autoencoder on images from Institute of Geography:
### img_to_pytorch_for_BernerOberland.ipynb: 
Converts .tif image files into .jpg image files, loads them into torchvision.dataset format, and trains some autoencoders (basic dense model) to test if the data format works with the training code.
IMPORTANT: Delete from early training epochs to save storage space!

##------------------------------------------------------------------------------------------------------------------------------------------
##------------------------------------------------------------------------------------------------------------------------------------------
# Database:
##------------------------------------------------------------------------------------------------------------------------------------------
##------------------------------------------------------------------------------------------------------------------------------------------
## Setup
##------------------------------------------------------------------------------------------------------------------------------------------
### 1. Prerequisites

- macOS with Homebrew installed
- Python 3.9+
- ~500 MB free disk space

### 2. Install PostgreSQL
```bash
# Install PostgreSQL 16
brew install postgresql@16

# Start PostgreSQL service
brew services start postgresql@16

# Add PostgreSQL to PATH (only need to do this once):
echo 'export PATH="$(brew --prefix postgresql@16)/bin:$PATH"' >> ~/.zshrc
# Apply to current terminal session
source ~/.zshrc

# Note: Future terminal sessions will automatically have PostgreSQL in PATH
# You only need 'source ~/.zshrc' if continuing in the same terminal

# Verify installation
psql --version
```

### 3. Clone the repository
```bash
git clone https://github.com/HieronymusEsteban/dias_digit_project.git
cd dias_digit_project
```

### 4. Set up Python virtual environment
```bash
# Create virtual environment
python3 -m venv dias_huggingface__venv

# Activate it
source dias_huggingface__venv/bin/activate

# Install dependencies
pip install -r requirements_huggingface.txt
```

### 5. Create database
```bash
# Create the database
createdb image_analysis_dev

```

### 6. Configure database connection
```bash
cd project_clean

# Copy environment template
cp .env.example .env


# Edit with your credentials (use nano, vim, or VS Code)
nano .env
```

Set these values in `.env`:
```bash
DB_NAME=image_analysis_dev
DB_USER=your_mac_username
DB_HOST=localhost
DB_PORT=5432
DB_PASSWORD=            # Leave empty for local Homebrew PostgreSQL
```

**⚠️ Never commit `.env` to Git - it contains credentials!**

### 7. Install database schema
```bash
# Run schema files in order (from project_clean directory)
psql image_analysis_dev -f schema/create_tables.sql
psql image_analysis_dev -f schema/create_indexes.sql
psql image_analysis_dev -f schema/create_views.sql
psql image_analysis_dev -f schema/schema_migration/alter_analysis_runs_clustering.sql
```

### 8. Load data via ETL notebooks

Run notebooks in this order:
```bash
jupyter lab
```

**1. Load LLM Classification Data:** `db_etl_llm.ipynb`
   - Loads giub TIF images
   - Loads ground truth labels
   - Loads MiniCPM-V predictions and responses

**2. Load YOLO Detection Data:** `db_etl_yolo.ipynb`
   - Reuses giub images (if already loaded)
   - Loads YOLO person detection predictions

**3. Load Clustering Data:** `db_etl_clustering.ipynb`
   - Loads Visual Genome JPG images
   - Loads clustering analysis results

### 9. Validate data loading (optional)

Run `validate_etl_data_loading.ipynb` to verify all data loaded correctly.

---

## Additional Resources

- **Detailed setup guide:** See `SETUP_GUIDE.md` for comprehensive installation instructions
- **Backup & restore:** See `BACKUP_RESTORE_GUIDE.md` for database backup procedures
- **Database documentation:** See `image_analysis_db_documentation.pdf` for complete schema reference
