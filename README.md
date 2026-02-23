# Project:
This project is a collaboration with the University Library of Bern and the Institute of Geography of the university of Bern.

The goal of this project is to support the subject cataloging of old slide films from a collection of the Institute of Geography. 


# Project workflows:

## Label preparation and people recognition with yolo:
### 1. load_transform_person_labels.ipynb: 
Loads a manually created person label file and saves a modified version ("with_without_person_mod.csv"). This only needs to be done once after the creation or manual modification of the original label file, after which the modified file is used by the following notebooks.
### 2. filter_out_people_try.ipynb: 
Separates images with people from images without people by using yolo, performance is measured by comparing results with labels in a modified label file ("with_without_person_mod.csv"). 

## Label preparation and recognition of image type, people, and objects with an LLM:
### 1. load_transform_labels.ipynb:  
Loads a manually created label file and saves a modified version ("labels_mod.csv"). This only needs to be done once after the creation or manual modification of the original label file, after which the modified file is used by the following notebooks.
### 2. rec_people_image_type_MiniCPM.ipynb: 
Classifies images by image type, content of people or other objects, performance is measured by comparing results with labels in a modified label file ("labels_mod.csv"). 

## Training and using Yolo for object detection:
### 1. Explore_VisualGenome_Dataset.ipynb: 
Is used to explore visual genome metadata and to download and explore desired parts of the visual genome data. Once the desired data has been downloaded this notebook is not necessary for any analysis of the downloaded data anymore. 
### 2. visual_genome_to_yolo.ipynb 
creating yolo compatible meta data files based on desired object classes. This process also selects the images to be used by creating meta data text files for them. IMPORTANT: meta data text files from previous analyses must be deleted first. The images downloaded in step one do not get changed, only the meta data files change and determine which images get used. A file structure for training and testing yolo is created and images to be used are copied to there and further processed: from coloured to black and white (grayscale), if desired aged image effects can be added. /
If necessary 
### vg_to_yolo_annotate_clean.ipynb 
can be used instead of  visual_genome_to_yolo.ipynb: This notebook also offers the possibility to add or change some annotations manually and save them in meta data text files.
### 3. TrainYolo_TryOut.ipynb
Use the file structure prepared in the previous step, train yolo on visual genome data and test it on the data provided by the institute of Geography. 

## Training autoencoder on visual genome data for clustering:
### 1. visual_genome_to_pytorch.ipynb: Selects image paths of images with minimum_side_length or bigger, downsize them if necessary, and save images in data_proc_path. Select image_file_paths of images with desired object class based on visual genome meta data and save image_file_paths along with image_ids and class label in a labels.csv file.
### 2. img_to_pytorch.ipynb: Takes the image_file_paths from labels.csv file, separates the file paths into training and validation set, and loads data into torchvision.dataset format and train convolutional autoencoder.

## Training autoencoder on images from Institute of Geography:
### img_to_pytorch_for_BernerOberland.ipynb: Converts .tif image files into .jpg image files, loads them into torchvision.dataset format, and trains some autoencoders (basic dense model) to test if the data format works with the training code.


# Database:

## Setup

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

# Add PostgreSQL to PATH
echo 'export PATH="$(brew --prefix postgresql@16)/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

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
pip install -r requirements.txt
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
