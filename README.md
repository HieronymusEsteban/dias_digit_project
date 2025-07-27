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

