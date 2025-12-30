#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import shutil
import pandas as pd
from source import image_id_converter as img_idc
from source import sort_img_files as sif
from source import llm_input as llm_i
from source import llm_output as llm_o
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np


# In[2]:


import ollama
import json
import re
import pickle


# In[3]:


os.getcwd()


# # Using LLM (mini-CPM) for image analysis

# ### Define Functions:

# In[4]:


def create_analysis_prompt():
    """Create the structured prompt for image analysis."""
    return """
    Analyze this image and return ONLY a Python dictionary in exactly this format:
    
    {
        'image_is_photography': X,  # True if the image is a photography, False otherwise
        'additional_comments': '' # Any additional observations or empty string if none
    }
    
    Replace X with True (image is a photography) or False (image is not a photography).
    Return ONLY the dictionary, no other text.
    Your answer MUST have the exact structue of the dictionary described above (all keys MUST be present). 
    If you cannot answer the question in the way implied by this structure, enter 'None' as value and offer 
    your answer and explanations under 'additional_comments'.
    """


# In[5]:


type(True)


# In[ ]:





# In[6]:


# More comprehensive check including empty strings and whitespace
def has_missing_comprehensive(df):
   # Standard missing values
   has_standard_missing = df.isnull().any().any()
   
   # Empty strings and whitespace-only strings
   has_empty_strings = False
   for col in df.select_dtypes(include=['object']):
       if (df[col].astype(str).str.strip() == '').any():
           has_empty_strings = True
           break
   
   return has_standard_missing or has_empty_strings


# In[7]:


def save_conf_matrix_tag(labels_results, label, prediction, cases, filename_tag):
    true_positives, false_positives, true_negatives, false_negatives, positives, negatives = cases
    # Calculate confusion matrix
    cm = confusion_matrix(labels_results[label], labels_results[prediction])
    
    number_true_positives = true_positives.shape[0]
    number_false_positives = false_positives.shape[0]
    number_true_negatives = true_negatives.shape[0]
    number_false_negatives = false_negatives.shape[0]
    
    sensitivity = number_true_positives / positives.shape[0]
    specificity = number_true_negatives / negatives.shape[0]
    
    if (number_true_positives > 0) and (number_false_positives > 0):
        precision = number_true_positives / (number_true_positives + number_false_positives)
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)
    else:
        precision = None
        f1_score = None
        
    if positives.shape[0] > 0:
        miss_rate = number_false_negatives / positives.shape[0]
    else:
        miss_rate = None
    
    
    plt.figure(figsize=(15,8))
    gs = plt.GridSpec(1, 2, width_ratios=[2, 1])
    
    plt.subplot(gs[0])
    confusion_matrix_data = [[number_true_negatives, number_false_positives], 
                             [number_false_negatives, number_true_positives]]
    heatmap = sns.heatmap(confusion_matrix_data, annot=True, fmt='d', 
               xticklabels=['Predicted Negative', 'Predicted Positive'], 
               yticklabels=['Actual Negative', 'Actual Positive'],
               cbar_kws={'label': 'Number of Instances'})
    plt.title('Confusion Matrix')
    
    plt.subplot(gs[1])
    plt.axis('off')
    if precision is not None:
        metrics_text = (f'Performance Metrics:\n\n'
                       f'True Positives: {number_true_positives}\n'
                       f'False Positives: {number_false_positives}\n'
                       f'True Negatives: {number_true_negatives}\n'
                       f'False Negatives: {number_false_negatives}\n\n'
                       f'Sensitivity: {sensitivity:.4f}\n'
                       f'Specificity: {specificity:.4f}\n'
                       f'Precision: {precision:.4f}\n'
                       f'Miss Rate: {miss_rate:.4f}\n'
                       f'F1 Score: {f1_score:.4f}')
    else:
        metrics_text = (f'Performance Metrics:\n\n'
                       f'True Positives: {number_true_positives}\n'
                       f'False Positives: {number_false_positives}\n'
                       f'True Negatives: {number_true_negatives}\n'
                       f'False Negatives: {number_false_negatives}\n\n'
                       f'Sensitivity: {sensitivity:.4f}\n'
                       f'Specificity: {specificity:.4f}\n')
        
    plt.text(0, 0.5, metrics_text, fontsize=10, 
            verticalalignment='center')
    
    plt.suptitle('Photography Detection: Confusion Matrix and Performance Metrics Based on is_photo Label as Ground Truth', fontsize=16)
    plt.tight_layout()
    filename = 'conf_matrix_metrics_' + filename_tag + '.pdf'
    output_path = data_path / filename
    plt.savefig(output_path)
    plt.close()


# In[8]:


def store_duration(time_analysis_dict, time_analysis_for_df_dict, analysis_name, duration,
                  timestamp_start_is_photo_analysis,
                  timestamp_end_is_photo_analysis):
    time_analysis_dict[analysis_name] = {}
    time_analysis_dict[analysis_name]['duration_str'] = f"Analysis took: {duration}"
    time_analysis_dict[analysis_name]['duration_seconds'] = total_seconds
    time_analysis_dict[analysis_name]['duration_seconds_str'] = f"Analysis took: {total_seconds:.2f} seconds"
    time_analysis_dict[analysis_name]['duration_minutes'] = total_seconds/60
    time_analysis_dict[analysis_name]['duration_minutes_str'] = f"Analysis took: {total_seconds/60:.2f} minutes"
    time_analysis_dict[analysis_name]['time_stamp_start'] = timestamp_start_is_photo_analysis
    time_analysis_dict[analysis_name]['time_stamp_end'] = timestamp_end_is_photo_analysis

    time_analysis_for_df_dict['analysis_name'].append(analysis_name)
    time_analysis_for_df_dict['time_stamp_start'].append(timestamp_start_is_photo_analysis)
    time_analysis_for_df_dict['duration_str'].append(f"Analysis took: {duration}")
    time_analysis_for_df_dict['duration_seconds'].append(total_seconds)
    time_analysis_for_df_dict['duration_seconds_str'].append(f"Analysis took: {total_seconds:.2f} seconds")
    time_analysis_for_df_dict['duration_minutes'].append(total_seconds/60)
    time_analysis_for_df_dict['duration_minutes_str'].append(f"Analysis took: {total_seconds/60:.2f} minutes")

    return time_analysis_dict, time_analysis_for_df_dict
    


# In[ ]:





# In[9]:


def analyse_giub_img_dir_llm(jpg_data_path, create_analysis_prompt, model_function):
    # Get time stamp:
    timestamp_start_is_photo_analysis = pd.Timestamp.now()
    
    # Get list of image files to analyse: 
    image_files = os.listdir(jpg_data_path)
    img_ids = [image_file.split('Oberland')[1].split('.')[0] for image_file in image_files]
    
    # Make empty dictionary to store results:
    image_descr = {}
    
    # Loop through images to get answers: 
    for image_file in image_files:
        image_path = jpg_data_path / image_file
        path_str = str(image_path)
        #print('\n')
        #print(path_str)
        parts = path_str.split('.jpg')
        img_id = parts[-2][-3:]
    
        # Analyse image, get values for each of the categorical variables specified above:
        #image_description = analyze_image_structured(image_path)
        #image_description = llm_o.analyze_image_structured(image_path, create_analysis_prompt)
        image_description = llm_o.analyze_image_structured(image_path, create_analysis_prompt, model_function)
        
        dict_type_bool = type(image_description) == dict
            
        #print(image_description)
        image_descr[img_id] = image_description
    
    timestamp_end_is_photo_analysis = pd.Timestamp.now()

    return timestamp_start_is_photo_analysis, timestamp_end_is_photo_analysis, image_descr
    


# ### Choose LLM Model

# In[10]:


model_function = llm_i.call_minicpm_model


# ### Prepare empty dictionary for time analyses and get time stamp for overall workflow duration:

# In[11]:


time_analyses = {}
time_analyses_for_df = {}
time_analyses_for_df['analysis_name'] = []
time_analyses_for_df['time_stamp_start'] = []
time_analyses_for_df['duration_str'] = []
time_analyses_for_df['duration_seconds'] = []
time_analyses_for_df['duration_seconds_str'] = []
time_analyses_for_df['duration_minutes'] = []
time_analyses_for_df['duration_minutes_str'] = []

timestamp_start_workflow = pd.Timestamp.now()
timestamp_start_workflow


# ### Prepare empty dictionary to store the different response dictionaries:

# In[12]:


response_dictionaries = {}


# ### Prepare empty dictionary for cases with unstructured answers for visual inspection:

# In[13]:


images_closer_inspection = {}


# ### Prepare empty dictionary for result dataframes:

# In[14]:


results_tabular = {}


# In[ ]:





# In[15]:


ml_metrics = pd.DataFrame({'positives': [],
              'negatives': [], 
              'true_positives': [], 
              'true_negatives': [],
              'false_negatives': [], 
              'false_positives': [], 
              'sensitivity': [], 
              'specificity': []})


# ### Set paths:

# In[16]:


#root_path = Path('/Users/stephanehess/Documents/CAS_AML/dias_digit_project')
#root_path = Path('/Users/stephanehess/Documents/CAS_AML/dias_digit_project/test_yolo_object_train')

project_path = Path.cwd()
#root_path = (project_path / 'test_LLM_prompt_experiments').resolve()
root_path = project_path
data_path = root_path / 'data'
tif_data_path = root_path / 'data_1'
#data_path = root_path / 'visual_genome_data_all'
jpg_data_path = root_path / 'data_jpg'
#yolo_path = root_path / 'visual_genome_yolo_all'
output_dir_not_photo = root_path / 'not_photo'
output_dir_with_person = root_path / 'with_person'
output_dir_without_person = root_path / 'without_person'



# ### Create directories for sorting the images:

# In[17]:


# Create output directories
os.makedirs(data_path, exist_ok=True)
os.makedirs(tif_data_path, exist_ok=True)
os.makedirs(jpg_data_path, exist_ok=True)
os.makedirs(output_dir_not_photo, exist_ok=True)
os.makedirs(output_dir_with_person, exist_ok=True)
os.makedirs(output_dir_without_person, exist_ok=True)
#os.chdir('root_path')


# In[ ]:





# ### Copy and convert image files from tif_data_path to jpg_data_path:

# In[18]:


source_folder = tif_data_path
destination_folder = jpg_data_path

llm_i.convert_tif_to_jpg(source_folder, destination_folder, quality=100)


# In[ ]:





# ### Load person label data (ground truth) to compare to LLM responses:

# The file with_without_person.csv contains labels added by (human) visual inspection that represent the ground truth. 
#  * Column with_person: whether or not any person is in the image.
#  * Column recognisable: whether any person that would be recognisable to a human familiar with said person is in the image.
#  * Column church: whether the image contains a church.
#  * Column is_photo: whether the image is a photography or something else. (this formulation is, I guess, unprecise, as most dias can probably be called a photography of sorts (if a dia shows a painting, I assume a photograph of the painting has been taken), so, to be precise: whether or not the image is showing anything that exists in the real world or is showing a representation of anything that exists in the real world or aspects thereof.

# In[19]:


label_data = pd.read_csv(data_path/'labels_mod.csv')
label_data.head()


# In[20]:


# Reconvert image ids to integers (e.g. '234') as strings from the form they were saved in (e.g. 'id234' 
# to ensure string data type to deal with duck typing): 
img_ids = list(label_data.image_id)
label_data['image_id'] = img_idc.reconvert_image_ids(img_ids)
label_data.head()


# ### The following cell is only required for the test run on the test data: 

# In[21]:


# Select only rows referring to images in the smaller data set (test run):

# Make sure no .DS_Store file is included in jpg_data_path: 
import os
ds_file_path = jpg_data_path / '.DS_Store'

# Remove a specific .DS_Store file
if os.path.exists(ds_file_path):
    os.remove(ds_file_path)
    print("Removed .DS_Store")
else:
    print(".DS_Store not found")

# Find all .ipynb_checkpoints directories
for checkpoint_dir in jpg_data_path.rglob('.ipynb_checkpoints'):
    if checkpoint_dir.is_dir():
        print(f"Removing: {checkpoint_dir}")
        shutil.rmtree(checkpoint_dir)



# Get list of image files present:
image_files = os.listdir(jpg_data_path)

#image_files.remove(".ipynb_checkpoints")



# Extract image ids from image file names:
img_ids = [image_file.split('Oberland')[1].split('.')[0] for image_file in image_files]
img_ids.sort()
print(img_ids)

# Select relevant rows from label_data data frame by id list: 
select_bools = [img_id in img_ids for img_id in label_data.image_id]

label_data = label_data[select_bools].copy()
label_data


# In[ ]:





# ### Prepare variations of LLM prompt functions to test:

# In[22]:


def create_analysis_prompt_basic():
    """Create the structured prompt for image analysis."""
    return """
    Analyze this image and return ONLY a Python dictionary in exactly this format:
    
    {
        'image_is_photography': X,  # True if the image is a photography, False otherwise
        'additional_comments': '' # Any additional observations or empty string if none
    }
    
    Replace X with True (image is a photography) or False (image is not a photography).
    Return ONLY the dictionary, no other text.
    Your answer MUST have the exact structue of the dictionary described above (all keys MUST be present). 
    If you cannot answer the question in the way implied by this structure, enter 'None' as value and offer 
    your answer and explanations under 'additional_comments'.
    """


# In[23]:


def create_analysis_prompt_precise():
    """Create the structured prompt for image analysis."""
    return """
    Analyze this image and return ONLY a Python dictionary in exactly this format:
    
    {
        'image_of_image': X,  # True if the image is a direct representation of something, False if it is an indirect representation (i.e. a representation of a representation of something).
        'additional_comments': '' # Any additional observations or empty string if none
    }
    
    Replace X with True (direct representation) or False (indirect representation).
    A representation can represent anything that exists or aspects thereof; the represantation can be concrete or abstract. 
    The image at hand can either be such a direct representation 
    or else a representation of a representation of anything that exists or of aspects thereof (concrete or abstract). 
    In other words, you have to determine if the image is a direct (Replace X with True) or an indirect representation (Replace X with False).
    Return ONLY the dictionary, no other text.
    Your answer MUST have the exact structue of the dictionary described above (all keys MUST be present). 
    If you cannot answer the question in the way implied by this structure, enter 'None' as value and offer 
    your answer and explanations under 'additional_comments'.
    """


# In[24]:


def create_analysis_prompt_intuitive():
    """Create the structured prompt for image analysis."""
    return """
    Analyze this image and return ONLY a Python dictionary in exactly this format:
    
    {
        'image_is_photography': X,  # True if the image is photography, False otherwise (e.g. if the image is a map or a painting)
        'additional_comments': '' # Any additional observations or empty string if none
    }
    
    Replace X with True (image is a photography) or False (otherwise).
    Return ONLY the dictionary, no other text.
    Your answer MUST have the exact structue of the dictionary described above (all keys MUST be present). 
    If you cannot answer the question in the way implied by this structure, enter 'None' as value and offer 
    your answer and explanations under 'additional_comments'.
    """


# In[25]:


def create_analysis_prompt_alternatives():
    """Create the structured prompt for image analysis."""
    return """
    Analyze this image and return ONLY a Python dictionary in exactly this format:
    
    {
        'image_is_photography': X,  # True if the image is photography, False if image is a map, a painting, a drawing, a scheme, a statistics figure, or other.
        'additional_comments': '' # Any additional observations or empty string if none.
    }
    
    Replace X with True (image is a photography) or False (image is a map, a painting, a drawing, a scheme, a statistics figure, or other).
    Return ONLY the dictionary, no other text.
    Your answer MUST have the exact structue of the dictionary described above (all keys MUST be present). 
    If you cannot answer the question in the way implied by this structure, enter 'None' as value and offer 
    your answer and explanations under 'additional_comments'.
    """


# In[ ]:





# ## Identify non-photo images with basic prompt:

# ### Set parameters:

# In[26]:


# Set parameters: 
analysis_name = 'is_photo_basic_struct_minicpm'
create_analysis_prompt = create_analysis_prompt_basic
keys_list_expected = ['image_is_photography', 'additional_comments']
response_variable = 'image_is_photography'

label_name = 'is_photo'
prediction_name = 'is_photo_pred'


# ### Prepare data objects: 

# In[27]:


# Prepare data objects: 
labels_results_repetitions = []
response_dictionaries[analysis_name] = {}

ml_metrics_analysis_name = []
ml_metrics_time_stamp = []
ml_metrics_positives = []
ml_metrics_negatives = []
ml_metrics_true_positives = []
ml_metrics_false_positives = []
ml_metrics_true_negatives = []
ml_metrics_false_negatives = []
ml_metrics_sensitivity = []
ml_metrics_specificity = []


# ### Extract and store results:

# In[ ]:


itercount = 0

while itercount < 5:

    # Analysis with LLM: 
    timestamp_start_is_photo_analysis, timestamp_end_is_photo_analysis, image_descr = analyse_giub_img_dir_llm(jpg_data_path, create_analysis_prompt, model_function)

    # Calculate duration of analysis: 
    duration = timestamp_end_is_photo_analysis - timestamp_start_is_photo_analysis
    total_seconds = duration.total_seconds()
    print(total_seconds)

    # Store information about duration: 
    time_analyses, time_analyses_for_df = store_duration(time_analyses, time_analyses_for_df, analysis_name, 
                   duration,timestamp_start_is_photo_analysis,
                  timestamp_end_is_photo_analysis)
    
    # Store dictionary with LLM responses:
    response_dictionaries[analysis_name][timestamp_start_is_photo_analysis] = image_descr
    
    # Extract LLM responses from dictionary:
    img_ids, is_photo, img_ids_closer_inspection = \
    llm_o.extract_vals_from_response_dict(img_ids, image_descr, keys_list_expected, response_variable)
    
    # Check if the response variable lists has the same length as id list:
    # print('Length of img_ids:')
    # print(len(img_ids))
    # print('Length of is_photo:')
    # print(len(is_photo))
    
    # Put response variables into data frame: 
    timestamp_ids = [timestamp_start_is_photo_analysis] * len(img_ids)
    predictions = pd.DataFrame({'image_id': img_ids, 
                               prediction_name: is_photo,
                               'time_stamp': timestamp_ids})
    
    # Check for missing values:
    # print(predictions.isnull().any().any())
    # print(predictions.isna().any().any())
    # print(has_missing_comprehensive(predictions))
    
    # Merge label data with the predictions:
    label_data_c = label_data.copy()
    labels_results = label_data_c.merge(predictions, how='inner', on='image_id')
    print(labels_results.shape)
    labels_results_repetitions.append(labels_results)
    
    # Save labels and predictions in dictionary: 
    #results_tabular[analysis_name] = {}
    #results_tabular[analysis_name][timestamp_start_is_photo_analysis] = labels_results
    
    # Save image list for closer inspection:
    timestamp_ids = [timestamp_start_is_photo_analysis] * len(img_ids_closer_inspection)
    imgs_closer_inspection = pd.DataFrame({'image_id': img_ids_closer_inspection,
    'time_stamp': timestamp_ids})
    images_closer_inspection[analysis_name] = imgs_closer_inspection
    
    
    # Calculate sensitivity and specificity for photography predictions and get lists images with positive photography predictions:
    subsets_and_metrics = llm_o.get_classification_subsets_metrics(labels_results, label_name, prediction_name)
    positives, negatives, true_positives, true_negatives, \
    false_negatives, false_positives, sensitivity, specificity = subsets_and_metrics
    # ml_metrics['analysis_name'] = analysis_name
    # ml_metrics['time_stamp'] = timestamp_start_is_photo_analysis
    # ml_metrics['positives'] = positives.shape[0]
    # ml_metrics['negatives'] = negatives.shape[0]
    # ml_metrics['true_positives'] = true_positives.shape[0]
    # ml_metrics['false_positives'] = false_positives.shape[0]
    # ml_metrics['true_negatives'] = true_negatives.shape[0]
    # ml_metrics['false_negatives'] = false_negatives.shape[0]
    # ml_metrics['sensitivity'] = sensitivity
    # ml_metrics['specificity'] = specificity


    ml_metrics_analysis_name.append(analysis_name)
    ml_metrics_time_stamp.append(timestamp_start_is_photo_analysis)
    ml_metrics_positives.append(positives.shape[0])
    ml_metrics_negatives.append(negatives.shape[0])
    ml_metrics_true_positives.append(true_positives.shape[0])
    ml_metrics_false_positives.append(false_positives.shape[0])
    ml_metrics_true_negatives.append(true_negatives.shape[0])
    ml_metrics_false_negatives.append(false_negatives.shape[0])
    ml_metrics_sensitivity.append(sensitivity)
    ml_metrics_specificity.append(specificity)
    
    
    
    print(f'True Positives: {true_positives.shape[0]}')
    print(f'False Positives: {false_positives.shape[0]}')
    print(f'True Negatives: {true_negatives.shape[0]}')
    print(f'False Negatives: {false_negatives.shape[0]}')
    
    itercount += 1

results_tabular[analysis_name] = pd.concat(labels_results_repetitions, ignore_index=True)


ml_metrics_one_analysis = pd.DataFrame({})

ml_metrics_one_analysis['analysis_name'] = ml_metrics_analysis_name
ml_metrics_one_analysis['time_stamp'] = ml_metrics_time_stamp
ml_metrics_one_analysis['positives'] = ml_metrics_positives
ml_metrics_one_analysis['negatives'] = ml_metrics_negatives
ml_metrics_one_analysis['true_positives'] = ml_metrics_true_positives
ml_metrics_one_analysis['false_positives'] = ml_metrics_false_positives
ml_metrics_one_analysis['true_negatives'] = ml_metrics_true_negatives
ml_metrics_one_analysis['false_negatives'] = ml_metrics_false_negatives
ml_metrics_one_analysis['sensitivity'] = ml_metrics_sensitivity
ml_metrics_one_analysis['specificity'] = ml_metrics_specificity

ml_metrics = pd.concat([ml_metrics, ml_metrics_one_analysis], ignore_index=True)



# In[ ]:


image_files[0].split('Oberland')[1].split('.')[0]


# In[ ]:


image_files


# In[ ]:


ml_metrics


# In[ ]:





# In[ ]:





# In[ ]:





# ## Identify non-photo images with intuitive prompt:

# ### Set parameters:

# In[ ]:


# Set parameters: 
analysis_name = 'is_photo_intuitive_struct_minicpm'
create_analysis_prompt = create_analysis_prompt_intuitive
keys_list_expected = ['image_is_photography', 'additional_comments']
response_variable = 'image_is_photography'

label_name = 'is_photo'
prediction_name = 'is_photo_pred'


# ### Prepare data objects: 

# In[ ]:


# Prepare data objects: 
labels_results_repetitions = []
response_dictionaries[analysis_name] = {}

ml_metrics_analysis_name = []
ml_metrics_time_stamp = []
ml_metrics_positives = []
ml_metrics_negatives = []
ml_metrics_true_positives = []
ml_metrics_false_positives = []
ml_metrics_true_negatives = []
ml_metrics_false_negatives = []
ml_metrics_sensitivity = []
ml_metrics_specificity = []


# ### Extract and store results:

# In[ ]:


itercount = 0

while itercount < 5:

    # Analysis with LLM: 
    timestamp_start_is_photo_analysis, timestamp_end_is_photo_analysis, image_descr = analyse_giub_img_dir_llm(jpg_data_path, create_analysis_prompt, model_function)

    # Calculate duration of analysis: 
    duration = timestamp_end_is_photo_analysis - timestamp_start_is_photo_analysis
    total_seconds = duration.total_seconds()
    print(total_seconds)

    # Store information about duration: 
    time_analyses, time_analyses_for_df = store_duration(time_analyses, time_analyses_for_df, analysis_name, 
                   duration,timestamp_start_is_photo_analysis,
                  timestamp_end_is_photo_analysis)
    
    # Store dictionary with LLM responses:
    response_dictionaries[analysis_name][timestamp_start_is_photo_analysis] = image_descr
    
    # Extract LLM responses from dictionary:
    img_ids, is_photo, img_ids_closer_inspection = \
    llm_o.extract_vals_from_response_dict(img_ids, image_descr, keys_list_expected, response_variable)
    
    # Check if the response variable lists has the same length as id list:
    # print('Length of img_ids:')
    # print(len(img_ids))
    # print('Length of is_photo:')
    # print(len(is_photo))
    
    # Put response variables into data frame: 
    timestamp_ids = [timestamp_start_is_photo_analysis] * len(img_ids)
    predictions = pd.DataFrame({'image_id': img_ids, 
                               prediction_name: is_photo,
                               'time_stamp': timestamp_ids})
    
    # Check for missing values:
    # print(predictions.isnull().any().any())
    # print(predictions.isna().any().any())
    # print(has_missing_comprehensive(predictions))
    
    # Merge label data with the predictions:
    label_data_c = label_data.copy()
    labels_results = label_data_c.merge(predictions, how='inner', on='image_id')
    print(labels_results.shape)
    labels_results_repetitions.append(labels_results)
    
    # Save labels and predictions in dictionary: 
    #results_tabular[analysis_name] = {}
    #results_tabular[analysis_name][timestamp_start_is_photo_analysis] = labels_results
    
    # Save image list for closer inspection:
    timestamp_ids = [timestamp_start_is_photo_analysis] * len(img_ids_closer_inspection)
    imgs_closer_inspection = pd.DataFrame({'image_id': img_ids_closer_inspection,
    'time_stamp': timestamp_ids})
    images_closer_inspection[analysis_name] = imgs_closer_inspection
    
    
    # Calculate sensitivity and specificity for photography predictions and get lists images with positive photography predictions:
    subsets_and_metrics = llm_o.get_classification_subsets_metrics(labels_results, label_name, prediction_name)
    positives, negatives, true_positives, true_negatives, \
    false_negatives, false_positives, sensitivity, specificity = subsets_and_metrics
    # ml_metrics['analysis_name'] = analysis_name
    # ml_metrics['time_stamp'] = timestamp_start_is_photo_analysis
    # ml_metrics['positives'] = positives.shape[0]
    # ml_metrics['negatives'] = negatives.shape[0]
    # ml_metrics['true_positives'] = true_positives.shape[0]
    # ml_metrics['false_positives'] = false_positives.shape[0]
    # ml_metrics['true_negatives'] = true_negatives.shape[0]
    # ml_metrics['false_negatives'] = false_negatives.shape[0]
    # ml_metrics['sensitivity'] = sensitivity
    # ml_metrics['specificity'] = specificity


    ml_metrics_analysis_name.append(analysis_name)
    ml_metrics_time_stamp.append(timestamp_start_is_photo_analysis)
    ml_metrics_positives.append(positives.shape[0])
    ml_metrics_negatives.append(negatives.shape[0])
    ml_metrics_true_positives.append(true_positives.shape[0])
    ml_metrics_false_positives.append(false_positives.shape[0])
    ml_metrics_true_negatives.append(true_negatives.shape[0])
    ml_metrics_false_negatives.append(false_negatives.shape[0])
    ml_metrics_sensitivity.append(sensitivity)
    ml_metrics_specificity.append(specificity)
    
    
    
    print(f'True Positives: {true_positives.shape[0]}')
    print(f'False Positives: {false_positives.shape[0]}')
    print(f'True Negatives: {true_negatives.shape[0]}')
    print(f'False Negatives: {false_negatives.shape[0]}')
    
    itercount += 1

results_tabular[analysis_name] = pd.concat(labels_results_repetitions, ignore_index=True)


ml_metrics_one_analysis = pd.DataFrame({'positives': [],
              'negatives': [], 
              'true_positives': [], 
              'true_negatives': [],
              'false_negatives': [], 
              'false_positives': [], 
              'sensitivity': [], 
              'specificity': []})

ml_metrics_one_analysis['analysis_name'] = ml_metrics_analysis_name
ml_metrics_one_analysis['time_stamp'] = ml_metrics_time_stamp
ml_metrics_one_analysis['positives'] = ml_metrics_positives
ml_metrics_one_analysis['negatives'] = ml_metrics_negatives
ml_metrics_one_analysis['true_positives'] = ml_metrics_true_positives
ml_metrics_one_analysis['false_positives'] = ml_metrics_false_positives
ml_metrics_one_analysis['true_negatives'] = ml_metrics_true_negatives
ml_metrics_one_analysis['false_negatives'] = ml_metrics_false_negatives
ml_metrics_one_analysis['sensitivity'] = ml_metrics_sensitivity
ml_metrics_one_analysis['specificity'] = ml_metrics_specificity

ml_metrics = pd.concat([ml_metrics, ml_metrics_one_analysis], ignore_index=True)



# In[ ]:





# In[ ]:


results_tabular[analysis_name]


# In[ ]:


ml_metrics


# In[ ]:


response_dictionaries


# In[ ]:





# In[ ]:





# ## Identify non-photo images with alternatives prompt:

# ### Set parameters:

# In[ ]:


# Set parameters: 
analysis_name = 'is_photo_alternatives_struct_minicpm'
create_analysis_prompt = create_analysis_prompt_alternatives
keys_list_expected = ['image_is_photography', 'additional_comments']
response_variable = 'image_is_photography'

label_name = 'is_photo'
prediction_name = 'is_photo_pred'


# ### Prepare data objects: 

# In[ ]:


# Prepare data objects: 
labels_results_repetitions = []
response_dictionaries[analysis_name] = {}

ml_metrics_analysis_name = []
ml_metrics_time_stamp = []
ml_metrics_positives = []
ml_metrics_negatives = []
ml_metrics_true_positives = []
ml_metrics_false_positives = []
ml_metrics_true_negatives = []
ml_metrics_false_negatives = []
ml_metrics_sensitivity = []
ml_metrics_specificity = []


# ### Extract and store results:

# In[ ]:


itercount = 0

while itercount < 5:

    # Analysis with LLM: 
    timestamp_start_is_photo_analysis, timestamp_end_is_photo_analysis, image_descr = analyse_giub_img_dir_llm(jpg_data_path, create_analysis_prompt, model_function)

    # Calculate duration of analysis: 
    duration = timestamp_end_is_photo_analysis - timestamp_start_is_photo_analysis
    total_seconds = duration.total_seconds()
    print(total_seconds)

    # Store information about duration: 
    time_analyses, time_analyses_for_df = store_duration(time_analyses, time_analyses_for_df, analysis_name, 
                   duration,timestamp_start_is_photo_analysis,
                  timestamp_end_is_photo_analysis)
    
    # Store dictionary with LLM responses:
    response_dictionaries[analysis_name][timestamp_start_is_photo_analysis] = image_descr
    
    # Extract LLM responses from dictionary:
    img_ids, is_photo, img_ids_closer_inspection = \
    llm_o.extract_vals_from_response_dict(img_ids, image_descr, keys_list_expected, response_variable)
    
    # Check if the response variable lists has the same length as id list:
    # print('Length of img_ids:')
    # print(len(img_ids))
    # print('Length of is_photo:')
    # print(len(is_photo))
    
    # Put response variables into data frame: 
    timestamp_ids = [timestamp_start_is_photo_analysis] * len(img_ids)
    predictions = pd.DataFrame({'image_id': img_ids, 
                               prediction_name: is_photo,
                               'time_stamp': timestamp_ids})
    
    # Check for missing values:
    # print(predictions.isnull().any().any())
    # print(predictions.isna().any().any())
    # print(has_missing_comprehensive(predictions))
    
    # Merge label data with the predictions:
    label_data_c = label_data.copy()
    labels_results = label_data_c.merge(predictions, how='inner', on='image_id')
    print(labels_results.shape)
    labels_results_repetitions.append(labels_results)
    
    # Save labels and predictions in dictionary: 
    #results_tabular[analysis_name] = {}
    #results_tabular[analysis_name][timestamp_start_is_photo_analysis] = labels_results
    
    # Save image list for closer inspection:
    timestamp_ids = [timestamp_start_is_photo_analysis] * len(img_ids_closer_inspection)
    imgs_closer_inspection = pd.DataFrame({'image_id': img_ids_closer_inspection,
    'time_stamp': timestamp_ids})
    images_closer_inspection[analysis_name] = imgs_closer_inspection
    
    
    # Calculate sensitivity and specificity for photography predictions and get lists images with positive photography predictions:
    subsets_and_metrics = llm_o.get_classification_subsets_metrics(labels_results, label_name, prediction_name)
    positives, negatives, true_positives, true_negatives, \
    false_negatives, false_positives, sensitivity, specificity = subsets_and_metrics
    # ml_metrics['analysis_name'] = analysis_name
    # ml_metrics['time_stamp'] = timestamp_start_is_photo_analysis
    # ml_metrics['positives'] = positives.shape[0]
    # ml_metrics['negatives'] = negatives.shape[0]
    # ml_metrics['true_positives'] = true_positives.shape[0]
    # ml_metrics['false_positives'] = false_positives.shape[0]
    # ml_metrics['true_negatives'] = true_negatives.shape[0]
    # ml_metrics['false_negatives'] = false_negatives.shape[0]
    # ml_metrics['sensitivity'] = sensitivity
    # ml_metrics['specificity'] = specificity


    ml_metrics_analysis_name.append(analysis_name)
    ml_metrics_time_stamp.append(timestamp_start_is_photo_analysis)
    ml_metrics_positives.append(positives.shape[0])
    ml_metrics_negatives.append(negatives.shape[0])
    ml_metrics_true_positives.append(true_positives.shape[0])
    ml_metrics_false_positives.append(false_positives.shape[0])
    ml_metrics_true_negatives.append(true_negatives.shape[0])
    ml_metrics_false_negatives.append(false_negatives.shape[0])
    ml_metrics_sensitivity.append(sensitivity)
    ml_metrics_specificity.append(specificity)
    
    
    
    print(f'True Positives: {true_positives.shape[0]}')
    print(f'False Positives: {false_positives.shape[0]}')
    print(f'True Negatives: {true_negatives.shape[0]}')
    print(f'False Negatives: {false_negatives.shape[0]}')
    
    itercount += 1

results_tabular[analysis_name] = pd.concat(labels_results_repetitions, ignore_index=True)


ml_metrics_one_analysis = pd.DataFrame({'positives': [],
              'negatives': [], 
              'true_positives': [], 
              'true_negatives': [],
              'false_negatives': [], 
              'false_positives': [], 
              'sensitivity': [], 
              'specificity': []})

ml_metrics_one_analysis['analysis_name'] = ml_metrics_analysis_name
ml_metrics_one_analysis['time_stamp'] = ml_metrics_time_stamp
ml_metrics_one_analysis['positives'] = ml_metrics_positives
ml_metrics_one_analysis['negatives'] = ml_metrics_negatives
ml_metrics_one_analysis['true_positives'] = ml_metrics_true_positives
ml_metrics_one_analysis['false_positives'] = ml_metrics_false_positives
ml_metrics_one_analysis['true_negatives'] = ml_metrics_true_negatives
ml_metrics_one_analysis['false_negatives'] = ml_metrics_false_negatives
ml_metrics_one_analysis['sensitivity'] = ml_metrics_sensitivity
ml_metrics_one_analysis['specificity'] = ml_metrics_specificity

ml_metrics = pd.concat([ml_metrics, ml_metrics_one_analysis], ignore_index=True)



# In[ ]:





# ## Identify non-photo images with precise prompt:

# ### Set parameters:

# In[ ]:


# Set parameters: 
analysis_name = 'is_photo_precise_struct_minicpm'
create_analysis_prompt = create_analysis_prompt_precise
keys_list_expected = ['image_of_image', 'additional_comments']
response_variable = 'image_of_image'

label_name = 'is_photo'
prediction_name = 'is_photo_pred'


# ### Prepare data objects: 

# In[ ]:


# Prepare data objects: 
labels_results_repetitions = []
response_dictionaries[analysis_name] = {}

ml_metrics_analysis_name = []
ml_metrics_time_stamp = []
ml_metrics_positives = []
ml_metrics_negatives = []
ml_metrics_true_positives = []
ml_metrics_false_positives = []
ml_metrics_true_negatives = []
ml_metrics_false_negatives = []
ml_metrics_sensitivity = []
ml_metrics_specificity = []


# ### Extract and store results:

# In[ ]:


itercount = 0

while itercount < 5:

    # Analysis with LLM: 
    timestamp_start_is_photo_analysis, timestamp_end_is_photo_analysis, image_descr = analyse_giub_img_dir_llm(jpg_data_path, create_analysis_prompt, model_function)

    # Calculate duration of analysis: 
    duration = timestamp_end_is_photo_analysis - timestamp_start_is_photo_analysis
    total_seconds = duration.total_seconds()
    print(total_seconds)

    # Store information about duration: 
    time_analyses, time_analyses_for_df = store_duration(time_analyses, time_analyses_for_df, analysis_name, 
                   duration,timestamp_start_is_photo_analysis,
                  timestamp_end_is_photo_analysis)
    
    # Store dictionary with LLM responses:
    response_dictionaries[analysis_name][timestamp_start_is_photo_analysis] = image_descr
    
    # Extract LLM responses from dictionary:
    img_ids, is_photo, img_ids_closer_inspection = \
    llm_o.extract_vals_from_response_dict(img_ids, image_descr, keys_list_expected, response_variable)
    
    # Check if the response variable lists has the same length as id list:
    # print('Length of img_ids:')
    # print(len(img_ids))
    # print('Length of is_photo:')
    # print(len(is_photo))
    
    # Put response variables into data frame: 
    timestamp_ids = [timestamp_start_is_photo_analysis] * len(img_ids)
    predictions = pd.DataFrame({'image_id': img_ids, 
                               prediction_name: is_photo,
                               'time_stamp': timestamp_ids})
    
    # Check for missing values:
    # print(predictions.isnull().any().any())
    # print(predictions.isna().any().any())
    # print(has_missing_comprehensive(predictions))
    
    # Merge label data with the predictions:
    label_data_c = label_data.copy()
    labels_results = label_data_c.merge(predictions, how='inner', on='image_id')
    print(labels_results.shape)
    labels_results_repetitions.append(labels_results)
    
    # Save labels and predictions in dictionary: 
    #results_tabular[analysis_name] = {}
    #results_tabular[analysis_name][timestamp_start_is_photo_analysis] = labels_results
    
    # Save image list for closer inspection:
    timestamp_ids = [timestamp_start_is_photo_analysis] * len(img_ids_closer_inspection)
    imgs_closer_inspection = pd.DataFrame({'image_id': img_ids_closer_inspection,
    'time_stamp': timestamp_ids})
    images_closer_inspection[analysis_name] = imgs_closer_inspection
    
    
    # Calculate sensitivity and specificity for photography predictions and get lists images with positive photography predictions:
    subsets_and_metrics = llm_o.get_classification_subsets_metrics(labels_results, label_name, prediction_name)
    positives, negatives, true_positives, true_negatives, \
    false_negatives, false_positives, sensitivity, specificity = subsets_and_metrics
    # ml_metrics['analysis_name'] = analysis_name
    # ml_metrics['time_stamp'] = timestamp_start_is_photo_analysis
    # ml_metrics['positives'] = positives.shape[0]
    # ml_metrics['negatives'] = negatives.shape[0]
    # ml_metrics['true_positives'] = true_positives.shape[0]
    # ml_metrics['false_positives'] = false_positives.shape[0]
    # ml_metrics['true_negatives'] = true_negatives.shape[0]
    # ml_metrics['false_negatives'] = false_negatives.shape[0]
    # ml_metrics['sensitivity'] = sensitivity
    # ml_metrics['specificity'] = specificity


    ml_metrics_analysis_name.append(analysis_name)
    ml_metrics_time_stamp.append(timestamp_start_is_photo_analysis)
    ml_metrics_positives.append(positives.shape[0])
    ml_metrics_negatives.append(negatives.shape[0])
    ml_metrics_true_positives.append(true_positives.shape[0])
    ml_metrics_false_positives.append(false_positives.shape[0])
    ml_metrics_true_negatives.append(true_negatives.shape[0])
    ml_metrics_false_negatives.append(false_negatives.shape[0])
    ml_metrics_sensitivity.append(sensitivity)
    ml_metrics_specificity.append(specificity)
    
    
    
    print(f'True Positives: {true_positives.shape[0]}')
    print(f'False Positives: {false_positives.shape[0]}')
    print(f'True Negatives: {true_negatives.shape[0]}')
    print(f'False Negatives: {false_negatives.shape[0]}')
    
    itercount += 1

results_tabular[analysis_name] = pd.concat(labels_results_repetitions, ignore_index=True)


ml_metrics_one_analysis = pd.DataFrame({'positives': [],
              'negatives': [], 
              'true_positives': [], 
              'true_negatives': [],
              'false_negatives': [], 
              'false_positives': [], 
              'sensitivity': [], 
              'specificity': []})

ml_metrics_one_analysis['analysis_name'] = ml_metrics_analysis_name
ml_metrics_one_analysis['time_stamp'] = ml_metrics_time_stamp
ml_metrics_one_analysis['positives'] = ml_metrics_positives
ml_metrics_one_analysis['negatives'] = ml_metrics_negatives
ml_metrics_one_analysis['true_positives'] = ml_metrics_true_positives
ml_metrics_one_analysis['false_positives'] = ml_metrics_false_positives
ml_metrics_one_analysis['true_negatives'] = ml_metrics_true_negatives
ml_metrics_one_analysis['false_negatives'] = ml_metrics_false_negatives
ml_metrics_one_analysis['sensitivity'] = ml_metrics_sensitivity
ml_metrics_one_analysis['specificity'] = ml_metrics_specificity

ml_metrics = pd.concat([ml_metrics, ml_metrics_one_analysis], ignore_index=True)



# In[ ]:





# In[ ]:


timestamp_id = timestamp_start_is_photo_analysis.strftime('%Y%m%d_%H%M%S')


# ## Save ml-metrics data frame: 

# In[ ]:


# Define file name: 
#date = str(timestamp_end_is_photo_analysis).split('.')[0][0:10]
filename = 'ml_metrics_prompt_exp_struct_minicpm_' + timestamp_id + '.csv'
ml_metrics_output_path = os.path.join(data_path, filename)

# Save csv-file: 
ml_metrics.to_csv(ml_metrics_output_path, index=False)

# Reload saved csv table to check if saving worked:
ml_metrics_reloaded = pd.read_csv(ml_metrics_output_path)
ml_metrics_reloaded.head()


# ## Save response dictionary:

# In[ ]:


# Define file name: 
#date = str(timestamp_end_is_photo_analysis).split('.')[0][0:10]
filename = 'responses_prompt_exp_struct_minicpm_' + timestamp_id + '.pkl'

# Save dictionary with LLM responses:
img_analysis_output_path = os.path.join(data_path, filename)
with open(img_analysis_output_path, 'wb') as f:
   pickle.dump(response_dictionaries, f)

# Reload saved dictionary to check if saving worked:
with open(img_analysis_output_path, 'rb') as f:
   reloaded_image_descr = pickle.load(f)

# Check if original and reloaded dictionary are the same:
print(len(image_descr))
print(type(image_descr))
print(type(reloaded_image_descr))
print(len(reloaded_image_descr))

print(image_descr.keys() == reloaded_image_descr.keys())


# ## Save labels and results:

# In[ ]:


# Define file name: 
#current_timestamp = pd.Timestamp.now()
#current_date_time = current_timestamp.strftime('%Y-%m-%d %H:%M')
results_file_name = 'results_prompt_exp_struct_minicpm_' + timestamp_id + '.pkl'

# Save dictionary with LLM responses:
results_tabular_output_path = os.path.join(data_path, results_file_name)
with open(results_tabular_output_path, 'wb') as f:
   pickle.dump(results_tabular, f)

# Reload saved dictionary to check if saving worked:
with open(results_tabular_output_path, 'rb') as f:
   reloaded_results_tabular = pickle.load(f)

# Check if original and reloaded dictionary are the same:
print(len(results_tabular))
print(type(results_tabular))
print(type(reloaded_results_tabular))
print(len(reloaded_results_tabular))

print(results_tabular.keys() == reloaded_results_tabular.keys())


# ## Calculate duration of analysis overall:

# In[ ]:


timestamp_end_workflow = pd.Timestamp.now()
timestamp_end_workflow


# ## Save time analyses: 

# In[ ]:


# Define file name: 
# current_date_time = current_timestamp.strftime('%Y-%m-%d %H:%M')

time_analyses_df_file_name = 'times_df_prompt_exp_struct_minicpm_' + timestamp_id + '.pkl'

# Save dictionary:
time_analyses_df_output_path = os.path.join(data_path, time_analyses_df_file_name)
with open(time_analyses_df_output_path, 'wb') as f:
   pickle.dump(time_analyses_for_df, f)

# Reload saved dictionary to check if saving worked:
with open(time_analyses_df_output_path, 'rb') as f:
   reloaded_time_analyses_for_df = pickle.load(f)

# Check if original and reloaded dictionary are the same:
print(len(time_analyses_for_df))
print(type(time_analyses_for_df))
print(type(reloaded_time_analyses_for_df))
print(len(reloaded_time_analyses_for_df))

print(time_analyses_for_df.keys() == reloaded_time_analyses_for_df.keys())


# In[ ]:


ml_metrics


# In[ ]:





# In[ ]:


time_analyses


# In[ ]:





# In[ ]:


ml_metrics


# In[ ]:





# In[ ]:





# In[ ]:




