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
import matplotlib.pyplot as plt
from source import llm_input as llm_i
from source import llm_output as llm_o
import os
from PIL import Image


# In[2]:


import ollama
import json
import re
import pickle


# In[3]:


os.getcwd()


# # Using LLM (mini-CPM) for image analysis

# ## Define Functions:

# In[4]:


def add_pred_values(idx, labels_results, columns, values_to_add):
    selection_bools = labels_results.image_id == idx
    
    labels_results.loc[selection_bools, columns] = values_to_add


# In[ ]:





# In[5]:


def create_prompt_img_type_multi_object():
    """Create the structured prompt for image analysis."""
    return """
    Analyze this image and return ONLY a Python dictionary in exactly this format:
    
    {
        'image_is_photograph': X,     # True if the image is a photograph, False otherwise (if the image is a drawing, painting, statistics figure, map, scheme, other)
        'high_alpine_environment': X, # True if this appears to be in an high Alpine environment, False if not
        'person': X,                  # True if present, False if not
        'glacier': X,                 # True if present, False if not
        'church': X,                  # True if present, False if not
        'water_body': X.              # True if present, False if not
        'other_objects': [],          # List of other noteworthy/dominant objects
        'additional_comments': ''     # Any additional observations or empty string if none
    }
    
    Replace X with True (present) or False (not present).
    Return ONLY the dictionary, no other text.
    """


# In[ ]:





# In[6]:


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
    


# In[7]:


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





# ### Prepare empty dictionary for time analyses and get time stamp for overall workflow duration:

# In[8]:


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


# In[ ]:





# ### Define LLM model to be used:

# In[9]:


model_function = llm_i.call_minicpm_model


# ## Set paths:

# In[10]:


#root_path = Path('/Users/stephanehess/Documents/CAS_AML/dias_digit_project')
#root_path = Path('/Users/stephanehess/Documents/CAS_AML/dias_digit_project/test_yolo_object_train')

project_path = Path.cwd()
root_path = project_path
#root_path = (project_path / '..').resolve()
#root_path = (project_path / '..' / 'test_yolo_object_train').resolve()
#root_path = project_path / 'test_llm_img_analysis'
data_path = root_path / 'data'
tif_data_path = root_path / 'data_1'
#data_path = root_path / 'visual_genome_data_all'
jpg_data_path = root_path / 'data_jpg'
#yolo_path = root_path / 'visual_genome_yolo_all'
output_dir_not_photo = root_path / 'not_photo'
output_dir_with_person = root_path / 'with_person'
output_dir_without_person = root_path / 'without_person'



# In[ ]:





# ### Copy and convert image files from tif_data_path to jpg_data_path:

# In[11]:


source_folder = tif_data_path
destination_folder = jpg_data_path

llm_i.convert_tif_to_jpg(source_folder, destination_folder, quality=100)


# In[ ]:





# In[ ]:





# ## Create directories for sorting the images:

# In[12]:


# Create output directories
#os.chdir(root_path/'..')
os.makedirs(output_dir_not_photo, exist_ok=True)
os.makedirs(output_dir_with_person, exist_ok=True)
os.makedirs(output_dir_without_person, exist_ok=True)
#os.chdir('root_path')


# In[ ]:





# ## Loop through images and analyze with miniCPM (LLM model):

# In[ ]:





# ### Load label data (ground truth) to compare to LLM responses:

# The file with_without_person.csv contains labels added by (human) visual inspection that represent the ground truth. 
#  * Column with_person: whether or not any person is in the image.
#  * Column recognisable: whether any person that would be recognisable to a human familiar with said person is in the image.
#  * Column photo: whether or not the image is a photograph (as opposed to some other kind of representation such as map, drawing, painting, scheme, figure)
#  * Column church: whether or not any church is in the image.
#  * Column high_alpine_environment: whether or not the scene shown in the image is situated in a high alpine environment (according to non-expert human judgement)

# In[13]:


label_data = pd.read_csv(data_path/'labels_mod.csv')
label_data.head()


# In[14]:


img_ids = list(label_data.image_id)


# In[15]:


# Reconvert image ids to integers (e.g. '234') as strings from the form they were saved in (e.g. 'id234' to ensure 
# string data type to deal with duck typing): 
label_data['image_id'] = img_idc.reconvert_image_ids(img_ids)


# In[16]:


label_data.head()


# In[ ]:





# ### The following cell is only required for the test run on the test data: 

# In[17]:


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





# ## Repeat the same procedure but with nicer code (modularized):

# ### Set parameters:

# In[18]:


# Set parameters: 
prompt_func = create_prompt_img_type_multi_object
prompt_template = prompt_func.__name__
prompt_id = prompt_template + '_v1'
prompt_text = prompt_func()


keys_list_expected = ['image_is_photograph', 'high_alpine_environment', 'person', 'glacier',
                      'church', 'water_body', 'other_objects', 'additional_comments']


# ### Carry out LLM analysis of the images:

# In[19]:


# Carry out the LLM analysis:
timestamp_start_is_photo_analysis, timestamp_end_is_photo_analysis, image_descr = analyse_giub_img_dir_llm(jpg_data_path, prompt_func, model_function)


# ### Prepare data objects: 

# In[ ]:





# In[20]:


# Prepare data objects: 
response_dictionaries = {}
response_dictionaries[prompt_id] = {}

images_closer_inspection = {}

results_tabular = {}

ml_metrics = pd.DataFrame({})

# ml_metrics_analysis_name = []
# ml_metrics_prompt_id = []
# ml_metrics_label_name = []
# ml_metrics_time_stamp = []
# ml_metrics_positives = []
# ml_metrics_negatives = []
# ml_metrics_true_positives = []
# ml_metrics_false_positives = []
# ml_metrics_true_negatives = []
# ml_metrics_false_negatives = []
# ml_metrics_sensitivity = []
# ml_metrics_specificity = []


# In[21]:


# Calculate duration of analysis: 
duration = timestamp_end_is_photo_analysis - timestamp_start_is_photo_analysis
total_seconds = duration.total_seconds()
print(total_seconds)


# In[ ]:





# ### Extract and organize information from the dictionary containing the LLM responses:

# In[22]:


# Calculate duration of analysis: 
duration = timestamp_end_is_photo_analysis - timestamp_start_is_photo_analysis
total_seconds = duration.total_seconds()
print(total_seconds)

# Store information about duration of LLM task: 
time_analyses, time_analyses_for_df = store_duration(time_analyses, time_analyses_for_df, prompt_id, 
                duration,timestamp_start_is_photo_analysis,
                timestamp_end_is_photo_analysis)

# Get timestamp_id as string from the time stamp:
timestamp_id = timestamp_start_is_photo_analysis.strftime('%Y%m%d_%H%M%S')

# Store dictionary with LLM responses as raw data:
response_dictionaries[prompt_id][timestamp_id] = image_descr

# convert img_ids pandas series into list:
img_ids_l = list(img_ids)

# Prepare response variable names and label names to loop through:
#response_variables = ['image_is_photograph', 'person', 'church']
response_variables = ['image_is_photograph', 'high_alpine_environment', 'person', 'church']
#label_names = ['is_photo', 'with_person', 'with_church']
label_names = ['is_photo', 'in_high_alpine_environment', 'with_person', 'with_church']
#analysis_names = ['is_photo_struct_minicpm', 'with_person_struct_minicpm', 'with_church_struct_minicpm']

# Prepare dictionary for long term storing of results: 
results_tabular[timestamp_id] = {}
results_tabular[timestamp_id]['prompt_id'] = prompt_id
results_tabular[timestamp_id]['prompt_template'] = prompt_template
results_tabular[timestamp_id]['prompt_text'] = prompt_text
results_tabular[timestamp_id]['predictions'] = {}

# Get copy of label data to merge with prediction for short term presentation of results:
labels_results_i = label_data.copy()
print('labels_results initial:')
print(labels_results_i.shape)
print(labels_results_i.columns)

# Extract predictions for different response variables:
for response_variable, label_name in zip(response_variables, label_names):
    # set prediction name: 
    prediction_name = label_name + '_pred'
    analysis_name = label_name + '_struct_minicpm'
    print('\n')
    print('\n')
    print('response_variable name and prediction_name:')
    print(response_variable)
    print(prediction_name)
    img_ids, response_values, img_ids_closer_inspection = \
    llm_o.extract_vals_from_response_dict(img_ids_l, image_descr, keys_list_expected, response_variable)

    timestamp_ids = [timestamp_id] * len(img_ids)
    
    predictions = pd.DataFrame({'image_id': img_ids, 
                                   prediction_name: response_values})
    predictions[prediction_name] = predictions[prediction_name].astype('Int8')

    # print('\n')
    # print('predictions:')
    # print(predictions.shape)
    # print(predictions.columns)

    results_tabular[timestamp_id]['predictions'][response_variable] = predictions
    
    # Merge label data with the predictions:
    labels_results_i = labels_results_i.merge(predictions, how='inner', on='image_id')
    # print('\n')
    # print('merged labels_results:')
    # print(labels_results_i.shape)
    # print(labels_results_i.columns)

    # Save image list for closer inspection:
    timestamp_ids = [timestamp_id] * len(img_ids_closer_inspection)
    imgs_closer_inspection = pd.DataFrame({'image_id': img_ids_closer_inspection,
    'time_stamp': timestamp_ids})
    images_closer_inspection[analysis_name] = imgs_closer_inspection
    
    # Calculate sensitivity and specificity for photography predictions and get lists images with positive photography predictions:
    subsets_and_metrics = llm_o.get_classification_subsets_metrics(labels_results_i, label_name, prediction_name)
    positives, negatives, true_positives, true_negatives, \
    false_negatives, false_positives, sensitivity, specificity = subsets_and_metrics


    ml_metrics_analysis_name = []
    ml_metrics_prompt_id = []
    ml_metrics_label_name = []
    ml_metrics_time_stamp = []
    ml_metrics_positives = []
    ml_metrics_negatives = []
    ml_metrics_true_positives = []
    ml_metrics_false_positives = []
    ml_metrics_true_negatives = []
    ml_metrics_false_negatives = []
    ml_metrics_sensitivity = []
    ml_metrics_specificity = []

    ml_metrics_analysis_name.append(analysis_name)
    ml_metrics_prompt_id.append(prompt_id)
    ml_metrics_label_name.append(label_name)
    ml_metrics_time_stamp.append(timestamp_start_is_photo_analysis)
    ml_metrics_positives.append(positives.shape[0])
    ml_metrics_negatives.append(negatives.shape[0])
    ml_metrics_true_positives.append(true_positives.shape[0])
    ml_metrics_false_positives.append(false_positives.shape[0])
    ml_metrics_true_negatives.append(true_negatives.shape[0])
    ml_metrics_false_negatives.append(false_negatives.shape[0])
    ml_metrics_sensitivity.append(sensitivity)
    ml_metrics_specificity.append(specificity)

    ml_metrics_one_analysis = pd.DataFrame({})

    ml_metrics_one_analysis['analysis_name'] = ml_metrics_analysis_name
    ml_metrics_one_analysis['prompt_id'] = ml_metrics_prompt_id
    ml_metrics_one_analysis['label_name'] = ml_metrics_label_name
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

    # print('\n')
    # print('ml_metrics:')
    # print(ml_metrics.shape)
    # print(ml_metrics)


# In[ ]:





# ### Save ml_metrics:

# In[23]:


# Define file name: 
date = str(timestamp_end_is_photo_analysis).split('.')[0][0:10]
filename = 'ml_metrics_multi_object_struct_minicpm_' + timestamp_id + '.csv'
ml_metrics_output_path = os.path.join(data_path, filename)

# Save csv-file: 
ml_metrics.to_csv(ml_metrics_output_path, index=False)

# Reload saved csv table to check if saving worked:
ml_metrics_reloaded = pd.read_csv(ml_metrics_output_path)
ml_metrics_reloaded.head()


# In[ ]:





# ### Save images for closer inspection:

# In[30]:


# Define file name: 

filename = 'img_closer_insp_multi_object_struct_minicpm_' + timestamp_id + '.pkl'

# Save dictionary with LLM responses:
img_analysis_output_path = os.path.join(data_path, filename)
with open(img_analysis_output_path, 'wb') as f:
   pickle.dump(images_closer_inspection, f)

# Reload saved dictionary to check if saving worked:
with open(img_analysis_output_path, 'rb') as f:
   reloaded_images_closer_inspection = pickle.load(f)

# Check if original and reloaded dictionary are the same:
print(len(images_closer_inspection))
print(len(reloaded_images_closer_inspection))
print(type(images_closer_inspection))
print(type(reloaded_images_closer_inspection))

print(images_closer_inspection.keys() == reloaded_images_closer_inspection.keys())


# In[ ]:





# ## Save response dictionary:

# In[24]:


# Define file name: 

filename = 'responses_multi_object_struct_minicpm_' + timestamp_id + '.pkl'

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


# In[25]:


#reloaded_image_descr


# ## Save labels and results:

# In[26]:


# Define file name: 

results_file_name = 'results_multi_object_struct_minicpm_' + timestamp_id + '.pkl'

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


# In[27]:


reloaded_results_tabular.keys()


# In[ ]:





# ## Calculate duration of analysis overall:

# In[28]:


timestamp_end_workflow = pd.Timestamp.now()
timestamp_end_workflow


# ## Save time analyses: 

# In[29]:


# Define file name: 

time_analyses_df_file_name = 'times_multi_object_struct_minicpm_' + timestamp_id + '.pkl'

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


# In[31]:


time_analyses_for_df


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




