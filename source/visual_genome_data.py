import os
import requests
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
from .visual_genome_meta_data import count_occurrences

def convert_url(original_url):
    """
    Converts URLs from VG_100K_2 format to VG_100K format.
    
    Example:
    https://cs.stanford.edu/people/rak248/VG_100K_2/1.jpg -> https://cs.stanford.edu/people/rak248/VG_100K/1.jpg
    """
    url_part = original_url.split('/')[-2].split('_')[-1]
    if url_part == '2': 
        # Replace VG_100K_2 with VG_100K in the URL
        new_url = original_url.replace("VG_100K_2", "VG_100K")
    else:
        # Replace VG_100K with VG_100K_2 in the URL
        new_url = original_url.replace("VG_100K", "VG_100K_2")
    return new_url

def download_image(original_url, directory_path, change_url=False):
    """
    Downloads an image from a URL and saves it to the specified file path.
    Assumes the directory already exists.
    
    Parameters:
    image_url (str): URL of the image to download
    file_path (str): Full path where the image will be saved
    
    Returns:
    bool: True if successful, False otherwise
    """
    # Modify url if necessary (it appears that for the moment only one image is present at the new address):
    if change_url:
        image_url = convert_url(original_url)
    else:
        image_url = original_url
    # Extract filename from URL
    filename_raw = os.path.basename(image_url)  # This will be "1.jpg"
    filename = 'visual_genome_' + filename_raw
    # Combine directory path with filename
    directory = directory_path
    file_path = os.path.join(directory, filename)
    try:
        # Download the image
        # requests.get() sends an HTTP GET request to the specified URL
        # This method retrieves the content from the URL
        # The response object contains the status code, headers, and content
        response = requests.get(image_url)
        
        # Check if the request was successful
        # HTTP status code 200 means "OK" (request succeeded)
        # Other common codes: 404 (Not Found), 403 (Forbidden), 500 (Server Error)
        if response.status_code == 200:
            # Write the image to the specified file path
            # Open a file in binary write mode ('wb')
            # This creates a new file or overwrites an existing one
            # The with statement ensures that the file is properly 
            # closed after writing, even if an error occurs during the write 
            # operation.
        
            with open(file_path, 'wb') as file:
                file.write(response.content)
            return True
        else:
            #print(f"Failed to download image. Status code: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False

def get_image_ids(dir_path):
    id_list = []
    os.chdir(dir_path)
    all_files = os.listdir()
    image_file_list = [f for f in all_files if f.lower().endswith('.jpg')]
    #print(image_file_list)
    for filename in image_file_list:
        last_part = filename.split('_')[-1]
        img_id_str = last_part.split('.')[0]
        img_id = int(img_id_str)
        #print(img_id)
        id_list.append(img_id)
    return id_list

def get_file_by_id(data_path, identifier, file_extension):
    filenames = []
    identifier_underlines = '_' + str(identifier) + '_'
    identifier_end = '_' + str(identifier) + '.'
    for file in os.listdir(str(data_path)):
        id_underline_bool = identifier_underlines in file
        id_end_bool = identifier_end in file
        id_bool = id_underline_bool or id_end_bool
        file_ext_bool = file.endswith(file_extension)
        if id_bool and file_ext_bool:
            filenames.append(file)
            
    return filenames