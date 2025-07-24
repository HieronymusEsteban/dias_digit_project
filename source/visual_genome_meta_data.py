import json
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def read_json_to_dict(file_path):
    try:
        # Open the file and load the JSON content
        with open(file_path, 'r') as file:
            data_dict = json.load(file)
        
        print(f"Successfully loaded JSON from {file_path}")
        return data_dict
    
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: {file_path} is not a valid JSON file.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        return None

def count_occurrences(occurrence_counts, search_list):
    """
    Count occurrences of keys from occurrence_counts in search_list.
    Updates the occurrence_counts dictionary in place.
    
    Args:
        occurrence_counts: Dictionary with strings as keys and current counts as values
        search_list: List of strings to search through
    
    Returns:
        The updated occurrence_counts dictionary (same object, modified in place)
    """
    # Count occurrences of each string and add to existing counts
    for item in search_list:
        if item in occurrence_counts:
            occurrence_counts[item] += 1
    
    return occurrence_counts

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

def get_image_meta_data(objects, desired_image_id):
    '''Get image meta data from objects (list of dictionaries) based on image id.'''
    for image_metadata in objects:
        image_id = image_metadata['image_id']
        if image_id == desired_image_id:
            #print(image_id)
            #print(image_metadata['image_url'])
            desired_image_metadata = image_metadata
    return desired_image_metadata

def bboxes_from_metadata(image_path, objects, desired_object):
    image_path = str(image_path)
    
    # Load image, get image dimensions:
    image = Image.open(image_path)
    img_width, img_height = image.size
    
    # Get image id: 
    image_path_last_part = image_path.split('_')[-1]
    image_id = int(image_path_last_part.split('.')[0])
    print(image_id)
    
    # Get image meta-data based on id: 
    img_data = get_image_meta_data(objects, image_id)
    print(img_data['image_id'])

    object_name_list = []
    for idx in list(range(0, len(img_data['objects']))):
        #print(idx)
        names = img_data['objects'][idx]['names']
        #print(names)
        object_name_list.extend(names)
    print(object_name_list)

    list_of_boxes = []

    for obj in img_data['objects']:
        if obj['names'][0] == desired_object:
            print(obj['names'][0])
            x, y = obj['x'], obj['y']
            w, h = obj['w'], obj['h']
            # Convert to YOLO format (normalized center coordinates)
            x_center = (x + w/2) / img_width
            y_center = (y + h/2) / img_height
            width = w / img_width
            height = h / img_height
        
                
            # Ensure coordinates are within [0, 1]
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            width = max(0, min(1, width))
            height = max(0, min(1, height))
                
            box_data = (x_center, y_center, width, height )
            list_of_boxes.append(box_data)
                
    return list_of_boxes

def plot_image_with_multiple_bboxes(image_path, bboxes, class_names=None):
    """
    Plot an image with multiple bounding boxes.
    
    Args:
        image_path (str): Path to the image file
        bboxes (list): List of bounding box coordinates, each as (x_center, y_center, width, height)
                       with values normalized between 0 and 1
        class_names (list, optional): List of class names for each box
    """
    # Load the image
    image = Image.open(image_path)
    img_width, img_height = image.size
    
    # Convert to numpy array for matplotlib
    image_np = np.array(image)
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Display the image
    ax.imshow(image_np)
    
    # Process each bounding box
    for i, bbox in enumerate(bboxes):
        # Unpack normalized bbox coordinates
        x_center, y_center, width, height = bbox
        
        # Convert normalized coordinates to pixel values
        x_center_px = x_center * img_width
        y_center_px = y_center * img_height
        width_px = width * img_width
        height_px = height * img_height
        
        # Calculate top-left corner of rectangle
        x_min = x_center_px - (width_px / 2)
        y_min = y_center_px - (height_px / 2)
        
        # Create rectangle patch
        rect = patches.Rectangle(
            (x_min, y_min), width_px, height_px,
            linewidth=2,
            edgecolor='red',
            facecolor='none'
        )
        
        # Add rectangle to the image
        ax.add_patch(rect)
        
        # Add class label if provided
        if class_names and i < len(class_names):
            plt.text(
                x_min, y_min - 5,
                class_names[i],
                color='red',
                fontsize=12,
                bbox=dict(facecolor='white', alpha=0.7)
            )
    
    # Hide axes and show plot
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return fig, ax