import yaml
import os
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from .visual_genome_meta_data import get_image_meta_data
from .visual_genome_data import count_occurrences



def create_class_mapping_from_list(desired_objects):
    """
    Create a class mapping from a list of desired object names.
    
    Args:
        desired_objects: List of strings with object names to detect
        
    Returns:
        dict: Mapping from class names to class IDs
        list: List of all class names (same as input, but useful for consistency)
    """
    # Create mapping from class names to IDs
    # Class IDs start from 0 and increment
    class_map = {name: idx for idx, name in enumerate(desired_objects)}
    
    return class_map

def save_class_map_to_yaml(class_map, output_path):
    """
    Save class mapping to a YAML file for YOLO.
    
    Args:
        class_map: Dictionary mapping class names to class IDs
        output_path: Path where to save the YAML file
    """
    # Create the data structure that YOLO expects
    yaml_data = {
        'names': list(class_map.keys()),
        'nc': len(class_map)  # number of classes
    }
    
    # Write to YAML file
    with open(output_path, 'w') as f:
        yaml.dump(yaml_data, f)
    
    print(f"Class mapping saved to {output_path}")

def read_yaml_to_class_map(yaml_path):
    """
    Read YAML file and create a class mapping dictionary.
    
    Args:
        yaml_path: Path to the YAML file
        
    Returns:
        dict: Mapping from class names to class IDs (e.g., 'tree': 0)
    """
    # Load the YAML file
    with open(yaml_path, 'r') as f:
        yaml_data = yaml.safe_load(f)
    
    # Extract class names
    class_names = yaml_data['names']
    
    # Create mapping from class names to IDs
    class_map = {name: idx for idx, name in enumerate(class_names)}
    
    return class_map

def convert_single_image_to_yolo(img_data, class_map, image_dir, output_dir):
    """
    Convert a single Visual Genome image dictionary to YOLO format.
    
    Args:
        img_data: Dictionary containing image metadata
        class_map: Dictionary mapping class names to class IDs
        image_dir: Directory containing the images
        output_dir: Directory to save YOLO format labels
        
    Returns:
        str: Path to the created label file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get image ID
    image_id = img_data['image_id']
    
    # Get image path and dimensions
    image_path = os.path.join(image_dir, f"visual_genome_{image_id}.jpg")
    
    # Get image dimensions
    try:
        with Image.open(image_path) as img:
            img_width, img_height = img.size
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        # If image can't be opened, skip this image
        return None
    
    # Create YOLO label file path
    label_path = os.path.join(output_dir, f"visual_genome_{image_id}.txt")
    
    # Convert coordinates and create YOLO label file
    with open(label_path, 'w') as f:
        for obj in img_data['objects']:
            # Get object class
            if isinstance(obj['names'], list) and len(obj['names']) > 0:
                class_name = obj['names'][0]
            elif isinstance(obj['names'], str):
                class_name = obj['names']
            else:
                continue  # Skip objects with no name
                
            # Skip if class not in mapping
            if class_name not in class_map:
                continue
                
            class_id = class_map[class_name]
            
            # Get bounding box coordinates
            x, y = obj['x'], obj['y']
            w, h = obj['w'], obj['h']
            
            # Skip invalid bounding boxes
            if w <= 0 or h <= 0:
                continue
                
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
            
            # Write to file
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
    
    return label_path


# def visualize_yolo_annotations(image_path, label_path, yaml_path):
#     """
#     Visualize an image with its YOLO format bounding boxes.
    
#     Args:
#         image_path: Path to the image file
#         label_path: Path to the YOLO format text file with bounding boxes
#         yaml_path: Path to the YAML file with class mappings
#     """
#     # Load the class mappings from YAML file
#     with open(yaml_path, 'r') as f:
#         yaml_data = yaml.safe_load(f)
    
#     class_names = yaml_data['names']
#     print(label_path)
#     # Load the image
#     print(image_path)
#     image = Image.open(image_path)
#     img_width, img_height = image.size
    
#     # Convert PIL Image to numpy array for matplotlib
#     image_np = np.array(image)
    
#     # Create a figure and axis
#     fig, ax = plt.subplots(1, figsize=(12, 9))
    
#     # Display the image
#     ax.imshow(image_np)
    
#     # Define colors for bounding boxes (one for each class)
#     # Using a colormap to generate distinct colors
#     cmap = plt.cm.get_cmap('hsv', len(class_names))
#     colors = [cmap(i) for i in range(len(class_names))]
    
#     # Load and draw bounding boxes from the label file
#     with open(label_path, 'r') as f:
#         for line in f:
#             data = line.strip().split()
#             print(data)
#             if len(data) != 5:  # YOLO format has 5 values per line
#                 continue
                
#             class_id = int(data[0])
#             x_center = float(data[1]) * img_width
#             y_center = float(data[2]) * img_height
#             width = float(data[3]) * img_width
#             height = float(data[4]) * img_height
            
#             # Calculate top-left corner for rectangle
#             x_min = x_center - (width / 2)
#             y_min = y_center - (height / 2)
            
#             # Create a rectangle patch
#             rect = patches.Rectangle(
#                 (x_min, y_min), width, height, 
#                 linewidth=2, 
#                 edgecolor=colors[class_id], 
#                 facecolor='none'
#             )
            
#             # Add the rectangle to the plot
#             ax.add_patch(rect)
            
#             # Add class label text above the bounding box
#             class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
#             plt.text(
#                 x_min, y_min - 5, 
#                 class_name,
#                 color=colors[class_id], 
#                 fontsize=12, 
#                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0)
#             )
    
#     plt.axis('off')  # Hide axes
#     plt.tight_layout()
    
#     return fig, ax

def read_yolo_metadata(file_path, class_map):
    """
    Read YOLO metadata from a text file and return class names and bounding boxes.
    
    Args:
        file_path (str): Path to the YOLO metadata text file
        class_map (dict): Dictionary mapping class names to class IDs
        
    Returns:
        tuple: (class_names, bounding_boxes)
            - class_names: List of class names in order
            - bounding_boxes: List of tuples (x_center, y_center, width, height)
    """
    class_names = []
    bounding_boxes = []
    
    # Create reverse mapping from class_id to class_name
    id_to_name = {}
    for name, class_id in class_map.items():
        id_to_name[class_id] = name
    
    with open(file_path, 'r') as f:
        for line in f:
            # Split the line into values
            values = line.strip().split()
            
            # Skip empty lines or invalid formats
            if len(values) != 5:
                continue
            
            # Extract class ID and bounding box values
            class_id = int(values[0])
            x_center = float(values[1])
            y_center = float(values[2])
            width = float(values[3])
            height = float(values[4])
            
            # Get class name from reversed mapping
            class_name = id_to_name.get(class_id, f"unknown_{class_id}")
            
            # Add to lists
            class_names.append(class_name)
            bounding_boxes.append((x_center, y_center, width, height))
    
    return class_names, bounding_boxes


def visual_genome_to_yolo_data_n(objects_and_ids, paths, class_map, with_class = True,
                              number_of_images = None):
    image_counter = 0
    
    data_path, yolo_path = paths

    objects, desired_objects, img_id_list = objects_and_ids
    
    desired_img_ids = []
    
    label_paths = []
    
    occurrence_counts = dict.fromkeys(desired_objects, 0)
    
    
    for idx in list(range(len(objects))):
    
        if objects[idx]['image_id'] == 1001:
            continue
    
        names = []
        for idx_obj in list(range(len(objects[idx]['objects']))):
            name = objects[idx]['objects'][idx_obj]['names']
            #print(name)
            names.extend(name)
        #print(names)  
        #print('trees' in names)
        
        if ('image_url' not in list(objects[idx].keys())):
            continue
    
        image_id = objects[idx]['image_id']
        #print('\n')
        #print(image_id)
        #print(type(image_id))
        #count_occurrences(occurrence_counts, names)
        inter_set = set(desired_objects).intersection(set(names))
        if with_class:
            object_condition = (len(inter_set) > 0)
        else:
            object_condition = (len(inter_set) == 0)
        #print('\n')
        #print(inter_set)
        if object_condition & (image_id in img_id_list): 
        #    print('\n')
        #    print('image id:')
        #    print(objects[idx]['image_id'])
           #print('\n')
           #print('inter_set:')
           #print(inter_set)
           #print(len(inter_set))
           #print('names')
           #print(names)
           #print('\n')
           #print('image url:')
           #print(im_url)
           count_occurrences(occurrence_counts, names)
           image_dir = str(data_path) + '/'
           output_dir = str(yolo_path)
           img_data = get_image_meta_data(objects, image_id)
           # print(img_data['image_id'])
           desired_img_ids.append(image_id)
            
            #label_path = convert_single_image_to_yolo(objects[0], class_map, image_dir, output_dir)
           label_path = convert_single_image_to_yolo(img_data, class_map, image_dir, output_dir)
           label_paths.append(label_path)

           image_counter += 1
        if image_counter == number_of_images:
            break
    return label_paths, occurrence_counts


# def visual_genome_to_yolo_data(objects_and_ids, paths, class_map):
    
#     data_path, yolo_path = paths

#     objects, desired_objects, img_id_list = objects_and_ids
    
#     desired_img_ids = []
    
#     label_paths = []
    
#     occurrence_counts = dict.fromkeys(desired_objects, 0)
    
    
#     for idx in list(range(len(objects))):
    
#         if objects[idx]['image_id'] == 1001:
#             continue
    
#         names = []
#         for idx_obj in list(range(len(objects[idx]['objects']))):
#             name = objects[idx]['objects'][idx_obj]['names']
#             #print(name)
#             names.extend(name)
#         #print(names)  
#         #print('trees' in names)
        
#         if ('image_url' not in list(objects[idx].keys())):
#             continue
    
#         image_id = objects[idx]['image_id']
#         #print('\n')
#         #print(image_id)
#         #print(type(image_id))
#         #count_occurrences(occurrence_counts, names)
#         inter_set = set(desired_objects).intersection(set(names))
#         #print('\n')
#         #print(inter_set)
#         if (len(inter_set) > 0) & (image_id in img_id_list): 
#         #    print('\n')
#         #    print('image id:')
#         #    print(objects[idx]['image_id'])
#            #print('\n')
#            #print('inter_set:')
#            #print(inter_set)
#            #print(len(inter_set))
#            #print('names')
#            #print(names)
#            #print('\n')
#            #print('image url:')
#            #print(im_url)
#            count_occurrences(occurrence_counts, names)
#            image_dir = str(data_path) + '/'
#            output_dir = str(yolo_path)
#            img_data = get_image_meta_data(objects, image_id)
#            # print(img_data['image_id'])
#            desired_img_ids.append(image_id)
            
#             #label_path = convert_single_image_to_yolo(objects[0], class_map, image_dir, output_dir)
#            label_path = convert_single_image_to_yolo(img_data, class_map, image_dir, output_dir)
#            label_paths.append(label_path)
            
#     return label_paths, occurrence_counts
    