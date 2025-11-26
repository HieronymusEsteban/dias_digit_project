
from pathlib import Path
from PIL import Image
import os
import shutil


import cv2
import numpy as np



def detect_persons_yolo(image_dir, model, threshold=0.25, file_format="tif"):
    img_ids = []
    with_person = []
    for image_path in Path(image_dir).glob(f"*.{file_format}"):
        path_str = str(image_path)
        parts = path_str.split(f'.{file_format}')
        img_id = parts[-2][-3:]
        img_ids.append(img_id)
        try:
            # Load image with OpenCV for better control
            img = cv2.imread(str(image_path))
            # If OpenCV fails, try PIL
            if img is None:
                print(f"OpenCV failed, trying PIL for {image_path.name}")
                pil_img = Image.open(image_path).convert('RGB')
                img = np.array(pil_img)
                # Convert RGB to BGR for consistency (OpenCV uses BGR)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # CRITICAL FIX: Ensure RGB format
            if img is None:
                print(f"Could not read {image_path}")
                continue
                
            # Convert grayscale to RGB if needed
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
            # Perform object detection with the properly formatted image
            results = model(img, verbose=False, conf=threshold)
            # Check if a person is detected
            has_person = any(int(box[5]) == 0 for box in results[0].boxes.data.tolist()) # Class ID 0 is for 'person'
            
            with_person.append(has_person)
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    print("Filtering complete!")
    return img_ids, with_person

    

def sort_img_files(image_dir, model, output_dir_with_person, output_dir_without_person, threshold=0.25):
    img_ids = []
    with_person = []

    for image_path in Path(image_dir).glob("*.tif"):
        path_str = str(image_path)
        parts = path_str.split('.tif')
        img_id = parts[-2][-3:]
        img_ids.append(img_id)

        try:
            # Load image with OpenCV for better control
            img = cv2.imread(str(image_path))
            
            # CRITICAL FIX: Ensure RGB format
            if img is None:
                print(f"Could not read {image_path}")
                continue
                
            # Convert grayscale to RGB if needed
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
            # Perform object detection with the properly formatted image
            results = model(img, verbose=False, conf=threshold)

            # Check if a person is detected
            has_person = any(int(box[5]) == 0 for box in results[0].boxes.data.tolist()) # Class ID 0 is for 'person'
            
            with_person.append(has_person)
            
           # # Move image to the corresponding folder
           # if has_person:
           #     shutil.move(str(image_path), os.path.join(output_dir_with_person, image_path.name))
           # else:
           #     shutil.move(str(image_path), os.path.join(output_dir_without_person, image_path.name))

        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    print("Filtering complete!")
    return img_ids, with_person