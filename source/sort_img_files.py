
from pathlib import Path
from PIL import Image
import os
import shutil


def sort_img_files(image_dir, model, output_dir_with_person, output_dir_without_person, threshold = 0.25):
    # Create empty lists to store the image ids and person detection results:
    img_ids = []
    with_person = []

    # Iterate through images
    for image_path in Path(image_dir).glob("*.tif"):
        path_str = str(image_path)
        parts = path_str.split('.tif')
        img_id = parts[-2][-3:]
        img_ids.append(img_id)

        try:
            # Ensure the file is an image
            img = Image.open(image_path)
            img.verify()

            # Perform object detection
            results = model(image_path, verbose=False, conf=threshold)

            # Check if a person is detected
            has_person = any(int(box[5]) == 0 for box in results[0].boxes.data.tolist())  # Class ID 0 is for 'person'
            
            with_person.append(has_person)
            
            # Move image to the corresponding folder
            if has_person:
                shutil.move(str(image_path), os.path.join(output_dir_with_person, image_path.name))
            else:
                shutil.move(str(image_path), os.path.join(output_dir_without_person, image_path.name))

        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    print("Filtering complete!")
    return img_ids, with_person