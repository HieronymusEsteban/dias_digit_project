import os
import shutil

from .visual_genome_data import get_file_by_id




def simulate_specific_old_effects(image_path, output_path):
    """
    Apply specific effects that match your old photos.
    Adjust these based on what you observe in your test images.
    """
    img = Image.open(image_path).convert('RGB')
    
    # Heavy JPEG compression (very low quality)
    img.save('temp.jpg', 'JPEG', quality=15)
    img = Image.open('temp.jpg')
    os.remove('temp.jpg')
    
    # Significant brightness reduction
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(0.8)
    
    # Low contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(0.9)
    
    # Add significant noise
    img_array = np.array(img)
    noise = np.random.normal(0, 0.1, img_array.shape).astype(np.uint8)
    img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    
    # Strong blur
    img = img.filter(ImageFilter.GaussianBlur(radius=1.5))
    
    img.save(output_path, 'JPEG', quality=85) 


def process_training_dataset_spec(input_dir, output_dir, augmentation_ratio=0.5):
    """
    Process a directory of training images to simulate old photo effects.
    
    Args:
        input_dir: Directory with original images
        output_dir: Directory to save processed images
        augmentation_ratio: Fraction of images to augment (0.5 = 50%)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif'))]
    
    for img_file in image_files:
        input_path = os.path.join(input_dir, img_file)
        
        # Always copy original
        original_output = os.path.join(output_dir, img_file)
        #img = Image.open(input_path)
        #img.save(original_output)
        

        # Create filename for augmented version
        name, ext = os.path.splitext(img_file)
        aug_filename = f"{name}_aged{ext}"
        aug_output = os.path.join(output_dir, aug_filename)
        #aug_output = os.path.join(output_dir, img_file)
        
        # Apply aging effects with random intensity
        intensity = random.uniform(0.3, 0.8)
        simulate_specific_old_effects(input_path, aug_output)
    
    print(f"Processed {len(image_files)} images in {input_dir}")


def copy_with_new_id(data_path_origin, data_path_destination, filename, new_id, file_extension):
    '''Take a file from one data_path_origin, adds a new id in the end, and moves it
    to data_path_destination.'''

    img_filename_parts = filename.split('.')[0]

    new_filename = img_filename_parts + '_' + str(new_id) + file_extension
    origin_file_path = os.path.join(str(data_path_origin), filename)
    destination_file_path = os.path.join(str(data_path_destination), new_filename)
    print(destination_file_path)
    shutil.copy(origin_file_path, destination_file_path)


def add_new_id_img_meta(old_effect_path, yolo_path, 
data_augmented_path, yolo_augmented_path, img_ids, 
max_id, tag, file_extensions):
    '''
    This function takes images from two directories and adds to ids at the end of the filenames, where one
    directory containes images and the other meta-data referring to the images. The modified files are moved to
    new directories.

    Args:
        old_effect_path: path to directory with artificially aged versions of original images.
        yolo_path: path to directory with meta-data files referring to the original images.
        data_augmented_path: directory to move the modified images to.
        yolo_augmented_path: directory to move the modified meta-data files to.
        img_ids: integer ids referring to the original image files and to the corresponding aged versions.
        max_id: largest id integer of original image files: the new ids start counting from max_id + 1
        tag: string in the filenames of the image files
        file_extensions: list of file extensions referring to image and meta-data files.
    '''
    next_id = max_id
    tag = 'aged'
    # print('\n')
    # print('yolo_path')
    # print(yolo_path)
    # print(os.listdir(yolo_path))
    for img_id in img_ids:
        next_id += 1
        
        img_files = get_file_by_id(old_effect_path, img_id, file_extensions[0])
        
        for img_file in img_files:
            if tag in img_file:
                
                img_file_tagged = img_file
        
        copy_with_new_id(old_effect_path, data_augmented_path, img_file_tagged, next_id, file_extensions[0])
        
        yolo_file = get_file_by_id(yolo_path, img_id, file_extensions[1])[0]
        
        copy_with_new_id(yolo_path, yolo_augmented_path, yolo_file, next_id, file_extensions[1])