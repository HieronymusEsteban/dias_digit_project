import unittest
import image_id_converter as img_id_c
import sort_img_files as sif
from pathlib import Path
from ultralytics import YOLO
import os

root_path = Path('/Users/stephanehess/Documents/CAS_AML/dias_digit_project/project')

# Define paths
image_dir = root_path/"../test_data"  # Replace with your directory containing images
output_dir_with_person = root_path/"../test_with_person"  # Replace with output directory for images with persons
output_dir_without_person = root_path/"../test_without_person"  # Replace with output directory for images without persons

# Create output directories
#os.chdir(root_path/'..')
os.makedirs(output_dir_with_person, exist_ok=True)
os.makedirs(output_dir_without_person, exist_ok=True)
#os.chdir('root_path')

# Load the YOLOv5 model
model = YOLO("yolov8n.pt")  # Use yolov8n (nano) for faster inference

expected_output = ['111', '058', '025', '035'], [False, True, False, True]

class TestCompleteImageIds(unittest.TestCase):
    def setUp(self):
        self.my_image_dir = image_dir
        self.my_output_with_person_dir = output_dir_with_person
        self.my_output_without_person_dir =  output_dir_without_person
        self.my_model = model
        self.my_expected_output = expected_output 
    
    def test_ExpectedOutput(self):
        self.assertEqual(sif.sort_img_files(self.my_image_dir, self.my_model, 
        self.my_output_with_person_dir, self.my_output_without_person_dir), 
        self.my_expected_output)
    
if __name__ == "__main__":
    unittest.main()

