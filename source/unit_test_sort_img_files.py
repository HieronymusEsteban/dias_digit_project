import unittest
from . import image_id_converter as img_id_c
from . import sort_img_files as sif
from pathlib import Path
from ultralytics import YOLO
import os

# Before executing the test the test images must be placed into the test_data
# folder. 

class TestCompleteImageIds(unittest.TestCase):
    def setUp(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.my_image_dir = os.path.join(script_dir, '../..', 'test_data')
        self.my_output_with_person_dir = os.path.join(script_dir, '../..', 'test_with_person')
        self.my_output_without_person_dir =  os.path.join(script_dir, '../..', 'test_without_person')
        self.my_model = YOLO("yolov8n.pt") 
        self.my_expected_output = ['111', '058', '025', '035'], [False, True, False, True]
    
    def test_ExpectedOutput(self):
        self.assertEqual(sif.sort_img_files(self.my_image_dir, self.my_model, 
        self.my_output_with_person_dir, self.my_output_without_person_dir), 
        self.my_expected_output)
    
if __name__ == "__main__":
    unittest.main()

