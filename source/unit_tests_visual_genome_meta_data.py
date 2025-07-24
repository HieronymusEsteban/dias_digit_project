import unittest
from . import visual_genome_meta_data as vg_md

# Importing necessary modules
import os
import sys


class TestReadJsonToDict(unittest.TestCase):
    def setUp(self):
        # Get the directory where your script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # The test should result in no failures when loading objects_exp.json and it should 
        # result in 3 failures when loading objects_f.json. 
        self.my_file_path_input = os.path.join(script_dir, '../..', 'test_visual_genome', 'objects_exp.json')
        # self.my_file_path_input = os.path.join(script_dir, '../..', 'test_visual_genome', 'objects_f.json')
        self.my_expected_output = [
    {
        "image_id": 1,
        "objects": [
            {"synsets": ["tree.n.01"], "h": 557},
            {"synsets": ["hello"], "first_name": "peter", "object_id": 1058534, "x": 78}
        ],
        "image_url": "https://cs.stanford.edu/people/rak248/VG_100K_2/1.jpg",
        "another_key": "hello"
    },
    {
        "image_id": 2,
        "objects": [
            {"synsets": ["road.n.01"], "h": 254, "object_id": 1023841, "merged_object_ids": [], "names": ["road"], "w": 364, "y": 345, "x": 0}
        ],
        "image_url": "just_any_string"
    }
]
    
    def test_inputExists(self):
        self.assertIsNotNone(self.my_file_path_input)
    
    def test_inputType(self):
        self.assertIsInstance(self.my_file_path_input, str)

    def test_functReturnsSomething(self):
        self.assertIsNotNone(vg_md.read_json_to_dict(self.my_file_path_input))
    
    def test_outputType(self):
        self.assertEqual(type(vg_md.read_json_to_dict(self.my_file_path_input)), list)    

    def test_outputLength(self):
        self.assertEqual(len(self.my_expected_output), len(vg_md.read_json_to_dict(self.my_file_path_input)))
    
    def test_outputElementsType(self):
        self.assertEqual(type(vg_md.read_json_to_dict(self.my_file_path_input)[0]), dict)
    
    def test_outputSubElementsType(self):
        self.assertEqual(type(vg_md.read_json_to_dict(self.my_file_path_input)[0]['image_id']), int)

    def test_outputElementsKeys(self):
        self.assertEqual(len(vg_md.read_json_to_dict(self.my_file_path_input)[1].keys()), len(self.my_expected_output[1].keys()))

    def test_expectedOutput(self):
        self.assertEqual(self.my_expected_output, vg_md.read_json_to_dict(self.my_file_path_input))       

# if __name__ == "__main__":
#     unittest.main()



class TestCountOccurrences(unittest.TestCase):
    def setUp(self):
        # The following input elements should result in no errors: 
        desired_objects = ['eins', 'drei']
        input_list = ['eins', 'zwei', 'eins', 'drei', 'eins', 'vier']

        # The following input elements should result 1 error:
        # desired_objects = ['zwei', 'drei']
        # input_list = ['eins', 'zwei', 'eins', 'drei', 'eins', 'vier']

        # The following input elements should result in 1 error:
        # desired_objects = ['zwei', 'drei']
        # input_list = ['eins', 'zwei', 'eins', 'drei', 'eins', 'vier', 'drei']

        # The following input elements should result in 2 errors:
        # desired_objects = ['eins', 'zwei', 'drei']
        # input_list = ['eins', 'zwei', 'eins', 'drei', 'eins', 'vier']

        input_dict = dict.fromkeys(desired_objects, 0)
        self.my_input_dict = input_dict
        self.my_input_list = input_list
        expected_output_count = {'eins': 3, 'drei': 1}
        self.my_expected_output = expected_output_count
    
    def test_inputExists(self):
        self.assertIsNotNone(self.my_input_dict)
        self.assertIsNotNone(self.my_input_list)
    
    def test_inputType(self):
        self.assertIsInstance(self.my_input_dict, dict)
        self.assertIsInstance(self.my_input_list, list)

    def test_functReturnsSomething(self):
        self.assertIsNotNone(vg_md.count_occurrences(self.my_input_dict, self.my_input_list))
    
    def test_outputType(self):
        self.assertIsInstance(vg_md.count_occurrences(self.my_input_dict, self.my_input_list), dict)

    def test_outputLength(self):
        self.assertEqual(len(vg_md.count_occurrences(self.my_input_dict, self.my_input_list).keys()), len(self.my_expected_output.keys()))    

    def test_outputIdentity(self):
        self.assertEqual(vg_md.count_occurrences(self.my_input_dict, self.my_input_list), self.my_expected_output)


class TestGetImageIds(unittest.TestCase):
    def setUp(self):
        # The following input elements should result in no errors: 
        number_of_images = 4
        image_ids= [1, 2, 9, 11]

        # Get the directory where your script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))

        self.my_input_dir = os.path.join(script_dir, '../..', 'test_visual_genome')
        self.my_expected_number_of_imgs = number_of_images
        self.my_expected_output = image_ids

    def test_inputExists(self):
        self.assertIsNotNone(self.my_expected_number_of_imgs)
        self.assertIsNotNone(self.my_input_dir)
    
    def test_inputType(self):
        self.assertIsInstance(self.my_expected_number_of_imgs, int)
        self.assertIsInstance(self.my_input_dir, str)

    def test_functReturnsSomething(self):
        self.assertIsNotNone(vg_md.get_image_ids(self.my_input_dir))
    
    def test_outputType(self):
        self.assertIsInstance(vg_md.get_image_ids(self.my_input_dir), list)
    
    def test_outputLength(self):
        self.assertEqual(len(vg_md.get_image_ids(self.my_input_dir)), self.my_expected_number_of_imgs)
    
    def test_outputIdentity(self):
        # I compare sets rather than the lists because you cannot rely on os.listdir() (part of my function)
        # to return filenames in alphanumerical order!
        self.assertEqual(set(vg_md.get_image_ids(self.my_input_dir)), set(self.my_expected_output))

class TestGetImageMetaData(unittest.TestCase):
    def setUp(self):
        # Get the directory where your script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))

        self.my_input_path = os.path.join(script_dir, '../..', 'test_visual_genome', 'objects_exp.json')
        self.my_input_object = vg_md.read_json_to_dict(self.my_input_path)
        self.my_expected_output = {
        "image_id": 2,
        "objects": [
            {"synsets": ["road.n.01"], "h": 254, "object_id": 1023841, "merged_object_ids": [], "names": ["road"], "w": 364, "y": 345, "x": 0}
        ],
        "image_url": "just_any_string"
    }

    def test_inputExists(self):
        self.assertIsNotNone(self.my_input_path)
    
    def test_inputType(self):
        self.assertIsInstance(self.my_input_object, list)

    def test_functReturnsSomething(self):
        self.assertIsNotNone(vg_md.get_image_meta_data(self.my_input_object, 2))
    
    def test_outputType(self):
        self.assertIsInstance(vg_md.get_image_meta_data(self.my_input_object, 2), dict)
    
    def test_outputLength(self):
        self.assertEqual(len(vg_md.get_image_meta_data(self.my_input_object, 2)), 3)
    
    def test_outputIdentity(self):
        self.assertEqual(vg_md.get_image_meta_data(self.my_input_object, 2), self.my_expected_output)

# bboxes_from_metadata(image_path, objects, desired_object)
class TestBboxesFromMetaData(unittest.TestCase):
    def setUp(self):
        # Get the directory where your script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))

        self.my_input_objects_path = os.path.join(script_dir, '../..', 'test_visual_genome', 'objects_first3.json')
        self.my_input_image_path = os.path.join(script_dir, '../..', 'test_visual_genome', 'visual_genome_1.jpg')
        self.my_input_objects = vg_md.read_json_to_dict(self.my_input_objects_path)
        self.my_desired_object = 'man'
        self.my_expected_output_length = 2

    def test_inputExists(self):
        self.assertIsNotNone(self.my_input_image_path)
        self.assertIsNotNone(self.my_input_objects)
        self.assertIsNotNone(self.my_desired_object)
    
    def test_inputType(self):
        self.assertIsInstance(self.my_input_image_path, str)
        self.assertIsInstance(self.my_input_objects, list)
        self.assertIsInstance(self.my_desired_object, str)

    def test_functReturnsSomething(self):
        self.assertIsNotNone(vg_md.bboxes_from_metadata(self.my_input_image_path, self.my_input_objects, self.my_desired_object))
    
    def test_outputType(self):
        self.assertIsInstance(vg_md.bboxes_from_metadata(self.my_input_image_path, self.my_input_objects, self.my_desired_object), list)
    
    def test_outputLength(self):
        self.assertEqual(len(vg_md.bboxes_from_metadata(self.my_input_image_path, self.my_input_objects, self.my_desired_object)), 2)

if __name__ == "__main__":
    unittest.main()

