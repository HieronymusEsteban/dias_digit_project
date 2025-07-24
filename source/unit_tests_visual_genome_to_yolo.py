import unittest
import os
from . import visual_genome_to_yolo as vg_ylo
from . import visual_genome_meta_data as vg_md
from . import visual_genome_data as vg_dt 
import yaml


class TestCreateClassMapping(unittest.TestCase):
    def setUp(self):
        # Get the directory where your script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # The first input should result in no failures, the second and third will result in failures:
        self.my_input = ['Eymatt', 'Eichholz', 'Gurten']
        self.my_expected_output = {'Eymatt': 0, 'Eichholz': 1, 'Gurten': 2}
        #self.my_expected_output = {'Eymatt': 0, 'Eichholz': 6, 'Gurten': 2}
        #self.my_expected_output = {'Eymatt': 0, 'Eichholz': 1, 'München': 2}

    def test_inputExists(self):
        self.assertIsNotNone(self.my_input)

    def test_inputType(self):
        self.assertIsInstance(self.my_input, list)
        self.assertIsInstance(self.my_input[0], str)
    
    def test_functionReturnsSomething(self):
        self.assertIsNotNone(vg_ylo.create_class_mapping_from_list(self.my_input))

    def test_expectedOutput(self):
        self.assertEqual(vg_ylo.create_class_mapping_from_list(self.my_input), self.my_expected_output)


class TestSaveClassMap(unittest.TestCase):
    def setUp(self):
        # Get the directory where your script is located:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Define path to folder used for test files:
        self.test_dir_path = os.path.join(script_dir, '../..', 'test_visual_genome')
        # Path to output file:
        self.path_yaml_file = os.path.join(self.test_dir_path, 'class_map.yaml')
        self.input_dictionary = {'Eymatt': 0, 'Eichholz': 1, 'Gurten': 2}
        # Expected content in the yaml files (I check the set of names rather than the list
        # because in earlier Python versions the order of elements in dictionaries were not guaranteed):
        self.expectedYamlNames = {'Eymatt', 'Eichholz', 'Gurten'}
        self.expectedYamlLength = len(self.expectedYamlNames)
        # Content that differs from the expected content (should yield failures):
        #self.expectedYamlNames = {'Eymatt', 'Eichholz', 'München'}
        #self.expectedYamlLength = 5
        # Path to a wrong yaml file, should yield failures: 
        self.path_wrong_yaml_file = os.path.join(self.test_dir_path, 'different_file.yaml')

    def test_inputExists(self):
        self.assertIsNotNone(self.path_yaml_file)
        self.assertIsNotNone(self.input_dictionary)
    
    def test_functionOutputsYaml(self):
        vg_ylo.save_class_map_to_yaml(self.input_dictionary, self.path_yaml_file)
        os.path.exists(self.path_yaml_file)
    
    def test_expectedYamlContent(self):
        vg_ylo.save_class_map_to_yaml(self.input_dictionary, self.path_yaml_file)
        # The following two lines should yield no failures:
        with open(self.path_yaml_file, 'r') as f:
            data = yaml.safe_load(f)
        # The following two lines should yield failures:
        # with open(self.path_wrong_yaml_file, 'r') as f:
        #     data = yaml.safe_load(f)
        self.assertEqual(set(data['names']), self.expectedYamlNames)
        self.assertEqual(data['nc'], self.expectedYamlLength)


class TestReadYamlToClassMap(unittest.TestCase):
    def setUp(self):
        # Get the directory where your script is located:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Define path to folder used for test files:
        self.test_dir_path = os.path.join(script_dir, '../..', 'test_visual_genome')
        # Path to yaml file 1 and corresponding expected output:
        self.path_yaml_file = os.path.join(self.test_dir_path, 'class_map.yaml')
        self.expected_output_1 = {'Eymatt': 0, 'Eichholz': 1, 'Gurten': 2}
        # Path to yaml file 2 and corresponding expected output:
        self.path_different_yaml_file = os.path.join(self.test_dir_path, 'different_file.yaml')
        self.expected_output_2 = {'tree': 0, 'man': 1}

    def test_inputExists(self):
        self.assertIsNotNone(self.path_yaml_file)
        self.assertIsNotNone(self.path_different_yaml_file)

    def test_InputType(self):
        self.assertIsInstance(self.path_yaml_file, str)
        self.assertIsInstance(self.path_different_yaml_file, str)
    
    def test_Output(self):
        self.assertEqual(vg_ylo.read_yaml_to_class_map(self.path_yaml_file), self.expected_output_1)
        self.assertEqual(vg_ylo.read_yaml_to_class_map(self.path_different_yaml_file), self.expected_output_2)


class Test_ConvertSingleImageToYolo(unittest.TestCase):
    def setUp(self):
        # Get the directory where your script is located:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Define path to folder used for test files:
        self.img_dir = os.path.join(script_dir, '../..', 'test_visual_genome/convert_single_img_test')
        self.yolo_dir = os.path.join(script_dir, '../..', 'test_visual_genome_yolo/convert_single_img_test')
        self.object_metadata = {'image_id': 1,
                                'objects': [{'synsets': ['tree.n.01'],
                                'h': 557,
                                'object_id': 1058549,
                                'merged_object_ids': [],
                                'names': ['trees'],
                                'w': 799,
                                'y': 0,
                                'x': 0},
                                {'synsets': ['sidewalk.n.01'],
                                'h': 290,
                                'object_id': 1058534,
                                'merged_object_ids': [5046],
                                'names': ['sidewalk'],
                                'w': 722,
                                'y': 308,
                                'x': 78},
                                {'synsets': ['building.n.01'],
                                'h': 538,
                                'object_id': 1058508,
                                'merged_object_ids': [],
                                'names': ['building'],
                                'w': 222,
                                'y': 0,
                                'x': 1},
                                {'synsets': ['street.n.01'],
                                'h': 258,
                                'object_id': 1058539,
                                'merged_object_ids': [3798578],
                                'names': ['street'],
                                'w': 359,
                                'y': 283,
                                'x': 439},
                                {'synsets': ['wall.n.01'],
                                'h': 535,
                                'object_id': 1058543,
                                'merged_object_ids': [],
                                'names': ['wall'],
                                'w': 135,
                                'y': 1,
                                'x': 0},
                                {'synsets': ['tree.n.01'],
                                'h': 360,
                                'object_id': 1058545,
                                'merged_object_ids': [],
                                'names': ['tree'],
                                'w': 476,
                                'y': 0,
                                'x': 178},
                                {'synsets': ['shade.n.01'],
                                'h': 189,
                                'object_id': 5045,
                                'merged_object_ids': [],
                                'names': ['shade'],
                                'w': 274,
                                'y': 344,
                                'x': 116},
                                {'synsets': ['van.n.05'],
                                'h': 176,
                                'object_id': 1058542,
                                'merged_object_ids': [1058536],
                                'names': ['van'],
                                'w': 241,
                                'y': 278,
                                'x': 533},
                                {'synsets': ['trunk.n.01'],
                                'h': 348,
                                'object_id': 5055,
                                'merged_object_ids': [],
                                'names': ['tree trunk'],
                                'w': 78,
                                'y': 213,
                                'x': 623},
                                {'synsets': ['clock.n.01'],
                                'h': 363,
                                'object_id': 1058498,
                                'merged_object_ids': [],
                                'names': ['clock'],
                                'w': 77,
                                'y': 63,
                                'x': 422},
                                {'synsets': ['window.n.01'],
                                'h': 147,
                                'object_id': 3798579,
                                'merged_object_ids': [],
                                'names': ['windows'],
                                'w': 198,
                                'y': 1,
                                'x': 602},
                                {'synsets': ['man.n.01'],
                                'h': 248,
                                'object_id': 3798576,
                                'merged_object_ids': [1058540],
                                'names': ['man'],
                                'w': 82,
                                'y': 264,
                                'x': 367},
                                {'synsets': ['man.n.01'],
                                'h': 259,
                                'object_id': 3798577,
                                'merged_object_ids': [],
                                'names': ['man'],
                                'w': 57,
                                'y': 254,
                                'x': 238},
                                {'synsets': [],
                                'h': 430,
                                'object_id': 1058548,
                                'merged_object_ids': [],
                                'names': ['lamp post'],
                                'w': 43,
                                'y': 63,
                                'x': 537},
                                {'synsets': ['sign.n.02'],
                                'h': 179,
                                'object_id': 1058507,
                                'merged_object_ids': [],
                                'names': ['sign'],
                                'w': 78,
                                'y': 13,
                                'x': 123},
                                {'synsets': ['car.n.01'],
                                'h': 164,
                                'object_id': 1058515,
                                'merged_object_ids': [],
                                'names': ['car'],
                                'w': 80,
                                'y': 342,
                                'x': 719},
                                {'synsets': ['back.n.01'],
                                'h': 164,
                                'object_id': 5060,
                                'merged_object_ids': [],
                                'names': ['back'],
                                'w': 70,
                                'y': 345,
                                'x': 716},
                                {'synsets': ['jacket.n.01'],
                                'h': 98,
                                'object_id': 1058530,
                                'merged_object_ids': [],
                                'names': ['jacket'],
                                'w': 82,
                                'y': 296,
                                'x': 367},
                                {'synsets': ['car.n.01'],
                                'h': 95,
                                'object_id': 5049,
                                'merged_object_ids': [],
                                'names': ['car'],
                                'w': 78,
                                'y': 319,
                                'x': 478},
                                {'synsets': ['trouser.n.01'],
                                'h': 128,
                                'object_id': 1058531,
                                'merged_object_ids': [],
                                'names': ['pants'],
                                'w': 48,
                                'y': 369,
                                'x': 388},
                                {'synsets': ['shirt.n.01'],
                                'h': 103,
                                'object_id': 1058511,
                                'merged_object_ids': [],
                                'names': ['shirt'],
                                'w': 54,
                                'y': 287,
                                'x': 241},
                                {'synsets': ['parking_meter.n.01'],
                                'h': 143,
                                'object_id': 1058519,
                                'merged_object_ids': [],
                                'names': ['parking meter'],
                                'w': 26,
                                'y': 325,
                                'x': 577},
                                {'synsets': ['trouser.n.01'],
                                'h': 118,
                                'object_id': 1058528,
                                'merged_object_ids': [],
                                'names': ['pants'],
                                'w': 44,
                                'y': 384,
                                'x': 245},
                                {'synsets': ['shirt.n.01'],
                                'h': 102,
                                'object_id': 1058547,
                                'merged_object_ids': [],
                                'names': ['shirt'],
                                'w': 82,
                                'y': 295,
                                'x': 368},
                                {'synsets': ['shoe.n.01'],
                                'h': 28,
                                'object_id': 1058525,
                                'merged_object_ids': [5048],
                                'names': ['shoes'],
                                'w': 48,
                                'y': 485,
                                'x': 388},
                                {'synsets': ['arm.n.01'],
                                'h': 41,
                                'object_id': 1058546,
                                'merged_object_ids': [],
                                'names': ['arm'],
                                'w': 30,
                                'y': 285,
                                'x': 370},
                                {'synsets': ['bicycle.n.01'],
                                'h': 36,
                                'object_id': 1058535,
                                'merged_object_ids': [],
                                'names': ['bike'],
                                'w': 27,
                                'y': 319,
                                'x': 337},
                                {'synsets': ['bicycle.n.01'],
                                'h': 41,
                                'object_id': 5051,
                                'merged_object_ids': [],
                                'names': ['bike'],
                                'w': 27,
                                'y': 311,
                                'x': 321},
                                {'synsets': ['headlight.n.01'],
                                'h': 9,
                                'object_id': 5050,
                                'merged_object_ids': [],
                                'names': ['headlight'],
                                'w': 18,
                                'y': 370,
                                'x': 517},
                                {'synsets': ['spectacles.n.01'],
                                'h': 23,
                                'object_id': 1058518,
                                'merged_object_ids': [],
                                'names': ['glasses'],
                                'w': 43,
                                'y': 317,
                                'x': 448},
                                {'synsets': ['chin.n.01'],
                                'h': 8,
                                'object_id': 1058541,
                                'merged_object_ids': [],
                                'names': ['chin'],
                                'w': 9,
                                'y': 288,
                                'x': 401}],
                                'image_url': 'https://cs.stanford.edu/people/rak248/VG_100K_2/1.jpg'}
        obj_list = [self.object_metadata['objects'][idx]['names'][0] for idx in range(0, len(self.object_metadata['objects']))]
        self.desired_object_class = 'tree'
        self.class_map = {self.desired_object_class: 0}
        self.expected_output_length = obj_list.count(self.desired_object_class)

    def test_FunctionReturnsSomething(self):
        path = vg_ylo.convert_single_image_to_yolo(self.object_metadata, self.class_map, self.img_dir, self.yolo_dir)
        self.assertIsNotNone(path)
    
    def test_outputFileContentLength(self):
        file_path = os.path.join(self.yolo_dir, 'visual_genome_1.txt')

        with open(file_path, 'r') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), self.expected_output_length)

    
class Test_VisualGenomeToYoloData(unittest.TestCase):
    def setUp(self):
        # Get the directory where your script is located:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Define path to folder used for test files:
        self.img_dir = os.path.join(script_dir, '../..', 'test_visual_genome')
        self.yolo_dir = os.path.join(script_dir, '../..', 'test_visual_genome_yolo')
        objects_file_path = os.path.join(self.img_dir, 'objects_first3.json')
        self.objects_metadata = vg_md.read_json_to_dict(objects_file_path)
        self.desired_objects = ['man']
        self.image_id_list = vg_dt.get_image_ids(self.img_dir)
        self.class_map = {self.desired_objects[0]: 0}

    def test_FunctionReturnsSomething(self):
        objects_and_ids = (self.objects_metadata, self.desired_objects, self.image_id_list)
        paths = (self.img_dir, self.yolo_dir)

        #label_paths_w, occurrence_counts = visual_genome_to_yolo_data(objects_and_ids, paths, class_map)
        #len(label_paths_w)

        label_paths_w, occurrence_counts = vg_ylo.visual_genome_to_yolo_data_n(objects_and_ids, paths, self.class_map)
        
        self.assertIsNotNone(label_paths_w, occurrence_counts)

    def test_OutputLength(self):
        objects_and_ids = (self.objects_metadata, self.desired_objects, self.image_id_list)
        paths = (self.img_dir, self.yolo_dir)

        #label_paths_w, occurrence_counts = visual_genome_to_yolo_data(objects_and_ids, paths, class_map)
        #len(label_paths_w)

        label_paths_w, occurrence_counts = vg_ylo.visual_genome_to_yolo_data_n(objects_and_ids, paths, self.class_map)
        
        self.assertEqual(len(label_paths_w), 2)
        self.assertEqual(occurrence_counts[self.desired_objects[0]], 3)



if __name__ == "__main__":
    unittest.main()