import unittest
#import shutil
from PIL import Image
from . import llm_input as llm_i

#from pathlib import Path
import os


class TestConvertTifToGpg(unittest.TestCase):
    def setUp(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.tif_dir = os.path.join(script_dir, '../..', 'test_llm_input/data_tif_to_jpg')
        self.jpg_dir = os.path.join(script_dir, '../..', 'test_llm_input/data_jpg')
        self.expected_formats_tif_dir_before = ['tiff', 'tiff']
        self.expected_formats_jpg_dir_after = ['jpeg', 'jpeg']
        self.expected_formats_tif_dir_after = ['jpeg', 'tiff', 'jpeg', 'tiff']
        self.expected_formats_tif_dir_after_test_test = ['jpeg', 'tiff', 'jpeg', 'jpeg'] # This is supposed to yield a test failure in order to test the test.
        self.expected_file_list_original = ['BernerOberland001.tif', 'BernerOberland_111.tif']
        self.expected_file_list_converted = ['BernerOberland001.jpg', 'BernerOberland_111.jpg']
    
    def test_ExpectedFormat(self):
        # Delete converted files in jpg_dir from previous test:
        file_list = os.listdir(self.jpg_dir)
        for file in file_list:
            file_path = os.path.join(self.jpg_dir , file)
            print(file_path)
            if os.path.isfile(file_path):
                os.remove(file_path)

        # Get the formats of the files in tif_dir
        # before executing the function:
        file_list = os.listdir(self.tif_dir)
        file_list.sort()
        file_formats = []
        for file in file_list:
            file_path = os.path.join(self.tif_dir, file)
            image = Image.open(file_path)
            actual_format = image.format.lower()
            file_formats.append(actual_format)
        
        # Verify file formats of original files:
        self.assertEqual(self.expected_formats_tif_dir_before, file_formats)

        # Verify the file list of jpg_dir (should be empty):
        file_list = os.listdir(self.jpg_dir)
        file_list.sort()
        self.assertEqual(file_list, [])

        # Execute function:
        llm_i.convert_tif_to_jpg(self.tif_dir, self.jpg_dir, quality=85)

        # Get the formats of the files in jpg_dir
        # after executing the function:
        file_list = os.listdir(self.jpg_dir)
        file_list.sort()
        file_formats = []
        for file in file_list:
            file_path = os.path.join(self.jpg_dir, file)
            image = Image.open(file_path)
            actual_format = image.format.lower()
            file_formats.append(actual_format)

        # Check if the files were converted to the desired format:
        self.assertEqual(self.expected_formats_jpg_dir_after, file_formats)

        # Check if the test is working by comparing to unexpected values:
        self.assertEqual(self.expected_formats_jpg_dir_after_test_test, file_formats)


    def test_ExpectedFileList(self):
        # Delete converted files in jpg_dir from previous test:
        file_list = os.listdir(self.jpg_dir)
        for file in file_list:
            file_path = os.path.join(self.jpg_dir , file)
            print(file_path)
            if os.path.isfile(file_path):
                os.remove(file_path)

        # Get the file list of tif_dir:
        file_list = os.listdir(self.tif_dir)
        file_list.sort()
        
        # Compare file list to expected file list:
        self.assertEqual(self.expected_file_list_original, file_list)

        # Verify the file list of jpg_dir (should be empty):
        file_list = os.listdir(self.jpg_dir)
        file_list.sort()
        self.assertEqual(file_list, [])

        # Execute function:
        llm_i.convert_tif_to_jpg(self.tif_dir, self.jpg_dir, quality=85)

        # Check file list of converted files in jpg_dir:
        file_list = os.listdir(self.jpg_dir)
        file_list.sort()

        self.assertEqual(self.expected_file_list_converted, file_list)


class TestConvertImgIfNeeded(unittest.TestCase):
    def setUp(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.tif_conv_dir = os.path.join(script_dir, '../..', 'test_llm_input/data_tif_jpg_conv')
        self.expected_file_list_before = ['BernerOberland001.tif', 'BernerOberland_111.tif']
        self.expected_file_list_after = ['BernerOberland001.tif',
 'BernerOberland001_converted.jpg',
 'BernerOberland_111.tif',
 'BernerOberland_111_converted.jpg']
    
    def test_ExpectedFormat(self):
        # Delete converted files in jpg_dir from previous test:
        file_list = os.listdir(self.tif_conv_dir)
        for file in file_list:
            file_path = os.path.join(self.tif_conv_dir , file)

            name_part = file.split('.')[0]
            if name_part.split('_')[-1] == 'converted':
                print(name_part)
                os.remove(file_path)
        
        # Get file list:
        file_list = os.listdir(self.tif_conv_dir)
        file_list.sort()

        # # Check file list:
        self.assertEqual(file_list, self.expected_file_list_before)

        # # Execute function:
        file_list = os.listdir(self.tif_conv_dir)
        for file in file_list:
            file_path = os.path.join(self.tif_conv_dir, file)
            file_path_conv = llm_i.convert_image_if_needed(file_path)
        
        # Get file list:
        file_list = os.listdir(self.tif_conv_dir)
        file_list.sort()

        # # Check file list:
        self.assertEqual(file_list, self.expected_file_list_after)
        

class TestCallOllamaModel(unittest.TestCase):
    def setUp(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.llm_test_dir = os.path.join(script_dir, '../..', 'test_llm_input/data_llm_test')
        self.promp = 'Analyze this image and say yes if you see one or multiple houses in the image, say no if not.'
        self.expected_output_type = str
        self.expected_answer = 'yes'

    def test_LlmResponse(self):
        # Delete converted files in jpg_dir from previous test:
        file_list = os.listdir(self.llm_test_dir)
        file = file_list[0]
        file_path = os.path.join(self.llm_test_dir, file)
        output = llm_i.call_ollama_model(file_path, self.promp)

        # Check output data type:
        self.assertEqual(type(output), self.expected_output_type)

        # Check first three letters of answer:
        self.assertEqual(output[0:3].lower(), self.expected_answer)

    
if __name__ == "__main__":
    unittest.main()



