import unittest
from . import visual_genome_data as vg_d

# Importing necessary modules
import os
import sys
import matplotlib.pyplot as plt
from PIL import Image
import shutil


class TestConvertURL(unittest.TestCase):
    def setUp(self):
        self.my_input_1 = 'https://cs.stanford.edu/people/rak248/VG_100K/2317239.jpg'
        self.my_input_2 = 'https://cs.stanford.edu/people/rak248/VG_100K_2/2317239.jpg'
        self.my_expected_output_1 = self.my_input_2
        self.my_expected_output_2 = self.my_input_1
    
    def test_inputExists(self):
        self.assertIsNotNone(self.my_input_1)
        self.assertIsNotNone(self.my_input_2)
    
    def test_inputType(self):
        self.assertIsInstance(self.my_input_1, str)
        self.assertIsInstance(self.my_input_2, str)

    def test_functReturnsSomething(self):
        self.assertIsNotNone(vg_d.convert_url(self.my_input_1))
        self.assertIsNotNone(vg_d.convert_url(self.my_input_2))
    
    def test_outputType(self):
        self.assertIsInstance(vg_d.convert_url(self.my_input_1), str)
        self.assertIsInstance(vg_d.convert_url(self.my_input_2), str)  

    def test_outputLength(self):
        self.assertEqual(len(self.my_expected_output_1), len(vg_d.convert_url(self.my_input_1)))
        self.assertEqual(len(self.my_expected_output_2), len(vg_d.convert_url(self.my_input_2)))
    
    def test_outputIdentity(self):
        self.assertEqual(self.my_expected_output_1, vg_d.convert_url(self.my_input_1))
        self.assertEqual(self.my_expected_output_2, vg_d.convert_url(self.my_input_2))

#download_image(original_url, directory_path, change_url=False)
class TestDownLoadImage(unittest.TestCase):
    def setUp(self):
        # Get the directory where my script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Make path to directory where the downloaded image is to be saved:  
        self.my_data_dir = os.path.join(script_dir, '../..', 'test_visual_genome')
        # The first two image urls should result in no errors, whereas the nonsense urls should result in errors:
        self.my_input_1 = 'https://cs.stanford.edu/people/rak248/VG_100K/2317239.jpg'
        self.my_input_2 = 'https://cs.stanford.edu/people/rak248/VG_100K/3000.jpg'
        # self.my_input_1 = 'https://cs.stanford.edu/people/rak248/VG_100K/Nonsense_1.jpg'
        # self.my_input_2 = 'https://cs.stanford.edu/people/rak248/VG_100K/Nonsense_2.jpg'
        self.img_filename_1= 'visual_genome_' + self.my_input_1.split('/')[-1]
        self.img_filename_2 = 'visual_genome_' + self.my_input_2.split('/')[-1]
        #self.my_expected_output_1 = self.my_expected_input_2
        #self.my_expected_output_2 = self.my_expected_input_1
    
    def test_inputExists(self):
        self.assertIsNotNone(self.my_input_1)
        self.assertIsNotNone(self.my_input_2)
        self.assertIsNotNone(self.my_data_dir)

    def test_inputType(self):
        self.assertIsInstance(self.my_input_1, str)
        self.assertIsInstance(self.my_input_2, str)
        self.assertIsInstance(self.my_data_dir, str)

    def test_functReturnsSomething(self):
        self.assertIsNotNone(vg_d.download_image(self.my_input_1, self.my_data_dir))
        self.assertIsNotNone(vg_d.download_image(self.my_input_2, self.my_data_dir, change_url=True))

        # Move downloaded images to archive folder so the next test can start fresh (without images from previous tests):
        self.my_image_path_1 = os.path.join(self.my_data_dir, self.img_filename_1)
        self.my_image_path_2 = os.path.join(self.my_data_dir, self.img_filename_2)

        self.dest_path_1 = os.path.join(self.my_data_dir,'archive', self.img_filename_1)
        self.dest_path_2 = os.path.join(self.my_data_dir,'archive', self.img_filename_2)

        if os.path.exists(self.my_image_path_1):
            shutil.move(self.my_image_path_1, self.dest_path_1)

        if os.path.exists(self.my_image_path_2):
            shutil.move(self.my_image_path_2, self.dest_path_2)
    
    def test_downloadedImageExists(self):
        # Download images: 
        vg_d.download_image(self.my_input_1, self.my_data_dir)
        vg_d.download_image(self.my_input_2, self.my_data_dir, change_url=True)
        self.my_image_path_1 = os.path.join(self.my_data_dir, self.img_filename_1)
        self.my_image_path_2 = os.path.join(self.my_data_dir, self.img_filename_2)
        self.assertIsNotNone(self.my_data_dir)
        self.assertTrue(os.path.exists(self.my_image_path_1))
        self.assertTrue(os.path.exists(self.my_image_path_2))

        # Move downloaded images to archive folder so the next test can start fresh (without images from previous tests):
        self.dest_path_1 = os.path.join(self.my_data_dir,'archive', self.img_filename_1)
        self.dest_path_2 = os.path.join(self.my_data_dir,'archive', self.img_filename_2)

        if os.path.exists(self.my_image_path_1):
            shutil.move(self.my_image_path_1, self.dest_path_1)

        if os.path.exists(self.my_image_path_2):
            shutil.move(self.my_image_path_2, self.dest_path_2)

    def test_downloadedImageSize(self):
        vg_d.download_image(self.my_input_1, self.my_data_dir)
        vg_d.download_image(self.my_input_2, self.my_data_dir, change_url=True)
        self.my_image_path_1 = os.path.join(self.my_data_dir, self.img_filename_1)
        self.my_image_path_2 = os.path.join(self.my_data_dir, self.img_filename_2)
        self.assertIsNotNone(self.my_data_dir)
        self.assertTrue(os.path.getsize(self.my_image_path_1)>0)
        self.assertTrue(os.path.getsize(self.my_image_path_2)>0)

        # Move downloaded images to archive folder so the next test can start fresh (without images from previous tests):
        self.dest_path_1 = os.path.join(self.my_data_dir,'archive', self.img_filename_1)
        self.dest_path_2 = os.path.join(self.my_data_dir,'archive', self.img_filename_2)

        if os.path.exists(self.my_image_path_1):
            shutil.move(self.my_image_path_1, self.dest_path_1)

        if os.path.exists(self.my_image_path_2):
            shutil.move(self.my_image_path_2, self.dest_path_2)
    
    def test_downloadedImageValid(self):
        vg_d.download_image(self.my_input_1, self.my_data_dir)
        vg_d.download_image(self.my_input_2, self.my_data_dir, change_url=True)
        self.my_image_path_1 = os.path.join(self.my_data_dir, self.img_filename_1)
        self.my_image_path_2 = os.path.join(self.my_data_dir, self.img_filename_2)
        try:
            Image.open(self.my_image_path_1).verify()
        except Exception:
            self.fail("Image is not valid")
        try:
            Image.open(self.my_image_path_2).verify()
        except Exception:
            self.fail("Image is not valid")

        # Move downloaded images to archive folder so the next test can start fresh (without images from previous tests):
        self.dest_path_1 = os.path.join(self.my_data_dir,'archive', self.img_filename_1)
        self.dest_path_2 = os.path.join(self.my_data_dir,'archive', self.img_filename_2)

        if os.path.exists(self.my_image_path_1):
            shutil.move(self.my_image_path_1, self.dest_path_1)

        if os.path.exists(self.my_image_path_2):
            shutil.move(self.my_image_path_2, self.dest_path_2)

    def test_showImage(self):
        vg_d.download_image(self.my_input_1, self.my_data_dir)
        vg_d.download_image(self.my_input_2, self.my_data_dir, change_url=True)
        self.my_image_path_1 = os.path.join(self.my_data_dir, self.img_filename_1)
        self.my_image_path_2 = os.path.join(self.my_data_dir, self.img_filename_2)
        img = Image.open(self.my_image_path_1)
        plt.imshow(img)
        plt.show()

        img = Image.open(self.my_image_path_2)
        plt.imshow(img)
        plt.show()
        
        # Move downloaded images to archive folder so the next test can start fresh (without images from previous tests):
        self.dest_path_1 = os.path.join(self.my_data_dir,'archive', self.img_filename_1)
        self.dest_path_2 = os.path.join(self.my_data_dir,'archive', self.img_filename_2)

        if os.path.exists(self.my_image_path_1):
            shutil.move(self.my_image_path_1, self.dest_path_1)

        if os.path.exists(self.my_image_path_2):
            shutil.move(self.my_image_path_2, self.dest_path_2)


class TestGetImageIds(unittest.TestCase):
    def setUp(self):
        # Get the directory where my script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Make path to directory where the image files are located:  
        self.my_data_dir = os.path.join(script_dir, '../..', 'test_visual_genome')
        # Define input: 
        self.my_input = self.my_data_dir

        # Expected output: List of image ids of images in the test_visual_genome folder
        # (visual_genome_1.jpg, visual_genome_2.jpg, visual_genome_9.jpg, visual_genome_11.jpg): 
        self.my_expected_output = [1, 2, 9, 11]
    
    def test_inputExists(self):
        self.assertIsNotNone(self.my_input)

    def test_inputType(self):
        self.assertIsInstance(self.my_input, str)

    def test_functReturnsSomething(self):
        self.assertIsNotNone(vg_d.get_image_ids(self.my_input))

    def test_outputType(self):
        self.assertIsInstance(vg_d.get_image_ids(self.my_input), list)
        self.assertIsInstance(vg_d.get_image_ids(self.my_input)[0], int)  

    def test_outputLength(self):
        self.assertEqual(len(vg_d.get_image_ids(self.my_input)), len(self.my_expected_output))
    
    def test_outputIdentity(self):
        self.assertEqual(set(self.my_expected_output), set(vg_d.get_image_ids(self.my_input)))


class TestGetFileById(unittest.TestCase):
    def setUp(self):
        # Get the directory where my script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Make path to directory where image data are located:  
        self.my_data_dir = os.path.join(script_dir, '../..', 'test_visual_genome')
        self.input_id = 1
        self.file_extension = '.jpg'
        self.expected_output = 'visual_genome_1.jpg'
    
    def test_outputType(self):
        filename = vg_d.get_file_by_id(self.my_data_dir, self.input_id, self.file_extension)
        self.assertIsInstance(filename, list)
        self.assertIsInstance(filename[0], str)

    def test_outputContent(self):
        filename = vg_d.get_file_by_id(self.my_data_dir, self.input_id, self.file_extension)
        self.assertEqual(self.expected_output, filename[0])


if __name__ == "__main__":
    unittest.main()
