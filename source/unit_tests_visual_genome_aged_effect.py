import unittest
import os
from . import visual_genome_aged_effect as vg_ae 



class TestCopyWithNewId(unittest.TestCase):
    def setUp(self):
        # Get the directory where your script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Set paths to test data directories:
        self.source_dir = os.path.join(script_dir, '../..', 'test_old_photo_effect', 'new_id_source')
        self.dest_dir = os.path.join(script_dir, '../..', 'test_old_photo_effect', 'new_id_dest')
        self.filename_img = 'visual_genome_1.jpg'
        self.filename_txt = 'visual_genome_1.txt'

    def test_inputExists(self):
        self.assertIsNotNone(self.filename_img)
        self.assertIsNotNone(self.filename_txt)

    def test_sourceFile(self):
        self.assertTrue(self.filename_img in os.listdir(self.source_dir))
        self.assertTrue(self.filename_txt in os.listdir(self.source_dir))

    def test_BeforeAfterFunction(self):
        self.assertTrue(len(os.listdir(self.dest_dir))==0)
        vg_ae.copy_with_new_id(self.source_dir, self.dest_dir, self.filename_img, 9, '.jpg')
        vg_ae.copy_with_new_id(self.source_dir, self.dest_dir, self.filename_txt, 9, '.txt')
        self.assertTrue(len(os.listdir(self.dest_dir))==2)
        for file in os.listdir(self.dest_dir):
            dest_file_img = os.path.join(self.dest_dir, file)
            os.remove(dest_file_img)
        self.assertTrue(len(os.listdir(self.dest_dir))==0)

    def test_destFileName(self):
        vg_ae.copy_with_new_id(self.source_dir, self.dest_dir, self.filename_img, 9, '.jpg')
        vg_ae.copy_with_new_id(self.source_dir, self.dest_dir, self.filename_txt, 9, '.txt')
        self.assertTrue('visual_genome_1_9.jpg' in os.listdir(self.dest_dir))
        self.assertTrue('visual_genome_1_9.txt' in os.listdir(self.dest_dir))
        for file in os.listdir(self.dest_dir):
            dest_file_img = os.path.join(self.dest_dir, file)
            os.remove(dest_file_img)
        self.assertTrue(len(os.listdir(self.dest_dir))==0)


if __name__ == '__main__':
    unittest.main()

