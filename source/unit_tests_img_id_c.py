import unittest
from . import image_id_converter as img_id_c

img_id_strs = ['111', '1', '32']
img_id_strs_output = ['id111', 'id001', 'id032']

#img_id_strs = ['0']
#img_id_strs_output = ['id000']

class TestCompleteImageIds(unittest.TestCase):
    def setUp(self):
        self.my_list_of_strs = img_id_strs
        self.my_expected_output = img_id_strs_output
    
    def test_inputExists(self):
        self.assertIsNotNone(self.my_list_of_strs)
    
    def test_inputType(self):
        self.assertIsInstance(self.my_list_of_strs[0], str)

    def test_functReturnsSomething(self):
        self.assertIsNotNone(img_id_c.complete_image_ids(self.my_list_of_strs))
    
    def test_outputType(self):
        self.assertEqual(type(img_id_c.complete_image_ids(self.my_list_of_strs)), list)    

    def test_lenIO(self):
        self.assertEqual(len(self.my_list_of_strs), len(img_id_c.complete_image_ids(self.my_list_of_strs)))
    
    def test_outputElementsType(self):
        self.assertEqual(type(img_id_c.complete_image_ids(self.my_list_of_strs)[0]), str)

    def test_outputCheck(self):
        self.assertEqual(img_id_c.complete_image_ids(self.my_list_of_strs), self.my_expected_output)        


img_id_strs_re = img_id_strs_output
img_id_strs_output_re = ['111', '001', '032']

#img_id_strs_re = ['id0', 'id00']
#img_id_strs_output_re = 0

img_id_c.reconvert_image_ids(img_id_strs_re)

class TestReconvertImageIds(unittest.TestCase):
    def setUp(self):
        self.my_list_of_strs = img_id_strs_re
        self.my_expected_output = img_id_strs_output_re
    
    def test_inputExists(self):
        self.assertIsNotNone(self.my_list_of_strs)
    
    def test_inputType(self):
        self.assertIsInstance(self.my_list_of_strs[0], str)

    def test_functReturnsSomething(self):
        self.assertIsNotNone(img_id_c.reconvert_image_ids(self.my_list_of_strs))

    def test_outputType(self):
        self.assertEqual(type(img_id_c.reconvert_image_ids(self.my_list_of_strs)), list) 

    def test_lenIO(self):
        self.assertEqual(len(self.my_list_of_strs), len(img_id_c.reconvert_image_ids(self.my_list_of_strs)))   
    
    def test_outputElementsType(self):
        self.assertEqual(type(img_id_c.reconvert_image_ids(self.my_list_of_strs)[0]), str)

    def test_outputCheck(self):
        self.assertEqual(img_id_c.reconvert_image_ids(self.my_list_of_strs), self.my_expected_output)     


if __name__ == "__main__":
    unittest.main()


