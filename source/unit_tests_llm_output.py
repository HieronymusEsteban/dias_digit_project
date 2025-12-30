import unittest
#import shutil
#from PIL import Image
import matplotlib.pyplot as plt

import pandas as pd
from . import llm_input as llm_i
from . import llm_output as llm_o

#from pathlib import Path
import os

from unittest.mock import patch, MagicMock
import sys
from io import StringIO

class TestParseResponseToDict(unittest.TestCase):
    def setUp(self):
        self.structured_text = "{'image_is_photography': True, 'additional_comments': 'no comment'}"
        self.semi_structured_text = "OK, I don't like dictionaries but here is one: {'image_is_photography': False, 'additional_comments': ''}"
        self.unstructured_text = "hey, this is so funny! Hahaha!"

        self.expected_output_structured = (True, {'image_is_photography': True, 'additional_comments': 'no comment'})
        self.expected_output_semi_structured = (True, {'image_is_photography': False, 'additional_comments': ''})
        self.expected_output_unstructured = (False, None)

    def testOutputStructuredText(self):
        parsed_output = llm_o.parse_response_to_dict(self.structured_text)
        self.assertEqual(parsed_output, self.expected_output_structured)
    
    def testOutputSemiStructuredText(self):
        parsed_output = llm_o.parse_response_to_dict(self.semi_structured_text)
        self.assertEqual(parsed_output, self.expected_output_semi_structured)
    
    def testOutputUnstructuredText(self):
        parsed_output = llm_o.parse_response_to_dict(self.unstructured_text)
        self.assertEqual(parsed_output, self.expected_output_unstructured)


class TestAnalyzeImageStructured(unittest.TestCase):
    def setUp(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.llm_test_dir = os.path.join(script_dir, '../..', 'test_llm_input/data_llm_test')
        self.example_file = os.listdir(self.llm_test_dir)[0]
        self.example_path = os.path.join(self.llm_test_dir, self.example_file)
        self.expected_output_basic_length = 2
        self.expected_output_basic_keys = ['image_is_photography', 'additional_comments']
        self.expected_output_unstructured_length = 1
        self.expected_output_unstructured_keys = ['raw_response']

    def create_analysis_prompt_basic(self):
        """Create the structured prompt for image analysis."""
        return """
        Analyze this image and return ONLY a Python dictionary in exactly this format:
        
        {
            'image_is_photography': X,  # True if the image is a photography, False otherwise
            'additional_comments': '' # Any additional observations or empty string if none
        }
        
        Replace X with True (image is a photography) or False (image is not a photography).
        Return ONLY the dictionary, no other text.
        Your answer MUST have the exact structue of the dictionary described above (all keys MUST be present). 
        If you cannot answer the question in the way implied by this structure, enter 'None' as value and offer 
        your answer and explanations under 'additional_comments'.
        """

    def create_analysis_prompt_yes_no_basic(self):
        """Create the structured prompt for image analysis."""
        return """
        Is this image a photography or is it something else?

        Please, answwer yes if it is a photography, answer no if it is something else.
        
        """

    def testOutputBasicPromptLength(self):
        output = llm_o.analyze_image_structured(self.example_path, self.create_analysis_prompt_basic)
        self.assertEqual(len(output), self.expected_output_basic_length)

    def testOutputBasicPromptKeys(self):
        output = llm_o.analyze_image_structured(self.example_path, self.create_analysis_prompt_basic)
        self.assertEqual(list(output.keys()), self.expected_output_basic_keys)
    
    def testOutputUnstructuredPromptLength(self):
        output = llm_o.analyze_image_structured(self.example_path, self.create_analysis_prompt_yes_no_basic)
        self.assertEqual(len(output), self.expected_output_unstructured_length)
    
    def testOutputUnstructuredPromptKeys(self):
        output = llm_o.analyze_image_structured(self.example_path, self.create_analysis_prompt_yes_no_basic)
        self.assertEqual(list(output.keys()), self.expected_output_unstructured_keys)


class TestAddPredValues(unittest.TestCase):
    def setUp(self):
        self.test_df = pd.DataFrame({'image_id': [0, 1, 2, 3], 
        'col1': [1, None, 3, 2], 'col2': [4, None, 3, 1]})
    
    def test_InputExists(self):
        self.assertIsNotNone(self.test_df)

    def test_InputType(self):
        self.assertIsInstance(self.test_df, pd.DataFrame)
    
    def test_ExecuteFunction(self):
        row_1_df = self.test_df.loc[1, ['col1', 'col2']]
        self.assertTrue(row_1_df.isna().all())

        llm_o.add_pred_values(1, self.test_df, ['col1', 'col2'], [4, 5])

        self.assertFalse(self.test_df.isna().any().any())


# Mock the parse_response_to_dict function for testing
# def parse_response_to_dict(response_text):
#     """Mock implementation for testing purposes."""
#     if response_text == "valid dict text":
#         return True, {"image_is_photography": True}
#     elif response_text == "empty dict text":
#         return True, {}
#     elif response_text == "invalid dict text":
#         return False, None
#     elif response_text == "I am not sure if the image is a photography. I think {maybe yes}":
#         return False, None
#     else:
#         return False, None


class TestConvertBooleanToEncoding(unittest.TestCase):
    """Test cases for convert_boolean_to_encoding function."""
    
    def test_boolean_true(self):
        """Test with boolean True."""
        result = llm_o.convert_boolean_to_encoding(True)
        self.assertEqual(result, 1)
    
    def test_boolean_false(self):
        """Test with boolean False."""
        result = llm_o.convert_boolean_to_encoding(False)
        self.assertEqual(result, 0)
    
    def test_string_true_variations(self):
        """Test with string representations of True."""
        test_cases = ['true', 'True', 'TRUE', '1', 'yes', 'Yes', 'YES']
        for test_case in test_cases:
            with self.subTest(test_case=test_case):
                result = llm_o.convert_boolean_to_encoding(test_case)
                self.assertEqual(result, 1)
    
    def test_string_false_variations(self):
        """Test with string representations of False."""
        test_cases = ['false', 'False', 'FALSE', '0', 'no', 'No', 'NO']
        for test_case in test_cases:
            with self.subTest(test_case=test_case):
                result = llm_o.convert_boolean_to_encoding(test_case)
                self.assertEqual(result, 0)
    
    def test_numeric_values(self):
        """Test with numeric values."""
        # Truthy numbers
        self.assertEqual(llm_o.convert_boolean_to_encoding(1), 1)
        self.assertEqual(llm_o.convert_boolean_to_encoding(42), 1)
        self.assertEqual(llm_o.convert_boolean_to_encoding(3.14), 1)
        
        # Falsy numbers
        self.assertEqual(llm_o.convert_boolean_to_encoding(0), 0)
        self.assertEqual(llm_o.convert_boolean_to_encoding(0.0), 0)
    
    def test_invalid_inputs(self):
        """Test with invalid inputs that should return None."""
        invalid_inputs = ['maybe', 'invalid', None, [], {}, object()]
        for invalid_input in invalid_inputs:
            with self.subTest(invalid_input=invalid_input):
                result = llm_o.convert_boolean_to_encoding(invalid_input)
                self.assertIsNone(result)


class TestExtractFromStructuredDict(unittest.TestCase):
    """Test cases for extract_from_structured_dict function."""
    
    def test_valid_structured_dict_true(self):
        """Test with valid structured dictionary containing True."""
        img_pred = {'image_is_photography': True, 'additional_comments': ''}
        keys_expected = ['image_is_photography', 'additional_comments']
        response_variable = 'image_is_photography'
        
        result = llm_o.extract_from_structured_dict(img_pred, keys_expected, response_variable)
        self.assertEqual(result, 1)
    
    def test_valid_structured_dict_false(self):
        """Test with valid structured dictionary containing False."""
        img_pred = {'image_is_photography': False, 'additional_comments': 'painting'}
        keys_expected = ['image_is_photography', 'additional_comments']
        response_variable = 'image_is_photography'
        
        result = llm_o.extract_from_structured_dict(img_pred, keys_expected, response_variable)
        self.assertEqual(result, 0)
    
    def test_wrong_keys(self):
        """Test with dictionary having wrong keys."""
        img_pred = {'wrong_key': True, 'another_wrong_key': ''}
        keys_expected = ['image_is_photography', 'additional_comments']
        response_variable = 'image_is_photography'
        
        result = llm_o.extract_from_structured_dict(img_pred, keys_expected, response_variable)
        self.assertIsNone(result)
    
    def test_missing_response_variable(self):
        """Test with dictionary missing the response variable."""
        img_pred = {'additional_comments': 'some comment'}
        keys_expected = ['image_is_photography', 'additional_comments']
        response_variable = 'image_is_photography'
        
        result = llm_o.extract_from_structured_dict(img_pred, keys_expected, response_variable)
        self.assertIsNone(result)
    
    def test_non_dict_input(self):
        """Test with non-dictionary input."""
        non_dict_inputs = ['string', 123, [], set(), None]
        keys_expected = ['image_is_photography', 'additional_comments']
        response_variable = 'image_is_photography'
        
        for non_dict_input in non_dict_inputs:
            with self.subTest(non_dict_input=non_dict_input):
                result = llm_o.extract_from_structured_dict(non_dict_input, keys_expected, response_variable)
                self.assertIsNone(result)
    
    def test_extra_keys(self):
        """Test with dictionary having extra keys."""
        img_pred = {
            'image_is_photography': True, 
            'additional_comments': '', 
            'extra_key': 'extra_value'
        }
        keys_expected = ['image_is_photography', 'additional_comments']
        response_variable = 'image_is_photography'
        
        result = llm_o.extract_from_structured_dict(img_pred, keys_expected, response_variable)
        self.assertIsNone(result)


class TestExtractFromRawResponse(unittest.TestCase):
    """Test cases for extract_from_raw_response function."""

    def test_short_response(self):
        """Test with raw_response that cannot be parsed."""
        img_pred = {'raw_response': "{'image_is_photography': True}"}
        response_variable = 'image_is_photography'
        
        result = llm_o.extract_from_raw_response(img_pred, response_variable)
        self.assertEqual(result, 1)
    
    def test_unparseable_raw_response(self):
        """Test with raw_response that cannot be parsed."""
        img_pred = {'raw_response': 'invalid dict text'}
        response_variable = 'image_is_photography'
        
        result = llm_o.extract_from_raw_response(img_pred, response_variable)
        self.assertIsNone(result)
    
    def test_no_raw_response_key(self):
        """Test with dictionary without raw_response key."""
        img_pred = {'other_key': True}
        response_variable = 'image_is_photography'
        
        result = llm_o.extract_from_raw_response(img_pred, response_variable)
        self.assertIsNone(result)
    
    def test_non_dict_input(self):
        """Test with non-dictionary input."""
        non_dict_inputs = ['string', 123, [], set(), None, 'Yes, this is True.', 'True']
        response_variable = 'image_is_photography'
        
        for non_dict_input in non_dict_inputs:
            with self.subTest(non_dict_input=non_dict_input):
                result = llm_o.extract_from_raw_response(non_dict_input, response_variable)
                self.assertIsNone(result)
    
    def test_non_string_raw_response(self):
        """Test with non-string raw_response value."""
        img_pred = {'raw_response': 123}
        response_variable = 'image_is_photography'
        
        result = llm_o.extract_from_raw_response(img_pred, response_variable)
        self.assertIsNone(result)


class TestProcessSingleEntry(unittest.TestCase):
    """Test cases for process_single_entry function."""
    
    def setUp(self):
        """Redirect stdout to capture print statements."""
        self.held, sys.stdout = sys.stdout, StringIO()
    
    def tearDown(self):
        """Restore stdout."""
        sys.stdout = self.held
    
    def test_successful_structured_extraction(self):
        """Test successful extraction from structured dictionary."""
        img_id = 'test_001'
        img_pred = {'image_is_photography': True, 'additional_comments': ''}
        keys_expected = ['image_is_photography', 'additional_comments']
        response_variable = 'image_is_photography'
        
        encoded_value, needs_inspection = llm_o.process_single_entry(
            img_id, img_pred, keys_expected, response_variable
        )
        
        self.assertEqual(encoded_value, 1)
        self.assertFalse(needs_inspection)
        
        # Check print output
        output = sys.stdout.getvalue()
        self.assertIn('test_001', output)
        self.assertIn('both conditions true', output)
    
    def test_successful_raw_response_extraction(self):
        """Test successful extraction from raw_response."""
        img_id = 'test_002'
        img_pred = {'raw_response': 'invalid dict text'}
        keys_expected = ['image_is_photography', 'additional_comments']
        response_variable = 'image_is_photography'
        
        encoded_value, needs_inspection = llm_o.process_single_entry(
            img_id, img_pred, keys_expected, response_variable
        )
        
        self.assertIsNone(encoded_value)
        self.assertTrue(needs_inspection)
        
        # Check print output
        output = sys.stdout.getvalue()
        self.assertIn('test_002', output)
    
    def test_failed_extraction(self):
        """Test failed extraction requiring inspection."""
        img_id = 'test_003'
        img_pred = 'invalid string input'
        keys_expected = ['image_is_photography', 'additional_comments']
        response_variable = 'image_is_photography'
        
        encoded_value, needs_inspection = llm_o.process_single_entry(
            img_id, img_pred, keys_expected, response_variable
        )
        
        self.assertIsNone(encoded_value)
        self.assertTrue(needs_inspection)
        
        # Check print output
        output = sys.stdout.getvalue()
        self.assertIn('test_003', output)
        self.assertIn('no structure at all:', output)


class TestExtractValsFromResponseDict(unittest.TestCase):
    """Test cases for the main extract_vals_from_response_dict function."""
    
    def setUp(self):
        """Redirect stdout to capture print statements."""
        self.held, sys.stdout = sys.stdout, StringIO()
    
    def tearDown(self):
        """Restore stdout."""
        sys.stdout = self.held
    
    def test_valid_input_mixed_success(self):
        """Test with mixed successful and failed extractions."""
        img_ids = ['001', '002', '003']
        image_descr = {
            '001': {'image_is_photography': True, 'additional_comments': ''},
            '002': {'raw_response': 'invalid dict text'},
            '003': 'invalid string input'
        }
        keys_expected = ['image_is_photography', 'additional_comments']
        response_variable = 'image_is_photography'
        
        result_ids, response_values, inspection_ids = llm_o.extract_vals_from_response_dict(
            img_ids, image_descr, keys_expected, response_variable
        )
        
        self.assertEqual(result_ids, ['001', '002', '003'])
        self.assertEqual(response_values, [1, None, None])
        self.assertEqual(inspection_ids, ['002', '003'])
    
    def test_empty_img_ids_list(self):
        """Test with empty img_ids list."""
        with self.assertRaises(ValueError):
            llm_o.extract_vals_from_response_dict([], {}, [], 'test_var')
    
    def test_non_list_img_ids(self):
        """Test with non-list img_ids."""
        with self.assertRaises(ValueError):
            llm_o.extract_vals_from_response_dict('not_a_list', {}, [], 'test_var')
    
    def test_non_dict_image_descr(self):
        """Test with non-dictionary image_descr."""
        with self.assertRaises(ValueError):
            llm_o.extract_vals_from_response_dict(['001'], 'not_a_dict', [], 'test_var')
    
    def test_missing_img_id(self):
        """Test with img_id not in image_descr."""
        img_ids = ['001', '002']
        image_descr = {'001': {'image_is_photography': True, 'additional_comments': ''}}
        keys_expected = ['image_is_photography', 'additional_comments']
        response_variable = 'image_is_photography'
        
        with self.assertRaises(KeyError):
            llm_o.extract_vals_from_response_dict(img_ids, image_descr, keys_expected, response_variable)
    
    def test_all_successful_extractions(self):
        """Test with all successful extractions."""
        img_ids = ['001', '002']
        image_descr = {
            '001': {'image_is_photography': True, 'additional_comments': ''},
            '002': {'image_is_photography': False, 'additional_comments': 'painting'}
        }
        keys_expected = ['image_is_photography', 'additional_comments']
        response_variable = 'image_is_photography'
        
        result_ids, response_values, inspection_ids = llm_o.extract_vals_from_response_dict(
            img_ids, image_descr, keys_expected, response_variable
        )
        
        self.assertEqual(result_ids, ['001', '002'])
        self.assertEqual(response_values, [1, 0])
        self.assertEqual(inspection_ids, [])
    
    def test_all_failed_extractions(self):
        """Test with all failed extractions."""
        img_ids = ['001', '002']
        image_descr = {
            '001': 'invalid string',
            '002': {'invalid': 'structure'}
        }
        keys_expected = ['image_is_photography', 'additional_comments']
        response_variable = 'image_is_photography'
        
        result_ids, response_values, inspection_ids = llm_o.extract_vals_from_response_dict(
            img_ids, image_descr, keys_expected, response_variable
        )
        
        self.assertEqual(result_ids, ['001', '002'])
        self.assertEqual(response_values, [None, None])
        self.assertEqual(inspection_ids, ['001', '002'])


class TestGetClassificationSubsetsMetrics(unittest.TestCase):
    """Test case for the get_classification_subsets_metrics function."""

    def setUp(self):
        self.label_name = 'is_photo'
        self.prediction_name = 'is_photo_pred'

    def test_balanced_data_balanced_pred(self):

        print('test_balanced_data_balanced_pred')

        positives = 4
        negatives = 4
        true_positives = 2
        false_positives = 2
        true_negatives = 2
        false_negatives = 2

        sensitivity = true_positives / positives
        specificity = true_negatives / negatives

        image_ids = [f'{i:03d}' for i in range(1, 9)]
        with_person = [1] * 6 + [0] * 2
        is_photo = [1] * positives + [0] * negatives
        is_photo_pred = [1] * true_positives + [0] * false_negatives + [1] * false_positives + [0] * true_negatives
        labels_results = pd.DataFrame({'image_id': image_ids, 'with_person': with_person, 
                 'is_photo': is_photo, 'is_photo_pred': is_photo_pred})

        subsets_and_metrics = llm_o.get_classification_subsets_metrics(labels_results, self.label_name , self.prediction_name)
        positives_comp, negatives_comp, true_positives_comp, true_negatives_comp, \
        false_negatives_comp, false_positives_comp, sensitivity_comp, specificity_comp = subsets_and_metrics

        pos_bools = labels_results.is_photo == 1
        neg_bools = labels_results.is_photo == 0
        pred_pos_bools = labels_results.is_photo_pred == 1
        pred_neg_bools = labels_results.is_photo_pred == 0

        self.assertEqual(labels_results.shape[0], positives + negatives)
        self.assertEqual(labels_results[pos_bools & pred_pos_bools].shape[0], true_positives)
        self.assertEqual(labels_results[pos_bools & pred_neg_bools].shape[0], false_negatives)
        self.assertEqual(labels_results[neg_bools & pred_neg_bools].shape[0], true_negatives)
        self.assertEqual(labels_results[neg_bools & pred_pos_bools].shape[0], false_positives)
        self.assertEqual(sensitivity, sensitivity_comp)
        self.assertEqual(specificity, specificity_comp)


    def test_balanced_data_neg_pred(self):
        print('test_balanced_data_neg_pred')

        positives = 4
        negatives = 4
        true_positives = 0
        false_positives = 0
        true_negatives = 4
        false_negatives = 4

        sensitivity = true_positives / positives
        specificity = true_negatives / negatives

        image_ids = [f'{i:03d}' for i in range(1, 9)]
        with_person = [1] * 6 + [0] * 2
        is_photo = [1] * positives + [0] * negatives
        is_photo_pred = [1] * true_positives + [0] * false_negatives + [1] * false_positives + [0] * true_negatives
        labels_results = pd.DataFrame({'image_id': image_ids, 'with_person': with_person, 
                 'is_photo': is_photo, 'is_photo_pred': is_photo_pred})

        subsets_and_metrics = llm_o.get_classification_subsets_metrics(labels_results, self.label_name , self.prediction_name)
        positives_comp, negatives_comp, true_positives_comp, true_negatives_comp, \
        false_negatives_comp, false_positives_comp, sensitivity_comp, specificity_comp = subsets_and_metrics

        pos_bools = labels_results.is_photo == 1
        neg_bools = labels_results.is_photo == 0
        pred_pos_bools = labels_results.is_photo_pred == 1
        pred_neg_bools = labels_results.is_photo_pred == 0

        self.assertEqual(labels_results.shape[0], positives + negatives)
        self.assertEqual(labels_results[pos_bools & pred_pos_bools].shape[0], true_positives)
        self.assertEqual(labels_results[pos_bools & pred_neg_bools].shape[0], false_negatives)
        self.assertEqual(labels_results[neg_bools & pred_neg_bools].shape[0], true_negatives)
        self.assertEqual(labels_results[neg_bools & pred_pos_bools].shape[0], false_positives)
        self.assertEqual(sensitivity, sensitivity_comp)
        self.assertEqual(specificity, specificity_comp)

    def test_balanced_data_pos_pred(self):
        print('test_balanced_data_pos_pred')

        positives = 4
        negatives = 4
        true_positives = 4
        false_positives = 4
        true_negatives = 0
        false_negatives = 0

        sensitivity = true_positives / positives
        specificity = true_negatives / negatives

        image_ids = [f'{i:03d}' for i in range(1, 9)]
        with_person = [1] * 6 + [0] * 2
        is_photo = [1] * positives + [0] * negatives
        is_photo_pred = [1] * true_positives + [0] * false_negatives + [1] * false_positives + [0] * true_negatives
        labels_results = pd.DataFrame({'image_id': image_ids, 'with_person': with_person, 
                 'is_photo': is_photo, 'is_photo_pred': is_photo_pred})

        subsets_and_metrics = llm_o.get_classification_subsets_metrics(labels_results, self.label_name , self.prediction_name)
        positives_comp, negatives_comp, true_positives_comp, true_negatives_comp, \
        false_negatives_comp, false_positives_comp, sensitivity_comp, specificity_comp = subsets_and_metrics

        pos_bools = labels_results.is_photo == 1
        neg_bools = labels_results.is_photo == 0
        pred_pos_bools = labels_results.is_photo_pred == 1
        pred_neg_bools = labels_results.is_photo_pred == 0

        self.assertEqual(labels_results.shape[0], positives + negatives)
        self.assertEqual(labels_results[pos_bools & pred_pos_bools].shape[0], true_positives)
        self.assertEqual(labels_results[pos_bools & pred_neg_bools].shape[0], false_negatives)
        self.assertEqual(labels_results[neg_bools & pred_neg_bools].shape[0], true_negatives)
        self.assertEqual(labels_results[neg_bools & pred_pos_bools].shape[0], false_positives)
        self.assertEqual(sensitivity, sensitivity_comp)
        self.assertEqual(specificity, specificity_comp)

    def test_balanced_data_correct_pred(self):
        print('test_balanced_data_correct_pred')

        positives = 4
        negatives = 4
        true_positives = 4
        false_positives = 0
        true_negatives = 4
        false_negatives = 0

        sensitivity = true_positives / positives
        specificity = true_negatives / negatives

        image_ids = [f'{i:03d}' for i in range(1, 9)]
        with_person = [1] * 6 + [0] * 2
        is_photo = [1] * positives + [0] * negatives
        is_photo_pred = [1] * true_positives + [0] * false_negatives + [1] * false_positives + [0] * true_negatives
        labels_results = pd.DataFrame({'image_id': image_ids, 'with_person': with_person, 
                 'is_photo': is_photo, 'is_photo_pred': is_photo_pred})

        subsets_and_metrics = llm_o.get_classification_subsets_metrics(labels_results, self.label_name , self.prediction_name)
        positives_comp, negatives_comp, true_positives_comp, true_negatives_comp, \
        false_negatives_comp, false_positives_comp, sensitivity_comp, specificity_comp = subsets_and_metrics

        pos_bools = labels_results.is_photo == 1
        neg_bools = labels_results.is_photo == 0
        pred_pos_bools = labels_results.is_photo_pred == 1
        pred_neg_bools = labels_results.is_photo_pred == 0

        self.assertEqual(labels_results.shape[0], positives + negatives)
        self.assertEqual(labels_results[pos_bools & pred_pos_bools].shape[0], true_positives)
        self.assertEqual(labels_results[pos_bools & pred_neg_bools].shape[0], false_negatives)
        self.assertEqual(labels_results[neg_bools & pred_neg_bools].shape[0], true_negatives)
        self.assertEqual(labels_results[neg_bools & pred_pos_bools].shape[0], false_positives)
        self.assertEqual(sensitivity, sensitivity_comp)
        self.assertEqual(specificity, specificity_comp)

    def test_squewed_data_correct_pred(self):
        print('test_squewed_data_correct_pred')

        positives = 5
        negatives = 3
        true_positives = 5
        false_positives = 0
        true_negatives = 3
        false_negatives = 0

        sensitivity = true_positives / positives
        specificity = true_negatives / negatives

        image_ids = [f'{i:03d}' for i in range(1, 9)]
        with_person = [1] * 6 + [0] * 2
        is_photo = [1] * positives + [0] * negatives
        is_photo_pred = [1] * true_positives + [0] * false_negatives + [1] * false_positives + [0] * true_negatives
        labels_results = pd.DataFrame({'image_id': image_ids, 'with_person': with_person, 
                 'is_photo': is_photo, 'is_photo_pred': is_photo_pred})

        subsets_and_metrics = llm_o.get_classification_subsets_metrics(labels_results, self.label_name , self.prediction_name)
        positives_comp, negatives_comp, true_positives_comp, true_negatives_comp, \
        false_negatives_comp, false_positives_comp, sensitivity_comp, specificity_comp = subsets_and_metrics

        pos_bools = labels_results.is_photo == 1
        neg_bools = labels_results.is_photo == 0
        pred_pos_bools = labels_results.is_photo_pred == 1
        pred_neg_bools = labels_results.is_photo_pred == 0

        self.assertEqual(labels_results.shape[0], positives + negatives)
        self.assertEqual(labels_results[pos_bools & pred_pos_bools].shape[0], true_positives)
        self.assertEqual(labels_results[pos_bools & pred_neg_bools].shape[0], false_negatives)
        self.assertEqual(labels_results[neg_bools & pred_neg_bools].shape[0], true_negatives)
        self.assertEqual(labels_results[neg_bools & pred_pos_bools].shape[0], false_positives)
        self.assertEqual(sensitivity, sensitivity_comp)
        self.assertEqual(specificity, specificity_comp)
    
    def test_squewed_data_neg_pred(self):
        print('test_squewed_data_neg_pred')

        positives = 5
        negatives = 3
        true_positives = 0
        false_positives = 0
        true_negatives = 3
        false_negatives = 5

        sensitivity = true_positives / positives
        specificity = true_negatives / negatives

        image_ids = [f'{i:03d}' for i in range(1, 9)]
        with_person = [1] * 6 + [0] * 2
        is_photo = [1] * positives + [0] * negatives
        is_photo_pred = [1] * true_positives + [0] * false_negatives + [1] * false_positives + [0] * true_negatives
        labels_results = pd.DataFrame({'image_id': image_ids, 'with_person': with_person, 
                 'is_photo': is_photo, 'is_photo_pred': is_photo_pred})

        subsets_and_metrics = llm_o.get_classification_subsets_metrics(labels_results, self.label_name , self.prediction_name)
        positives_comp, negatives_comp, true_positives_comp, true_negatives_comp, \
        false_negatives_comp, false_positives_comp, sensitivity_comp, specificity_comp = subsets_and_metrics

        pos_bools = labels_results.is_photo == 1
        neg_bools = labels_results.is_photo == 0
        pred_pos_bools = labels_results.is_photo_pred == 1
        pred_neg_bools = labels_results.is_photo_pred == 0

        self.assertEqual(labels_results.shape[0], positives + negatives)
        self.assertEqual(labels_results[pos_bools & pred_pos_bools].shape[0], true_positives)
        self.assertEqual(labels_results[pos_bools & pred_neg_bools].shape[0], false_negatives)
        self.assertEqual(labels_results[neg_bools & pred_neg_bools].shape[0], true_negatives)
        self.assertEqual(labels_results[neg_bools & pred_pos_bools].shape[0], false_positives)
        self.assertEqual(sensitivity, sensitivity_comp)
        self.assertEqual(specificity, specificity_comp)

    def test_squewed_data_partial_pred(self):
        print('test_squewed_data_partial_pred')

        positives = 5
        negatives = 3
        true_positives = 4
        false_positives = 0
        true_negatives = 3
        false_negatives = 1

        sensitivity = true_positives / positives
        specificity = true_negatives / negatives

        image_ids = [f'{i:03d}' for i in range(1, 9)]
        with_person = [1] * 6 + [0] * 2
        is_photo = [1] * positives + [0] * negatives
        is_photo_pred = [1] * true_positives + [0] * false_negatives + [1] * false_positives + [0] * true_negatives
        labels_results = pd.DataFrame({'image_id': image_ids, 'with_person': with_person, 
                 'is_photo': is_photo, 'is_photo_pred': is_photo_pred})

        subsets_and_metrics = llm_o.get_classification_subsets_metrics(labels_results, self.label_name , self.prediction_name)
        positives_comp, negatives_comp, true_positives_comp, true_negatives_comp, \
        false_negatives_comp, false_positives_comp, sensitivity_comp, specificity_comp = subsets_and_metrics

        pos_bools = labels_results.is_photo == 1
        neg_bools = labels_results.is_photo == 0
        pred_pos_bools = labels_results.is_photo_pred == 1
        pred_neg_bools = labels_results.is_photo_pred == 0

        self.assertEqual(labels_results.shape[0], positives + negatives)
        self.assertEqual(labels_results[pos_bools & pred_pos_bools].shape[0], true_positives)
        self.assertEqual(labels_results[pos_bools & pred_neg_bools].shape[0], false_negatives)
        self.assertEqual(labels_results[neg_bools & pred_neg_bools].shape[0], true_negatives)
        self.assertEqual(labels_results[neg_bools & pred_pos_bools].shape[0], false_positives)
        self.assertEqual(sensitivity, sensitivity_comp)
        self.assertEqual(specificity, specificity_comp)

    def test_balanced_data_balanced_pred_large(self):

        print('test_balanced_data_balanced_pred_large')

        positives = 400
        negatives = 400
        true_positives = 200
        false_positives = 200
        true_negatives = 200
        false_negatives = 200

        sensitivity = true_positives / positives
        specificity = true_negatives / negatives

        image_ids = [f'{i:03d}' for i in range(1, 801)]
        with_person = [1] * 600 + [0] * 200
        is_photo = [1] * positives + [0] * negatives
        is_photo_pred = [1] * true_positives + [0] * false_negatives + [1] * false_positives + [0] * true_negatives
        labels_results = pd.DataFrame({'image_id': image_ids, 'with_person': with_person, 
                 'is_photo': is_photo, 'is_photo_pred': is_photo_pred})

        subsets_and_metrics = llm_o.get_classification_subsets_metrics(labels_results, self.label_name , self.prediction_name)
        positives_comp, negatives_comp, true_positives_comp, true_negatives_comp, \
        false_negatives_comp, false_positives_comp, sensitivity_comp, specificity_comp = subsets_and_metrics

        pos_bools = labels_results.is_photo == 1
        neg_bools = labels_results.is_photo == 0
        pred_pos_bools = labels_results.is_photo_pred == 1
        pred_neg_bools = labels_results.is_photo_pred == 0

        self.assertEqual(labels_results.shape[0], positives + negatives)
        self.assertEqual(labels_results[pos_bools & pred_pos_bools].shape[0], true_positives)
        self.assertEqual(labels_results[pos_bools & pred_neg_bools].shape[0], false_negatives)
        self.assertEqual(labels_results[neg_bools & pred_neg_bools].shape[0], true_negatives)
        self.assertEqual(labels_results[neg_bools & pred_pos_bools].shape[0], false_positives)
        self.assertEqual(sensitivity, sensitivity_comp)
        self.assertEqual(specificity, specificity_comp)


class TestPlotConfMatrix(unittest.TestCase):
    """Test case for the plot_conf_matrix function."""

    def setUp(self):
        self.label_name = 'is_photo'
        self.prediction_name = 'is_photo_pred'

    def test_balanced_data_balanced_pred(self):

        print('test_balanced_data_balanced_pred')

        positives = 4
        negatives = 4
        true_positives = 2
        false_positives = 2
        true_negatives = 2
        false_negatives = 2

        sensitivity = true_positives / positives
        specificity = true_negatives / negatives

        image_ids = [f'{i:03d}' for i in range(1, 9)]
        with_person = [1] * 6 + [0] * 2
        is_photo = [1] * positives + [0] * negatives
        is_photo_pred = [1] * true_positives + [0] * false_negatives + [1] * false_positives + [0] * true_negatives
        labels_results = pd.DataFrame({'image_id': image_ids, 'with_person': with_person, 
                 'is_photo': is_photo, 'is_photo_pred': is_photo_pred})

        pos_bools = labels_results.is_photo == 1
        neg_bools = labels_results.is_photo == 0
        pred_pos_bools = labels_results.is_photo_pred == 1
        pred_neg_bools = labels_results.is_photo_pred == 0

        positives = labels_results[pos_bools]
        negatives = labels_results[neg_bools]
        true_positives = labels_results[pos_bools & pred_pos_bools]
        false_positives = labels_results[neg_bools & pred_pos_bools]
        true_negatives = labels_results[neg_bools & pred_neg_bools]
        false_negatives = labels_results[pos_bools & pred_neg_bools]

        cases = (true_positives, false_positives, true_negatives, false_negatives, positives, negatives)
        fig = llm_o.plot_conf_matrix(labels_results, self.label_name, self.prediction_name, cases)

        plt.show()


    def test_balanced_data_neg_pred(self):
        print('test_balanced_data_neg_pred')

        positives = 4
        negatives = 4
        true_positives = 0
        false_positives = 0
        true_negatives = 4
        false_negatives = 4

        sensitivity = true_positives / positives
        specificity = true_negatives / negatives

        image_ids = [f'{i:03d}' for i in range(1, 9)]
        with_person = [1] * 6 + [0] * 2
        is_photo = [1] * positives + [0] * negatives
        is_photo_pred = [1] * true_positives + [0] * false_negatives + [1] * false_positives + [0] * true_negatives
        labels_results = pd.DataFrame({'image_id': image_ids, 'with_person': with_person, 
                 'is_photo': is_photo, 'is_photo_pred': is_photo_pred})

        pos_bools = labels_results.is_photo == 1
        neg_bools = labels_results.is_photo == 0
        pred_pos_bools = labels_results.is_photo_pred == 1
        pred_neg_bools = labels_results.is_photo_pred == 0

        positives = labels_results[pos_bools]
        negatives = labels_results[neg_bools]
        true_positives = labels_results[pos_bools & pred_pos_bools]
        false_positives = labels_results[neg_bools & pred_pos_bools]
        true_negatives = labels_results[neg_bools & pred_neg_bools]
        false_negatives = labels_results[pos_bools & pred_neg_bools]

        cases = (true_positives, false_positives, true_negatives, false_negatives, positives, negatives)
        fig = llm_o.plot_conf_matrix(labels_results, self.label_name, self.prediction_name, cases)


    def test_balanced_data_pos_pred(self):
        print('test_balanced_data_pos_pred')

        positives = 4
        negatives = 4
        true_positives = 4
        false_positives = 4
        true_negatives = 0
        false_negatives = 0

        sensitivity = true_positives / positives
        specificity = true_negatives / negatives

        image_ids = [f'{i:03d}' for i in range(1, 9)]
        with_person = [1] * 6 + [0] * 2
        is_photo = [1] * positives + [0] * negatives
        is_photo_pred = [1] * true_positives + [0] * false_negatives + [1] * false_positives + [0] * true_negatives
        labels_results = pd.DataFrame({'image_id': image_ids, 'with_person': with_person, 
                 'is_photo': is_photo, 'is_photo_pred': is_photo_pred})

        pos_bools = labels_results.is_photo == 1
        neg_bools = labels_results.is_photo == 0
        pred_pos_bools = labels_results.is_photo_pred == 1
        pred_neg_bools = labels_results.is_photo_pred == 0

        positives = labels_results[pos_bools]
        negatives = labels_results[neg_bools]
        true_positives = labels_results[pos_bools & pred_pos_bools]
        false_positives = labels_results[neg_bools & pred_pos_bools]
        true_negatives = labels_results[neg_bools & pred_neg_bools]
        false_negatives = labels_results[pos_bools & pred_neg_bools]

        cases = (true_positives, false_positives, true_negatives, false_negatives, positives, negatives)
        fig = llm_o.plot_conf_matrix(labels_results, self.label_name, self.prediction_name, cases)


    def test_balanced_data_correct_pred(self):
        print('test_balanced_data_correct_pred')

        positives = 4
        negatives = 4
        true_positives = 4
        false_positives = 0
        true_negatives = 4
        false_negatives = 0

        sensitivity = true_positives / positives
        specificity = true_negatives / negatives

        image_ids = [f'{i:03d}' for i in range(1, 9)]
        with_person = [1] * 6 + [0] * 2
        is_photo = [1] * positives + [0] * negatives
        is_photo_pred = [1] * true_positives + [0] * false_negatives + [1] * false_positives + [0] * true_negatives
        labels_results = pd.DataFrame({'image_id': image_ids, 'with_person': with_person, 
                 'is_photo': is_photo, 'is_photo_pred': is_photo_pred})

        pos_bools = labels_results.is_photo == 1
        neg_bools = labels_results.is_photo == 0
        pred_pos_bools = labels_results.is_photo_pred == 1
        pred_neg_bools = labels_results.is_photo_pred == 0

        positives = labels_results[pos_bools]
        negatives = labels_results[neg_bools]
        true_positives = labels_results[pos_bools & pred_pos_bools]
        false_positives = labels_results[neg_bools & pred_pos_bools]
        true_negatives = labels_results[neg_bools & pred_neg_bools]
        false_negatives = labels_results[pos_bools & pred_neg_bools]

        cases = (true_positives, false_positives, true_negatives, false_negatives, positives, negatives)
        fig = llm_o.plot_conf_matrix(labels_results, self.label_name, self.prediction_name, cases)


    def test_squewed_data_correct_pred(self):
        print('test_squewed_data_correct_pred')

        positives = 5
        negatives = 3
        true_positives = 5
        false_positives = 0
        true_negatives = 3
        false_negatives = 0

        sensitivity = true_positives / positives
        specificity = true_negatives / negatives

        image_ids = [f'{i:03d}' for i in range(1, 9)]
        with_person = [1] * 6 + [0] * 2
        is_photo = [1] * positives + [0] * negatives
        is_photo_pred = [1] * true_positives + [0] * false_negatives + [1] * false_positives + [0] * true_negatives
        labels_results = pd.DataFrame({'image_id': image_ids, 'with_person': with_person, 
                 'is_photo': is_photo, 'is_photo_pred': is_photo_pred})

        pos_bools = labels_results.is_photo == 1
        neg_bools = labels_results.is_photo == 0
        pred_pos_bools = labels_results.is_photo_pred == 1
        pred_neg_bools = labels_results.is_photo_pred == 0

        positives = labels_results[pos_bools]
        negatives = labels_results[neg_bools]
        true_positives = labels_results[pos_bools & pred_pos_bools]
        false_positives = labels_results[neg_bools & pred_pos_bools]
        true_negatives = labels_results[neg_bools & pred_neg_bools]
        false_negatives = labels_results[pos_bools & pred_neg_bools]

        cases = (true_positives, false_positives, true_negatives, false_negatives, positives, negatives)
        fig = llm_o.plot_conf_matrix(labels_results, self.label_name, self.prediction_name, cases)

    
    def test_squewed_data_neg_pred(self):
        print('test_squewed_data_neg_pred')

        positives = 5
        negatives = 3
        true_positives = 0
        false_positives = 0
        true_negatives = 3
        false_negatives = 5

        sensitivity = true_positives / positives
        specificity = true_negatives / negatives

        image_ids = [f'{i:03d}' for i in range(1, 9)]
        with_person = [1] * 6 + [0] * 2
        is_photo = [1] * positives + [0] * negatives
        is_photo_pred = [1] * true_positives + [0] * false_negatives + [1] * false_positives + [0] * true_negatives
        labels_results = pd.DataFrame({'image_id': image_ids, 'with_person': with_person, 
                 'is_photo': is_photo, 'is_photo_pred': is_photo_pred})

        pos_bools = labels_results.is_photo == 1
        neg_bools = labels_results.is_photo == 0
        pred_pos_bools = labels_results.is_photo_pred == 1
        pred_neg_bools = labels_results.is_photo_pred == 0

        positives = labels_results[pos_bools]
        negatives = labels_results[neg_bools]
        true_positives = labels_results[pos_bools & pred_pos_bools]
        false_positives = labels_results[neg_bools & pred_pos_bools]
        true_negatives = labels_results[neg_bools & pred_neg_bools]
        false_negatives = labels_results[pos_bools & pred_neg_bools]

        cases = (true_positives, false_positives, true_negatives, false_negatives, positives, negatives)
        fig = llm_o.plot_conf_matrix(labels_results, self.label_name, self.prediction_name, cases)


    def test_squewed_data_partial_pred(self):
        print('test_squewed_data_partial_pred')

        positives = 5
        negatives = 3
        true_positives = 4
        false_positives = 0
        true_negatives = 3
        false_negatives = 1

        sensitivity = true_positives / positives
        specificity = true_negatives / negatives

        image_ids = [f'{i:03d}' for i in range(1, 9)]
        with_person = [1] * 6 + [0] * 2
        is_photo = [1] * positives + [0] * negatives
        is_photo_pred = [1] * true_positives + [0] * false_negatives + [1] * false_positives + [0] * true_negatives
        labels_results = pd.DataFrame({'image_id': image_ids, 'with_person': with_person, 
                 'is_photo': is_photo, 'is_photo_pred': is_photo_pred})

        subsets_and_metrics = llm_o.get_classification_subsets_metrics(labels_results, self.label_name , self.prediction_name)
        positives_comp, negatives_comp, true_positives_comp, true_negatives_comp, \
        false_negatives_comp, false_positives_comp, sensitivity_comp, specificity_comp = subsets_and_metrics

        pos_bools = labels_results.is_photo == 1
        neg_bools = labels_results.is_photo == 0
        pred_pos_bools = labels_results.is_photo_pred == 1
        pred_neg_bools = labels_results.is_photo_pred == 0

        positives = labels_results[pos_bools]
        negatives = labels_results[neg_bools]
        true_positives = labels_results[pos_bools & pred_pos_bools]
        false_positives = labels_results[neg_bools & pred_pos_bools]
        true_negatives = labels_results[neg_bools & pred_neg_bools]
        false_negatives = labels_results[pos_bools & pred_neg_bools]

        cases = (true_positives, false_positives, true_negatives, false_negatives, positives, negatives)
        fig = llm_o.plot_conf_matrix(labels_results, self.label_name, self.prediction_name, cases)


    def test_balanced_data_balanced_pred_large(self):

        print('test_balanced_data_balanced_pred_large')

        positives = 400
        negatives = 400
        true_positives = 200
        false_positives = 200
        true_negatives = 200
        false_negatives = 200

        sensitivity = true_positives / positives
        specificity = true_negatives / negatives

        image_ids = [f'{i:03d}' for i in range(1, 801)]
        with_person = [1] * 600 + [0] * 200
        is_photo = [1] * positives + [0] * negatives
        is_photo_pred = [1] * true_positives + [0] * false_negatives + [1] * false_positives + [0] * true_negatives
        labels_results = pd.DataFrame({'image_id': image_ids, 'with_person': with_person, 
                 'is_photo': is_photo, 'is_photo_pred': is_photo_pred})

        pos_bools = labels_results.is_photo == 1
        neg_bools = labels_results.is_photo == 0
        pred_pos_bools = labels_results.is_photo_pred == 1
        pred_neg_bools = labels_results.is_photo_pred == 0

        positives = labels_results[pos_bools]
        negatives = labels_results[neg_bools]
        true_positives = labels_results[pos_bools & pred_pos_bools]
        false_positives = labels_results[neg_bools & pred_pos_bools]
        true_negatives = labels_results[neg_bools & pred_neg_bools]
        false_negatives = labels_results[pos_bools & pred_neg_bools]

        cases = (true_positives, false_positives, true_negatives, false_negatives, positives, negatives)
        fig = llm_o.plot_conf_matrix(labels_results, self.label_name, self.prediction_name, cases)


if __name__ == "__main__":
    unittest.main()