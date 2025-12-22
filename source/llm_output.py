import ast
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
#from . import llm_input as llm_i
from .llm_input import call_ollama_model

def parse_response_to_dict(response_text):
    """Parse the model response into a Python dictionary."""
    try:
        # First try to find dictionary in code blocks
        code_block_match = re.search(r'```(?:python)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if code_block_match:
            dict_str = code_block_match.group(1)
        else:
            # Fallback to finding any dictionary pattern
            dict_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if dict_match:
                dict_str = dict_match.group()
            else:
                return False, None
        
        # Clean up the dictionary string
        dict_str = dict_str.replace('\\_', '_')
        dict_str = dict_str.strip()
        
        # Parse the dictionary
        result_dict = ast.literal_eval(dict_str)
        return True, result_dict
        
    except Exception as e:
        return False, None

def analyze_image_structured(image_path: str, create_analysis_prompt, model_function):
    """
    Main function that orchestrates the image analysis using specified model.
    
    Args:
        image_path (str): Path to the image file
        create_analysis_prompt (callable): Function that returns analysis prompt
        model_function (callable): Either call_minicpm_model or call_nanollava_model
        
    Returns:
        dict: Parsed result dictionary or raw response if parsing fails
    """
    # Define prompt for LLM model
    # prompt = create_analysis_prompt()
    
    # Ask LLM to analyze image using the specified model function
    response_text = model_function(image_path, create_analysis_prompt)
    
    # Parse response text to find dictionary of expected structure
    success, result_dict = parse_response_to_dict(response_text)
    
    if success:
        return result_dict
    else:
        # Save response text in dictionary if parsing fails
        llm_response = {"raw_response": response_text}
        return llm_response


# def analyze_image_structured(image_path, create_analysis_prompt):
#     """Main function that orchestrates the image analysis."""
#     # Define prompt for LLM model:
#     prompt = create_analysis_prompt()
#     # Ask LLM to analyse image, by calling the model and providing 
#     # the defined prompt: 
#     response_text = call_ollama_model(image_path, prompt)
#     # Parse response text, i.e. find dictionary of expected structure
#     # in the response text:
#     success, result_dict = parse_response_to_dict(response_text)
#     
#     if success:
#         return result_dict
#     else:
#         # Save response text in dictionary paired with key "raw_response"
#         # if parsing the response text fails:
#         llm_response = {"raw_response": response_text}
#         return llm_response


def analyze_image_yes_no(image_path, create_analysis_prompt, model_function):
    """Main function that orchestrates the image analysis."""
    # Define prompt for LLM model:
    #prompt = create_analysis_prompt()
    #print(prompt)
    # Ask LLM to analyse image, by calling the model and providing 
    # the defined prompt: 
    response_text = model_function(image_path, create_analysis_prompt)
    
    success, result_dict = parse_yes_no_text(response_text)
    
    if success:
        return result_dict
    else:
        # Save response text in dictionary paired with key "raw_response"
        # if parsing the response text fails:
        llm_response = {"raw_response": response_text}
        return llm_response


def parse_yes_no_text(text):
    """
    Parse text that may start with 'yes' or 'no' and return structured data.
    
    Args:
        text (str): The input text to parse
    
    Returns:
        tuple: (bool, dict or None) - Success status and result dictionary
    """
    try:
        if not text or not text.strip():
            return False, None
        
        text = text.strip()
        is_yes_or_no, first_word = first_word_is_yes_or_no(text)
        
        if is_yes_or_no:
            return True, {'answer': first_word, 'additional_comments': text}
        else:
            return False, None
    except Exception as e:
        return False, None


def first_word_is_yes_or_no(text):
    """
    Check if the first word of a text is 'yes' or 'no' (case-insensitive).
    
    Args:
        text (str): The input text to check
    
    Returns:
        tuple: (bool, str or None) - (True if first word is 'yes' or 'no', the cleaned first word or None)
    """
    if not text or not text.strip():
        return False, None
    
    first_token = text.strip().split()[0].lower()
    
    # Check if it starts with 'yes' or 'no' followed by punctuation
    if re.match(r'^(yes|no)[.,!?;:"()[\]{}]', first_token):
        answer = re.match(r'^(yes|no)', first_token).group(1)
        return True, answer
    
    # Fallback to the original method for clean words
    cleaned_word = re.sub(r'[^a-zA-Z]', '', first_token)
    is_yes_or_no = cleaned_word in ['yes', 'no']
    return is_yes_or_no, cleaned_word if is_yes_or_no else None


def add_pred_values(idx, labels_results, columns, values_to_add):

    """Function to manually add list of values to results data frame, in case
    structural anomalies in the llm response require manual processing
    of the data.
    
    idx: integer referring to image_id
    labels_result: dataframe containing the labels and prediction results
    columns: list of column names
    values_to_add: list of values to add to the row referred to by idx

    """
    selection_bools = labels_results.image_id == idx
    
    labels_results.loc[selection_bools, columns] = values_to_add


def convert_boolean_to_encoding(bool_value):
    """
    Convert boolean value to one-hot encoding (1 or 0).
    
    Args:
        bool_value: The value to convert (should be boolean)
    
    Returns:
        1 if True, 0 if False, None if conversion fails
    """
    try:
        if isinstance(bool_value, bool):
            return int(bool_value)
        # Handle string representations of booleans
        elif isinstance(bool_value, str):
            if bool_value.lower() in ['true', '1', 'yes']:
                return 1
            elif bool_value.lower() in ['false', '0', 'no']:
                return 0
        # Handle numeric representations
        elif isinstance(bool_value, (int, float)):
            return 1 if bool_value else 0
    except Exception:
        pass
    
    return None


def extract_from_structured_dict(img_pred, keys_list_expected, response_variable):
    """
    Extract and convert response variable from a properly structured dictionary.
    
    Args:
        img_pred: Dictionary containing the response
        keys_list_expected: Expected keys in the dictionary
        response_variable: Key for the variable of interest
    
    Returns:
        Encoded value (0 or 1) or None if extraction fails
    """
    try:
        # Check if it's a dictionary
        if not isinstance(img_pred, dict):
            return None
            
        # Get keys and check structure
        keys_list = list(img_pred.keys())
        if sorted(keys_list_expected) != sorted(keys_list):
            return None
        
        # Extract and convert the response variable
        if response_variable not in img_pred:
            return None
            
        bool_value = img_pred[response_variable]
        return convert_boolean_to_encoding(bool_value)
        
    except Exception:
        return None


def extract_from_raw_response(img_pred, response_variable):
    """
    Extract and convert response variable from raw_response text.
    
    Args:
        img_pred: Dictionary containing 'raw_response' key
        response_variable: Key for the variable of interest
    
    Returns:
        Encoded value (0 or 1) or None if extraction fails
    """
    try:
        # Check if it's a dictionary and has raw_response key
        if not isinstance(img_pred, dict):
            return None
            
        if 'raw_response' not in img_pred:
            return None
            
        response_text = img_pred['raw_response']
        if not isinstance(response_text, str):
            return None
        
        # Use the existing parse_response_to_dict function
        success_bool, parsed_dict = parse_response_to_dict(response_text)
        
        # Check if parsing was successful and has our variable
        if not success_bool or parsed_dict is None:
            return None
            
        if not isinstance(parsed_dict, dict) or response_variable not in parsed_dict:
            return None
            
        bool_value = parsed_dict[response_variable]
        return convert_boolean_to_encoding(bool_value)
        
    except Exception:
        return None


def process_single_entry(img_id, img_pred, keys_list_expected, response_variable):
    """
    Process a single entry from the image description dictionary.
    
    Args:
        img_id: Image identifier (for logging)
        img_pred: The entry to process
        keys_list_expected: Expected dictionary keys
        response_variable: Key for the variable of interest
    
    Returns:
        Tuple of (encoded_value, needs_inspection)
        - encoded_value: 0, 1, or None
        - needs_inspection: True if processing failed
    """
    # print(f"\n{img_id}")
    
    # Strategy 1: Try structured dictionary extraction
    try:
        encoded_value = extract_from_structured_dict(img_pred, keys_list_expected, response_variable)
        if encoded_value is not None:
            # print("both conditions true")
            return encoded_value, False
    except Exception:
        pass
    
    # Strategy 2: Try raw_response extraction
    try:
        encoded_value = extract_from_raw_response(img_pred, response_variable)
        if encoded_value is not None:
            # print("\nelif true")
            # print("raw_repsonse_dict:")
            # print(img_id)
            # print("success_bool:")
            # print(True)
            return encoded_value, False
    except Exception:
        pass
    
    # Strategy 3: Failed - needs manual inspection
    print("\nno structure at all:")
    print(img_id)
    
    return None, True


def extract_vals_from_response_dict(img_ids, image_descr, keys_list_expected, response_variable):
    """
    Function to extract values from response dictionary provided by an llm.

    input:
    - img_ids: list of image identifiers defining which entries of the 
    input dictionary should be processed.
    - image_descr: Dictionary where each entry is the analysis result referring to
    one image. Ideally, each entry is itself a dictionary, but there can be exceptions
    where an entry is just a character string.
    - keys_list_expected: List of dictionary keys that each entry of image_descr
    ideally should have if the entry is itself a dictionary.
    - response_variable: The dictionary key referring to the variable of interest, which must
    be a boolean. Each entry of image_descr should have this key if the entry is itself a dictionary.
    This boolean values are translated into one-hot-encoding by this function.

    output:
    - img_ids: the same list of image identifiers that is in the input.
    - response_values: A list of integers (either 1 or 0) or None (when the input could not be 
    successfully processed), where each integer value is the one-hot-encoding of the boolean value in 
    one entry of image_descr. In the image_descr entry this boolean value is referred to
    as response_variable (see input). In cases where the value of the response_variable
    could not be extraced and processed, the value of response_values is None.
    - img_ids_closer_inspection: List of image identifiers referring to those
    entries of image_descr that did not have the desired structure or keys and
    that therefore could not be successfully processed. 
    """
    # Input validation
    if not isinstance(img_ids, list) or not img_ids:
        raise ValueError("img_ids must be a non-empty list")
    
    if not isinstance(image_descr, dict):
        raise ValueError("image_descr must be a dictionary")
    
    # Check that all img_ids exist in image_descr
    missing_ids = [img_id for img_id in img_ids if img_id not in image_descr]
    if missing_ids:
        raise KeyError(f"Image IDs not found in image_descr: {missing_ids}")
    
    # Initialize output lists
    response_values = []
    img_ids_closer_inspection = []
    iter_count = 0
    
    # Process each image ID using try/except approach
    for img_id in img_ids:
        try:
            # Get response from LLM for image id in question
            img_pred = image_descr[img_id]
            
            # Process the entry using modular approach
            encoded_value, needs_inspection = process_single_entry(
                img_id, img_pred, keys_list_expected, response_variable
            )
            
            response_values.append(encoded_value)
            
            if needs_inspection:
                img_ids_closer_inspection.append(img_id)
                
        except KeyError:
            # This shouldn't happen due to validation above, but just in case
            print(f"\nKeyError: {img_id} not found in image_descr")
            response_values.append(None)
            img_ids_closer_inspection.append(img_id)
            
        except Exception as e:
            # Catch any other unexpected errors
            print(f"\nUnexpected error processing {img_id}: {e}")
            response_values.append(None)
            img_ids_closer_inspection.append(img_id)
        
        iter_count += 1
    
    # Verify output consistency
    assert len(img_ids) == len(response_values), "Output lists length mismatch"
    
    return img_ids, response_values, img_ids_closer_inspection


def extract_vals_from_yes_no_response(img_ids, image_descr, keys_list_expected, 
                                    response_variable): 
   #img_type = []
   response_values = []
   #with_person = []
   #with_church = []
   
   # Make empty list to store responses that cannot be parsed
   # due to faulty structure for closer inspection: 
   img_ids_closer_inspection = []
   
   iter_count = 0
   
   # Loop through image ids:
   for img_id in img_ids:
   
       # Get response from LLM for image id in question:
       img_pred = image_descr[img_id]
   
       # Get keys from response dictionary:
       keys_list = list(img_pred.keys())
   
       # Check if structure and keys of response match expectation:
       dict_struct_condition = (type(img_pred) == dict)
       keys_condition = (keys_list_expected == keys_list)
   
       # Check if response key 
       raw_key_condition = keys_list == ['raw_response']
       
       # If the llm response corresponds to the expected
       # structure, get response values as planned:
       if dict_struct_condition and keys_condition:
           
           response_val = img_pred[response_variable]
   
           if response_val == 'yes':
               int_value = int(1)
           else:
               int_value = int(0)
   
           response_values.append(int_value)
           
       else:
           print('\n')
           print('not expected structure:')
           print(img_id)
           img_ids_closer_inspection.append(img_id)
           response_values.append(None)
           
       
       iter_count += 1
   return img_ids, response_values, img_ids_closer_inspection


def get_classification_subsets_metrics(labels_results, var_name, pred_var_name):
    positive_bools = labels_results[var_name] == 1
    negative_bools = labels_results[var_name] == 0
    positive_pred_bools = labels_results[pred_var_name] == 1
    negative_pred_bools = labels_results[pred_var_name] == 0
    
    positives = labels_results[positive_bools]
    negatives = labels_results[negative_bools]
    true_positives = labels_results[positive_bools & positive_pred_bools]
    true_negatives = labels_results[negative_bools & negative_pred_bools]
    
    false_negatives = labels_results[positive_bools & negative_pred_bools]
    false_positives = labels_results[negative_bools & positive_pred_bools]

    sensitivity = true_positives.shape[0] / positives.shape[0]
    # print('sensitivity:')
    # print(sensitivity)
    
    specificity = true_negatives.shape[0] / negatives.shape[0]
    # print('specificity:')
    # print(specificity)

    subsets_and_metrics = (positives, negatives, true_positives, true_negatives, 
                           false_negatives, false_positives, sensitivity, specificity)
    
    return subsets_and_metrics


def plot_conf_matrix(labels_results, label, prediction, cases):
    true_positives, false_positives, true_negatives, false_negatives, positives, negatives = cases
    # Calculate confusion matrix
    cm = confusion_matrix(labels_results[label], labels_results[prediction])
    
    number_true_positives = true_positives.shape[0]
    number_false_positives = false_positives.shape[0]
    number_true_negatives = true_negatives.shape[0]
    number_false_negatives = false_negatives.shape[0]
    
    sensitivity = number_true_positives / positives.shape[0]
    specificity = number_true_negatives / negatives.shape[0]
    if (number_true_positives > 0) and (number_false_positives > 0):
        precision = number_true_positives / (number_true_positives + number_false_positives)
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)
    else:
        precision = None
        f1_score = None
    if positives.shape[0] > 0:
        miss_rate = number_false_negatives / positives.shape[0]
    else:
        miss_rate = None
    
    print("Confusion Matrix:")
    
    plt.figure(figsize=(8,6))
    confusion_matrix_data = [[number_true_negatives, number_false_positives], 
                              [number_false_negatives, number_true_positives]]
    sns.heatmap(confusion_matrix_data, annot=True, fmt='d', 
                xticklabels=['Predicted Negative', 'Predicted Positive'], 
                yticklabels=['Actual Negative', 'Actual Positive'])
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()
    
    print(f'True Positives: {number_true_positives}')
    print(f'False Positives: {number_false_positives}')
    print(f'True Negatives: {number_true_negatives}')
    print(f'False Negatives: {number_false_negatives}')
    print(f'\nSensitivity (Recall): {sensitivity:.4f}')
    print(f'Specificity: {specificity:.4f}')
    if precision is not None:
        print(f'Precision: {precision:.4f}')
        print(f'Miss Rate (False Negative Rate): {miss_rate:.4f}')
        print(f'F1 Score: {f1_score:.4f}')


def save_conf_matrix(labels_results, label, prediction, cases, save_path=None):
    true_positives, false_positives, true_negatives, false_negatives, positives, negatives = cases
    # Calculate confusion matrix
    cm = confusion_matrix(labels_results[label], labels_results[prediction])
    
    number_true_positives = true_positives.shape[0]
    number_false_positives = false_positives.shape[0]
    number_true_negatives = true_negatives.shape[0]
    number_false_negatives = false_negatives.shape[0]
    
    sensitivity = number_true_positives / positives.shape[0]
    specificity = number_true_negatives / negatives.shape[0]
    if (number_true_positives > 0) and (number_false_positives > 0):
        precision = number_true_positives / (number_true_positives + number_false_positives)
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)
    else:
        precision = None
        f1_score = None
    if positives.shape[0] > 0:
        miss_rate = number_false_negatives / positives.shape[0]
    else:
        miss_rate = None
    
    print("Confusion Matrix:")
    
    fig = plt.figure(figsize=(8,6))  # Capture the figure object
    confusion_matrix_data = [[number_true_negatives, number_false_positives], 
                              [number_false_negatives, number_true_positives]]
    sns.heatmap(confusion_matrix_data, annot=True, fmt='d', 
                xticklabels=['Predicted Negative', 'Predicted Positive'], 
                yticklabels=['Actual Negative', 'Actual Positive'])
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, format='pdf', bbox_inches='tight')
    
    plt.show()
    
    print(f'True Positives: {number_true_positives}')
    print(f'False Positives: {number_false_positives}')
    print(f'True Negatives: {number_true_negatives}')
    print(f'False Negatives: {number_false_negatives}')
    print(f'\nSensitivity (Recall): {sensitivity:.4f}')
    print(f'Specificity: {specificity:.4f}')
    if precision is not None:
        print(f'Precision: {precision:.4f}')
        print(f'Miss Rate (False Negative Rate): {miss_rate:.4f}')
        print(f'F1 Score: {f1_score:.4f}')
    
    return fig  # Return the figure object