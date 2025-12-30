import os
from PIL import Image
from pathlib import Path
import ollama
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import warnings
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports


def convert_tif_to_jpg(source_folder, destination_folder, quality=100):
    """
    Convert .tif files to .jpg format and move copies to destination folder.
    Original .tif files remain in source folder.
    
    Args:
        source_folder (str): Path to folder containing .tif files
        destination_folder (str): Path to destination folder for .jpg files
        quality (int): JPEG quality (1-100, default 85)
    """
    # Create destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)
    
    converted_files = []
    
    # Process all .tif files in source folder
    for filename in os.listdir(source_folder):
        if filename.lower().endswith(('.tif', '.tiff')):
            source_path = os.path.join(source_folder, filename)
            
            # Create new filename with .jpg extension
            base_name = os.path.splitext(filename)[0]
            jpg_filename = f"{base_name}.jpg"
            destination_path = os.path.join(destination_folder, jpg_filename)
            
            try:
                # Open and convert image
                with Image.open(source_path) as img:
                    # Convert to RGB if necessary (TIFF might be in different modes)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Save as JPEG in destination folder
                    img.save(destination_path, 'JPEG', quality=quality, optimize=True)
                
                converted_files.append(jpg_filename)
                print(f"Converted: {filename} -> {jpg_filename}")
                
            except Exception as e:
                print(f"Error converting {filename}: {str(e)}")
    
    print(f"Successfully converted {len(converted_files)} files")
    return converted_files


def convert_image_if_needed(image_path):
    """Convert TIFF to JPG."""
    path = Path(image_path)
    
    if path.suffix.lower() in ['.tif', '.tiff']:
        try:
            img = Image.open(path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Build new path manually
            jpg_path = path.parent / f"{path.stem}_converted.jpg"
            
            img.save(jpg_path, 'JPEG', quality=100)
            print(f"Converted {path} to {jpg_path}")
            return str(jpg_path)
        except Exception as e:
            print(f"Error converting {path}: {e}")
            return None
    else:
        return str(path)


def call_minicpm_model(image_path: str, prompt_function) -> str:
    """
    Call MiniCPM-V model for image analysis.
    
    Args:
        image_path (str): Path to the image file
        prompt (str): Text prompt for the model
        
    Returns:
        str: Generated response from the model
    """

    prompt = prompt_function()

    
    # Set random seed for reproducibility
    torch.manual_seed(100)
    
    # Load model and tokenizer
    model = AutoModel.from_pretrained(
        'openbmb/MiniCPM-V-4_5', 
        trust_remote_code=True,
        attn_implementation='sdpa', 
        torch_dtype=torch.bfloat16
    )
    model = model.eval().cuda()
    
    tokenizer = AutoTokenizer.from_pretrained(
        'openbmb/MiniCPM-V-4_5', 
        trust_remote_code=True
    )
    
    # Load and process image
    image = Image.open(image_path).convert('RGB')
    
    # Configure generation settings
    enable_thinking = False
    stream = True
    
    # Create conversation message
    msgs = [{'role': 'user', 'content': [image, prompt]}]
    
    # Generate response
    answer = model.chat(
        msgs=msgs,
        tokenizer=tokenizer,
        enable_thinking=enable_thinking,
        stream=stream
    )
    
    # Collect streamed response
    generated_text = ""
    for new_text in answer:
        generated_text += new_text
    
    return generated_text


def call_nanollava_model(image_path: str, prompt_function) -> str:
    """
    Call nanoLLaVA model for image analysis.
    
    Args:
        image_path (str): Path to the image file
        prompt (str): Text prompt for the model
        
    Returns:
        str: Generated response from the model
    """

    prompt = prompt_function()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
        """Work around for flash_attn compatibility issues."""
        imports = get_imports(filename)
        if not torch.cuda.is_available() and "flash_attn" in imports:
            imports.remove("flash_attn")
        return imports
    
    model_name = 'qnguyen3/nanoLLaVA-1.5'
    
    # Load model and tokenizer with workaround
    with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map='auto' if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
    
    # Prepare conversation messages
    messages = [
        {"role": "user", "content": f'<image>\n{prompt}'}
    ]
    
    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Process text chunks
    text_chunks = [tokenizer(chunk).input_ids for chunk in text.split('<image>')]
    input_ids = torch.tensor(
        text_chunks[0] + [-200] + text_chunks[1], 
        dtype=torch.long
    ).unsqueeze(0)
    
    # Load and process image
    image = Image.open(image_path)
    image_tensor = model.process_images([image], model.config).to(dtype=model.dtype)
    
    # Generate response
    output_ids = model.generate(
        input_ids.to(device),
        images=image_tensor.to(device),
        max_new_tokens=2048,
        use_cache=True
    )[0]
    
    # Decode response
    generated_text = tokenizer.decode(
        output_ids[input_ids.shape[1]:], 
        skip_special_tokens=True
    ).strip()
    
    return generated_text


def call_ollama_model(image_path, prompt):
    """Make the API call to Ollama."""
    # Convert image if needed
    processed_path = convert_image_if_needed(image_path)
    if processed_path is None:
        raise ValueError(f"Could not process image: {image_path}")
    
    response = ollama.chat(
        model="minicpm-v", 
        #model="llama3.2-vision:latest",
        #model="llama3.2-vision:90b",
        messages=[{
            'role': 'user', 
            'content': prompt,
            'images': [processed_path]
        }],
        options={
        'temperature': 0.1,  # Lower = more deterministic (0.0 to 1.0)
        #'seed': 42           # Fixed seed for reproducibility
    }
    )
    return response['message']['content']

