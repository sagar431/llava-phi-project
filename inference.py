import torch
import json
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from PIL import Image
import logging
from transformers import BitsAndBytesConfig
from llava_phi import LlavaPhiProcessor, LlavaPhiForCausalLM

# First, install the llava-phi package if not already installed
# !pip install llava-phi

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_model_and_tokenizer():
    """Load the trained model and required tokenizer"""
    # Load config
    with open("configs/training_config.json", "r") as f:
        config = json.load(f)
    
    # Initialize quantization config
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    # Load base model first
    model = LlavaPhiForCausalLM.from_pretrained(
        "susnato/llava-phi-architecture",
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    
    # Load checkpoint and filter out quantization-related keys
    checkpoint_path = "checkpoints/checkpoint_epoch_0_step_2218.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Filter out problematic keys from the state dict
    state_dict = checkpoint['model_state_dict']
    filtered_state_dict = {}
    
    for key, value in state_dict.items():
        # Skip quantization-related keys
        if any(x in key for x in ['.absmax', '.quant_map', '.nested_absmax', '.nested_quant_map', '.quant_state']):
            continue
        filtered_state_dict[key] = value
    
    # Load filtered state dict
    try:
        model.load_state_dict(filtered_state_dict, strict=False)
        logging.info(f"Loaded checkpoint from {checkpoint_path} (with filtered quantization states)")
    except Exception as e:
        logging.warning(f"Error loading checkpoint: {e}")
        logging.warning("Continuing with base model...")
    
    # Load processor and tokenizer
    processor = LlavaPhiProcessor.from_pretrained("susnato/llava-phi-architecture")
    tokenizer = processor.tokenizer
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    
    return model, tokenizer, processor

def prepare_image(image_path, processor):
    """Process image for model input"""
    image = Image.open(image_path)
    # Convert RGBA to RGB if needed
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    processed_image = processor(images=image, return_tensors="pt")
    return processed_image.pixel_values

def generate_response(model, tokenizer, image_tensor, prompt, max_length=512):
    """Generate response for the given image and prompt"""
    try:
        # Prepare input
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Forward pass with image
        with torch.no_grad():
            # First pass with image to get image embeddings
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                images=image_tensor,
                return_dict=True
            )
            
            # Generate text based on the outputs
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                use_cache=True
            )
        
        # Decode response
        response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return response
        
    except Exception as e:
        logging.error(f"Error in generate_response: {str(e)}")
        raise

def main():
    # Load model and processors
    logging.info("Loading model and processors...")
    model, tokenizer, processor = load_model_and_tokenizer()
    
    # Example prompts and images
    test_cases = [
        {
            "image_path": "/home/ubuntu/llava_phi_project/dog.jpg",
            "prompt": "<image>Describe this dog in detail. What breed is it and what is it doing?</image>"
        },
        {
            "image_path": "/home/ubuntu/llava_phi_project/dog.jpg",
            "prompt": "<image>What is the setting of this image? Describe the environment and the dog's behavior.</image>"
        }
    ]
    
    # Run inference
    for i, test_case in enumerate(test_cases, 1):
        logging.info(f"\nTest Case {i}:")
        logging.info(f"Image: {test_case['image_path']}")
        logging.info(f"Prompt: {test_case['prompt']}")
        
        try:
            # Process image
            image_tensor = prepare_image(test_case["image_path"], processor)
            
            # Generate response
            response = generate_response(
                model, 
                tokenizer, 
                image_tensor, 
                test_case["prompt"]
            )
            
            logging.info(f"Response: {response}")
            
        except Exception as e:
            logging.error(f"Error processing test case {i}: {e}")

if __name__ == "__main__":
    main() 