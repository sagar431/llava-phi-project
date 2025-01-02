import torch
import json
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from PIL import Image
from transformers import BitsAndBytesConfig

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
    model = AutoModelForCausalLM.from_pretrained(
        config["model_params"]["model_name"],
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
        print(f"Loaded checkpoint from {checkpoint_path} (with filtered quantization states)")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Continuing with base model...")
    
    # Load tokenizer and processor
    tokenizer = AutoTokenizer.from_pretrained(config["model_params"]["model_name"])
    clip_processor = AutoProcessor.from_pretrained(config["model_params"]["clip_model_name"])
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    
    return model, tokenizer, clip_processor

def prepare_image(image_path, processor):
    # (Same as above)
    pass

def generate_response(model, tokenizer, image_tensor, prompt, max_length=512):
    # (Same as above)
    pass

def interactive_inference():
    print("Loading model and processors...")
    model, tokenizer, clip_processor = load_model_and_tokenizer()
    
    while True:
        # Get input from user
        image_path = input("\nEnter image path (or 'quit' to exit): ")
        if image_path.lower() == 'quit':
            break
            
        prompt = input("Enter your prompt: ")
        
        try:
            # Process image
            image_tensor = prepare_image(image_path, clip_processor)
            
            # Generate response
            print("\nGenerating response...")
            prompt = f"<image>{prompt}</image>"
            response = generate_response(model, tokenizer, image_tensor, prompt)
            
            print(f"\nResponse: {response}")
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    interactive_inference() 