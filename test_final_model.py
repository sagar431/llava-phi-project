import torch
import json
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, AutoConfig
from PIL import Image
import logging
from transformers import BitsAndBytesConfig
import argparse
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_model_and_tokenizer(model_path):
    """Load the final trained model and required tokenizer"""
    try:
        # Initialize quantization config
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        # Use the same base model as training
        base_model_path = "microsoft/phi-1_5"
        logging.info(f"Loading tokenizer from base model: {base_model_path}")
        
        # Load tokenizer from base model
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        
        # Use CLIP processor as in training
        processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Load model configuration and weights from checkpoint
        logging.info(f"Loading model from checkpoint: {model_path}")
        config = AutoConfig.from_pretrained(model_path)
        
        # Load model with same settings as training
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.eos_token_id
        
        return model, tokenizer, processor
    
    except Exception as e:
        logging.error(f"Error loading model and tokenizer: {str(e)}")
        raise

def prepare_image(image_path, processor):
    """Process image for model input"""
    try:
        logging.info(f"Attempting to load image from: {image_path}")
        
        # Check if file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found at {image_path}")
            
        image = Image.open(image_path)
        logging.info(f"Image loaded successfully. Format: {image.format}, Size: {image.size}, Mode: {image.mode}")
        
        # Convert RGBA to RGB if needed
        if image.mode == 'RGBA':
            image = image.convert('RGB')
            logging.info("Converted image from RGBA to RGB")
        
        # Process image
        logging.info("Processing image through processor...")
        processed_image = processor(images=image, return_tensors="pt")
        
        # Move to device
        pixel_values = processed_image.pixel_values.to(device)
        logging.info(f"Image processed successfully. Tensor shape: {pixel_values.shape}")
        
        return pixel_values
        
    except Exception as e:
        logging.error(f"Error processing image {image_path}: {str(e)}")
        raise

def generate_response(model, tokenizer, image_tensor, prompt, max_length=256):
    """Generate response for the given image and prompt"""
    try:
        # Format prompt to match training data format
        formatted_prompt = f"human: <image>\n{prompt}\ngpt:"
        
        # Prepare input
        inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Move image tensor to the same device as model
        image_tensor = image_tensor.to(model.device)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                images=image_tensor,
                max_new_tokens=256,
                min_length=20,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                top_k=40,
                repetition_penalty=1.5,
                no_repeat_ngram_size=3,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode response and clean it up
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up response
        if "gpt:" in response:
            response = response.split("gpt:")[-1].strip()
        if "human:" in response:
            response = response.split("human:")[0].strip()
        if "<image>" in response:
            response = response.replace("<image>", "").strip()
        
        return response
        
    except Exception as e:
        logging.error(f"Error in generate_response: {str(e)}")
        raise

def generate_text_response(model, tokenizer, prompt, max_length=512):
    try:
        # Add system prompt for better context
        formatted_prompt = (
            "You are a helpful AI assistant. Provide clear, accurate, and engaging responses.\n\n"
            f"human: {prompt}\ngpt:"
        )
        
        inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=150,
                min_length=20,
                num_return_sequences=1,
                temperature=0.6,      # Lower temperature for more focused responses
                do_sample=True,
                top_p=0.85,
                top_k=30,            # Lower for more focused responses
                repetition_penalty=1.8,  # Higher to prevent repetition
                no_repeat_ngram_size=4,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                early_stopping=True
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Improved response cleaning
        if "gpt:" in response:
            response = response.split("gpt:")[-1].strip()
        if "human:" in response:
            response = response.split("human:")[0].strip()
        if "<image>" in response:
            response = response.replace("<image>", "").strip()
            
        return response.strip()
        
    except Exception as e:
        logging.error(f"Error in generate_text_response: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Test LLaVA-Phi model with images and text')
    parser.add_argument('--model-path', type=str, default='checkpoints/final_model',
                      help='Path to the final model checkpoint')
    parser.add_argument('--image-path', type=str, 
                      help='Path to the test image (optional)')
    parser.add_argument('--mode', type=str, choices=['text', 'image', 'interactive'],
                      default='interactive', help='Mode of operation')
    args = parser.parse_args()

    # Print arguments
    logging.info(f"Running with arguments:")
    logging.info(f"  Model path: {args.model_path}")
    logging.info(f"  Mode: {args.mode}")
    if args.image_path:
        logging.info(f"  Image path: {args.image_path}")

    try:
        # Load model and processors
        logging.info("Loading model and processors...")
        model, tokenizer, processor = load_model_and_tokenizer(args.model_path)
        
        image_tensor = None
        if args.image_path:
            # Process image if provided
            logging.info("Processing image...")
            image_tensor = prepare_image(args.image_path, processor)

        if args.mode == 'interactive':
            print("\nEntering interactive mode. Commands:")
            print("- 'quit': Exit the program")
            print("- 'load image <path>': Load a new image")
            print("- Just type your prompt for text-only query")
            print("- Start your prompt with 'image:' for image-related queries")
            
            while True:
                prompt = input("\nEnter your prompt: ").strip()
                if prompt.lower() == 'quit':
                    break
                    
                try:
                    if prompt.lower().startswith('load image '):
                        img_path = prompt[11:].strip()
                        image_tensor = prepare_image(img_path, processor)
                        print(f"Successfully loaded image: {img_path}")
                        continue
                        
                    if prompt.lower().startswith('image:'):
                        if image_tensor is None:
                            print("No image is loaded! Please load an image first.")
                            continue
                        actual_prompt = prompt[6:].strip()
                        response = generate_response(model, tokenizer, image_tensor, actual_prompt)
                    else:
                        response = generate_text_response(model, tokenizer, prompt)
                    
                    print(f"\nResponse: {response}")
                except Exception as e:
                    logging.error(f"Error: {e}")
                    
        elif args.mode == 'text':
            # Text-only mode
            test_prompts = [
                "What is the capital of France?",
                "Explain quantum computing in simple terms.",
                "Write a short poem about nature."
            ]
            
            for prompt in test_prompts:
                logging.info(f"\nTesting prompt: {prompt}")
                try:
                    response = generate_text_response(model, tokenizer, prompt)
                    print(f"\nPrompt: {prompt}")
                    print(f"Response: {response}")
                except Exception as e:
                    logging.error(f"Error processing prompt '{prompt}': {e}")
                    
        elif args.mode == 'image':
            if not args.image_path:
                raise ValueError("Image path must be provided for image mode")
                
            test_prompts = [
                "Describe this image in detail.",
                "What objects can you see in this image?",
                "Describe the setting and atmosphere of this image."
            ]
            
            for prompt in test_prompts:
                logging.info(f"\nTesting prompt: {prompt}")
                try:
                    response = generate_response(model, tokenizer, image_tensor, prompt)
                    print(f"\nPrompt: {prompt}")
                    print(f"Response: {response}")
                except Exception as e:
                    logging.error(f"Error processing prompt '{prompt}': {e}")
                    
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    main() 