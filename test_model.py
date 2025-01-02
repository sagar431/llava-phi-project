import torch
import torchvision.transforms as transforms
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoModel
from PIL import Image
import json
import os
from peft import PeftModel

def load_model_and_tokenizer(checkpoint_path, base_model_name="microsoft/phi-1_5"):
    """Load the model and tokenizer"""
    # Load base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        load_in_4bit=True,
        device_map="auto"
    )
    
    # Load LoRA weights
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()
    
    return model, tokenizer

def load_clip_model():
    """Load CLIP model for image processing"""
    clip_model_name = "openai/clip-vit-base-patch32"
    processor = AutoProcessor.from_pretrained(clip_model_name)
    clip_model = AutoModel.from_pretrained(clip_model_name).to("cuda")
    return clip_model, processor

def process_image(image_path, clip_model, processor):
    """Process image through CLIP"""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to("cuda")
    with torch.no_grad():
        clip_outputs = clip_model(**inputs)
        image_embeds = clip_outputs.pooler_output
    return image_embeds

def generate_response(model, tokenizer, image_embeds, prompt, max_length=200):
    """Generate response for the given image and prompt"""
    # Format the input
    input_text = f"<image>{prompt}"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=5,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.pad_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def test_model(checkpoint_path, image_path, prompt):
    """Main testing function"""
    # Load models
    model, tokenizer = load_model_and_tokenizer(checkpoint_path)
    clip_model, processor = load_clip_model()
    
    # Process image
    image_embeds = process_image(image_path, clip_model, processor)
    
    # Generate response
    response = generate_response(model, tokenizer, image_embeds, prompt)
    
    return response

if __name__ == "__main__":
    # Example usage
    checkpoint_path = "path/to/checkpoint"  # Replace with actual checkpoint path
    image_path = "path/to/test/image.jpg"   # Replace with test image path
    prompt = "What can you see in this image?"
    
    response = test_model(checkpoint_path, image_path, prompt)
    print(f"Model Response: {response}") 