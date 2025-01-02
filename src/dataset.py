import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import json
from transformers import AutoProcessor
from functools import lru_cache
import numpy as np
import logging

class InstructDataset(Dataset):
    def __init__(self, json_path, image_folder, clip_processor, tokenizer, max_length=512, cache_size=5000):
        """Initialize dataset with error checking"""
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Instructions file not found: {json_path}")
        if not os.path.exists(image_folder):
            raise FileNotFoundError(f"Image folder not found: {image_folder}")
        
        self.image_folder = image_folder
        self.clip_processor = clip_processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load instruction data with error handling
        try:
            with open(json_path, 'r') as f:
                self.data = json.load(f)
            logging.info(f"Loaded {len(self.data)} instructions from {json_path}")
            
            # Debug: Check first few image paths
            for i, item in enumerate(self.data[:5]):
                img_path = os.path.join(image_folder, item["image"])
                exists = os.path.exists(img_path)
                logging.info(f"Image {i+1}: {img_path} - Exists: {exists}")
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON file {json_path}: {e}")
        
        # Initialize image cache
        self.cache_size = cache_size
        self._initialize_cache()
    
    def _initialize_cache(self):
        """Initialize LRU cache for images"""
        # Use decorator to create cache
        @lru_cache(maxsize=self.cache_size)
        def _load_and_process_image(image_path):
            try:
                image = Image.open(image_path).convert("RGB")
                image_inputs = self.clip_processor(images=image, return_tensors="pt")
                return {k: v.squeeze(0) for k, v in image_inputs.items()}
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                return None
        
        self._cached_load_image = _load_and_process_image
    
    def _prepare_text(self, conversations):
        """Prepare text with optimized string concatenation"""
        conversation_parts = []
        for conv in conversations:
            # Update to match LLaVA format where keys are "from" and "value"
            role = "Assistant: " if conv["from"] == "gpt" else "Human: "
            conversation_parts.extend([role, conv["value"], "\n"])
        return "".join(conversation_parts)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load and process image using cache
        image_path = os.path.join(self.image_folder, item["image"])
        
        if not os.path.exists(image_path):
            logging.error(f"Image not found: {image_path}")
            return None
        
        image_inputs = self._cached_load_image(image_path)
        
        if image_inputs is None:
            logging.error(f"Failed to process image: {image_path}")
            return None
        
        # Format conversation with optimized string handling
        conversation_text = self._prepare_text(item["conversations"])
        
        # Tokenize text
        text_encoding = self.tokenizer(
            conversation_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "image_inputs": image_inputs,
            "input_ids": text_encoding["input_ids"].squeeze(0),
            "attention_mask": text_encoding["attention_mask"].squeeze(0),
            "labels": text_encoding["input_ids"].squeeze(0).clone()
        }
    
    def collate_fn(self, batch):
        """Custom collate function with batch processing"""
        # Filter None values
        batch = [b for b in batch if b is not None]
        if not batch:
            return None
        
        # Prepare tensors
        image_inputs = {
            k: torch.stack([b["image_inputs"][k] for b in batch])
            for k in batch[0]["image_inputs"].keys()
        }
        
        input_ids = torch.stack([b["input_ids"] for b in batch])
        attention_mask = torch.stack([b["attention_mask"] for b in batch])
        labels = torch.stack([b["labels"] for b in batch])
        
        return {
            "image_inputs": image_inputs,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
