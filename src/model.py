import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoModel
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training

class ImageProjection(nn.Module):
    def __init__(self, clip_hidden_size, phi_hidden_size):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(clip_hidden_size, phi_hidden_size * 2),
            nn.GELU(),
            nn.Linear(phi_hidden_size * 2, phi_hidden_size)
        )
    
    def forward(self, x):
        return self.projection(x)

class LLaVAPhiModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Load CLIP model
        self.clip_model = AutoModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        
        # Load Phi model
        self.phi_model = AutoModelForCausalLM.from_pretrained(
            config["model_params"]["model_name"],
            torch_dtype=torch.float16,
            trust_remote_code=True,
            load_in_4bit=config["model_params"]["load_in_4bit"]
        )
        
        # Prepare Phi model for LoRA training
        self.phi_model = prepare_model_for_kbit_training(self.phi_model)
        
        # Add LoRA adapter
        lora_config = LoraConfig(
            r=config["lora_params"]["r"],
            lora_alpha=config["lora_params"]["lora_alpha"],
            target_modules=config["lora_params"]["target_modules"],
            lora_dropout=config["lora_params"]["lora_dropout"],
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.phi_model = get_peft_model(self.phi_model, lora_config)
        
        # Create projection layer
        self.image_projection = ImageProjection(
            self.clip_model.config.hidden_size,
            self.phi_model.config.hidden_size
        )
        
        # Freeze CLIP model
        for param in self.clip_model.parameters():
            param.requires_grad = False
    
    def forward(self, input_ids, attention_mask, image_inputs, labels=None):
        # Process image through CLIP
        clip_outputs = self.clip_model(**image_inputs)
        image_features = clip_outputs.pooler_output
        
        # Project image features
        projected_features = self.image_projection(image_features)
        
        # Add image features to the input embeddings
        inputs_embeds = self.phi_model.get_input_embeddings()(input_ids)
        image_token_mask = (input_ids == self.tokenizer.convert_tokens_to_ids("<image>"))
        inputs_embeds[image_token_mask] = projected_features
        
        # Forward through Phi model
        outputs = self.phi_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels if labels is not None else None,
            return_dict=True
        )
        
        return outputs
    
    def generate(self, input_ids, attention_mask, image_inputs, **kwargs):
        # Process image through CLIP
        clip_outputs = self.clip_model(**image_inputs)
        image_features = clip_outputs.pooler_output
        
        # Project image features
        projected_features = self.image_projection(image_features)
        
        # Add image features to the input embeddings
        inputs_embeds = self.phi_model.get_input_embeddings()(input_ids)
        image_token_mask = (input_ids == self.tokenizer.convert_tokens_to_ids("<image>"))
        inputs_embeds[image_token_mask] = projected_features
        
        # Generate through Phi model
        outputs = self.phi_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs
        )
        
        return outputs
