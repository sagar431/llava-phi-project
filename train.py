import torch
import json
import os
import logging
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import random_split, DataLoader
import wandb
import glob
import heapq
import gc
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from src.dataset import InstructDataset
from tqdm import tqdm
import time
from transformers import BitsAndBytesConfig
import torch.amp as amp

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load config
with open("configs/training_config.json", "r") as f:
    config = json.load(f)

print(config)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

def cleanup_checkpoints(checkpoint_dir, keep_best_n=5, keep_last_n=3):
    """Clean up old checkpoints, keeping only the best N and last N checkpoints"""
    # Get all checkpoint files
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pt"))
    
    if len(checkpoint_files) <= (keep_best_n + keep_last_n):
        return
    
    # Load loss values for each checkpoint
    checkpoint_losses = []
    for checkpoint_file in checkpoint_files:
        try:
            checkpoint = torch.load(checkpoint_file, map_location='cpu')
            loss = checkpoint['loss']
            checkpoint_losses.append((loss, checkpoint_file))
        except:
            continue
    
    # Keep best N checkpoints
    best_checkpoints = set([x[1] for x in heapq.nsmallest(keep_best_n, checkpoint_losses)])
    
    # Keep last N checkpoints
    sorted_by_time = sorted(checkpoint_files, key=os.path.getmtime)
    last_checkpoints = set(sorted_by_time[-keep_last_n:])
    
    # Delete other checkpoints
    keep_checkpoints = best_checkpoints.union(last_checkpoints)
    for checkpoint_file in checkpoint_files:
        if checkpoint_file not in keep_checkpoints:
            try:
                os.remove(checkpoint_file)
                logging.info(f"Deleted checkpoint: {checkpoint_file}")
            except Exception as e:
                logging.error(f"Error deleting checkpoint {checkpoint_file}: {e}")

def evaluate(model, eval_dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in eval_dataloader:
            if batch is None:
                continue
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += outputs.loss.item()
    
    avg_loss = total_loss / len(eval_dataloader)
    model.train()
    return avg_loss

def save_checkpoint(model, optimizer, scheduler, epoch, step, loss, checkpoint_dir):
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
    }
    
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}_step_{step}.pt')
    torch.save(checkpoint, checkpoint_path)
    logging.info(f"Saved checkpoint: {checkpoint_path}")

def verify_data():
    """Verify data files exist and are valid"""
    json_path = config["data_params"]["json_path"]
    image_folder = config["data_params"]["image_folder"]
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Instructions file not found: {json_path}")
    if not os.path.exists(image_folder):
        raise FileNotFoundError(f"Image folder not found: {image_folder}")
    
    # Verify JSON format
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        logging.info(f"Found {len(data)} instructions in {json_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON file {json_path}: {e}")
    
    # Verify at least one image exists
    images = os.listdir(image_folder)
    if not images:
        raise FileNotFoundError(f"No images found in {image_folder}")
    logging.info(f"Found {len(images)} images in {image_folder}")

def get_grad_norm(model):
    """Calculate gradient norm for monitoring"""
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def train():
    # Add verification step
    verify_data()
    
    # Check required files and directories exist
    required_paths = [
        config["data_params"]["json_path"],
        config["data_params"]["image_folder"]
    ]
    
    for path in required_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required path '{path}' does not exist. "
                f"Please ensure all data files and directories are properly set up."
            )
    
    # Initialize wandb
    wandb.init(project="llava-phi", config=config)
    
    # Create checkpoint directory
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize quantization config properly
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,  # Change compute dtype to float16
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    # Initialize model with proper quantization config
    model = AutoModelForCausalLM.from_pretrained(
        config["model_params"]["model_name"],
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16  # Change to bfloat16
    )
    clip_processor = AutoProcessor.from_pretrained(config["model_params"]["clip_model_name"])
    tokenizer = AutoTokenizer.from_pretrained(config["model_params"]["model_name"])
    
    # Setup tokenizer with padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    
    # Initialize dataset
    full_dataset = InstructDataset(
        json_path=config["data_params"]["json_path"],
        image_folder=config["data_params"]["image_folder"],
        clip_processor=clip_processor,
        tokenizer=tokenizer,
        max_length=config["data_params"]["max_length"]
    )
    
    # Split dataset into train and eval
    train_size = int(len(full_dataset) * config["data_params"]["train_split"])
    eval_size = len(full_dataset) - train_size
    train_dataset, eval_dataset = random_split(full_dataset, [train_size, eval_size])
    
    # Create dataloaders using the collate_fn from the full dataset
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config["training_params"]["batch_size"],
        shuffle=True,
        num_workers=config["training_params"]["dataloader_num_workers"],
        prefetch_factor=config["training_params"]["prefetch_factor"],
        collate_fn=full_dataset.collate_fn  # Use collate_fn from full dataset
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config["training_params"]["batch_size"],
        shuffle=False,
        num_workers=config["training_params"]["dataloader_num_workers"],
        prefetch_factor=config["training_params"]["prefetch_factor"],
        collate_fn=full_dataset.collate_fn  # Use collate_fn from full dataset
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training_params"]["learning_rate"],
        weight_decay=config["training_params"]["weight_decay"]
    )
    
    # Learning rate scheduler
    num_training_steps = len(train_dataloader) * config["training_params"]["num_epochs"]
    num_warmup_steps = int(num_training_steps * config["training_params"]["warmup_ratio"])
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Enable gradient checkpointing if configured
    if config["training_params"]["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
    
    # Add debugging information for model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")

    # Verify model is in training mode and on correct device
    model.train()
    logging.info(f"Model device: {next(model.parameters()).device}")
    
    # Initialize gradient scaler
    scaler = amp.GradScaler()
    
    # Training loop
    best_eval_loss = float('inf')
    global_step = 0
    
    for epoch in range(config["training_params"]["num_epochs"]):
        model.train()
        epoch_loss = 0
        running_loss = 0
        samples_count = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
        batch_times = []
        
        for step, batch in enumerate(progress_bar):
            start_time = time.time()
            
            if batch is None:
                continue
            
            # Log first batch statistics
            if step == 0:
                logging.info(f"Batch size: {len(batch['input_ids'])}")
                logging.info(f"Input shape: {batch['input_ids'].shape}")
                logging.info(f"Memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f}MB")
                logging.info(f"Model dtype: {next(model.parameters()).dtype}")
            
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            try:
                # Use autocast for mixed precision
                with amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        use_cache=False
                    )
                    
                    loss = outputs.loss
                    
                    if torch.isnan(loss):
                        logging.warning(f"NaN loss detected at step {step}. Skipping batch.")
                        continue
                    
                    loss = loss / config["training_params"]["gradient_accumulation_steps"]
                
                # Backward pass
                loss.backward()
                
                # Update running loss and epoch loss
                current_loss = loss.item() * config["training_params"]["gradient_accumulation_steps"]
                running_loss += current_loss
                epoch_loss += current_loss  # Add to epoch loss
                samples_count += 1
                current_running_loss = running_loss / samples_count
                
                # Calculate batch processing time
                batch_time = time.time() - start_time
                batch_times.append(batch_time)
                
                # Update progress bar
                progress_bar.set_description(
                    f"Epoch {epoch + 1} | Loss: {current_running_loss:.4f} | "
                    f"Batch time: {batch_time:.2f}s"
                )
                
                if (step + 1) % config["training_params"]["gradient_accumulation_steps"] == 0:
                    # Clip gradients
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 
                        config["training_params"]["max_grad_norm"]
                    )
                    
                    # Optimizer step
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    # Log metrics
                    wandb.log({
                        "train_loss": current_running_loss,
                        "learning_rate": scheduler.get_last_lr()[0],
                        "epoch": epoch,
                        "global_step": global_step,
                        "grad_norm": grad_norm.item() if not torch.isnan(grad_norm) else 0,
                        "batch_time": batch_time,
                        "memory_usage": torch.cuda.memory_allocated() / 1024**2
                    })
                    
                    global_step += 1
                    
            except RuntimeError as e:
                logging.error(f"Error during training: {e}")
                torch.cuda.empty_cache()
                continue
            
            # Clear cache periodically
            if step % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()
        
        # Calculate and log epoch statistics
        avg_batch_time = sum(batch_times) / len(batch_times)
        avg_epoch_loss = epoch_loss / len(train_dataloader)  # Calculate average epoch loss
        
        logging.info(f"Epoch {epoch + 1} statistics:")
        logging.info(f"  Average batch time: {avg_batch_time:.2f}s")
        logging.info(f"  Average loss: {avg_epoch_loss:.4f}")  # Log the actual average loss
        logging.info(f"  Learning rate: {scheduler.get_last_lr()[0]}")
        logging.info(f"  Memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f}MB")
        
        # Save epoch checkpoint with the correct loss
        save_checkpoint(
            model, optimizer, scheduler,
            epoch, global_step, avg_epoch_loss,  # Use average epoch loss
            checkpoint_dir
        )
        cleanup_checkpoints(checkpoint_dir)
    
    # Save final model
    model.save_pretrained(os.path.join(checkpoint_dir, "final_model"))
    wandb.finish()

if __name__ == "__main__":
    train()
