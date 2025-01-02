import wandb
import logging
import time
import torch
import psutil
import os
from pathlib import Path
import json
from datetime import datetime

class TrainingMonitor:
    def __init__(self, config, project_name="llava-phi"):
        self.start_time = time.time()
        self.batch_start_time = None
        self.config = config
        self.best_eval_loss = float('inf')
        
        # Initialize wandb
        wandb.init(
            project=project_name,
            config=config,
            name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # Setup logging directory
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize metrics history
        self.metrics_history = {
            "train_loss": [],
            "eval_loss": [],
            "learning_rate": [],
            "gpu_utilization": [],
            "throughput": []
        }
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_file = self.log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def start_batch(self):
        """Mark the start of a batch"""
        self.batch_start_time = time.time()
    
    def end_batch(self, loss, learning_rate, batch_size, global_step):
        """Log metrics at the end of a batch"""
        batch_time = time.time() - self.batch_start_time
        throughput = batch_size / batch_time
        
        # Get GPU stats
        gpu_utilization = torch.cuda.utilization()
        gpu_memory_used = torch.cuda.max_memory_allocated() / 1e9  # Convert to GB
        gpu_memory_cached = torch.cuda.max_memory_reserved() / 1e9
        
        # Get CPU stats
        cpu_percent = psutil.cpu_percent()
        ram_used = psutil.virtual_memory().percent
        
        # Log to wandb
        metrics = {
            "train_loss": loss,
            "learning_rate": learning_rate,
            "throughput": throughput,
            "batch_time": batch_time,
            "gpu_utilization": gpu_utilization,
            "gpu_memory_used_gb": gpu_memory_used,
            "gpu_memory_cached_gb": gpu_memory_cached,
            "cpu_percent": cpu_percent,
            "ram_used_percent": ram_used,
            "global_step": global_step
        }
        wandb.log(metrics)
        
        # Update history
        for key in self.metrics_history:
            if key in metrics:
                self.metrics_history[key].append(metrics[key])
        
        # Log to file
        logging.info(
            f"Step {global_step}: loss={loss:.4f}, lr={learning_rate:.2e}, "
            f"throughput={throughput:.2f} img/s, GPU mem={gpu_memory_used:.2f}GB"
        )
    
    def log_evaluation(self, eval_loss, global_step):
        """Log evaluation metrics"""
        is_best = eval_loss < self.best_eval_loss
        if is_best:
            self.best_eval_loss = eval_loss
        
        wandb.log({
            "eval_loss": eval_loss,
            "best_eval_loss": self.best_eval_loss,
            "global_step": global_step
        })
        
        logging.info(
            f"Evaluation at step {global_step}: loss={eval_loss:.4f} "
            f"(best={self.best_eval_loss:.4f})"
        )
        
        return is_best
    
    def save_training_state(self, epoch, global_step):
        """Save training state and metrics"""
        state = {
            "epoch": epoch,
            "global_step": global_step,
            "best_eval_loss": self.best_eval_loss,
            "metrics_history": self.metrics_history,
            "training_time": time.time() - self.start_time
        }
        
        state_file = self.log_dir / f"training_state_{global_step}.json"
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def log_epoch_summary(self, epoch, epoch_loss):
        """Log summary at the end of an epoch"""
        epoch_time = time.time() - self.start_time
        hours, remainder = divmod(epoch_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        logging.info(
            f"\nEpoch {epoch} Summary:\n"
            f"Average Loss: {epoch_loss:.4f}\n"
            f"Best Eval Loss: {self.best_eval_loss:.4f}\n"
            f"Time Elapsed: {int(hours)}h {int(minutes)}m {int(seconds)}s\n"
            f"GPU Memory Peak: {torch.cuda.max_memory_allocated()/1e9:.2f}GB\n"
        )
    
    def finish(self):
        """Cleanup and final logging"""
        total_time = time.time() - self.start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        logging.info(
            f"\nTraining Complete!\n"
            f"Total Time: {int(hours)}h {int(minutes)}m {int(seconds)}s\n"
            f"Best Eval Loss: {self.best_eval_loss:.4f}\n"
        )
        
        # Save final state
        self.save_training_state(-1, -1)
        
        # Close wandb
        wandb.finish() 