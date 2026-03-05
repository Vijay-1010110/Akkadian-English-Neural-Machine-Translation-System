import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils.checkpoint_manager import CheckpointManager

class Trainer:
    """
    Main training loop for Seq2Seq model supporting mixed precision,
    gradient accumulation, and checkpoint saving/resuming.
    """
    def __init__(self, model, optimizer, scheduler, criterion, train_loader, val_loader,
                 config, checkpoint_manager, logger, device):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.ckpt_manager = checkpoint_manager
        self.logger = logger
        self.device = device
        
        self.epochs = config["training"]["epochs"]
        self.grad_accum_steps = config["training"]["gradient_accumulation_steps"]
        self.mixed_precision = config["training"].get("mixed_precision", True)
        
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)
        
        # Tensorboard writer
        self.writer = SummaryWriter(log_dir=config["paths"]["log_dir"])
        
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
    def load_state(self, state_dict):
        """Loads state to resume training."""
        self.model.load_state_dict(state_dict["model_state_dict"])
        self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        self.scheduler.load_state_dict(state_dict["scheduler_state_dict"])
        if "scaler_state_dict" in state_dict and self.mixed_precision:
            self.scaler.load_state_dict(state_dict["scaler_state_dict"])
            
        self.current_epoch = state_dict["epoch"]
        self.global_step = state_dict["step"]
        if "best_val_loss" in state_dict:
            self.best_val_loss = state_dict["best_val_loss"]
            
        self.logger.info(f"Resumed from epoch {self.current_epoch}, step {self.global_step}")
        
    def _save_state(self, val_loss):
        is_best = val_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = val_loss
            
        state_dict = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if self.mixed_precision else None,
            "epoch": self.current_epoch,
            "step": self.global_step,
            "best_val_loss": self.best_val_loss
        }
        
        self.ckpt_manager.save(is_best, state_dict, self.current_epoch, self.global_step)
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}/{self.epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            src = batch["source_ids"].to(self.device)
            # tgt for input shifted right
            tgt_input = batch["target_ids"][:, :-1].to(self.device)
            # tgt expected output shifted left
            tgt_expected = batch["target_ids"][:, 1:].to(self.device)
            
            # Forward pass with AMP
            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                output = self.model(src, tgt_input)
                # Output shape: [batch_size, seq_len, vocab_size]
                # Flatten output and target for loss computation
                output_flat = output.reshape(-1, output.size(-1))
                tgt_expected_flat = tgt_expected.reshape(-1)
                
                loss = self.criterion(output_flat, tgt_expected_flat)
                loss = loss / self.grad_accum_steps
                
            # Backward pass 
            self.scaler.scale(loss).backward()
            
            total_loss += loss.item() * self.grad_accum_steps
            
            # Optimize step
            if ((batch_idx + 1) % self.grad_accum_steps == 0) or (batch_idx + 1 == len(self.train_loader)):
                # Adjust learning weights
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                
                self.scheduler.step()
                self.global_step += 1
                
                # Logging
                lr = self.optimizer.param_groups[0]["lr"]
                self.writer.add_scalar("Train/Loss", loss.item() * self.grad_accum_steps, self.global_step)
                self.writer.add_scalar("Train/LearningRate", lr, self.global_step)
                
                progress_bar.set_postfix({"loss": f"{loss.item() * self.grad_accum_steps:.4f}", "lr": f"{lr:.2e}"})
                
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
        
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        
        progress_bar = tqdm(self.val_loader, desc="Validating")
        for batch in progress_bar:
            src = batch["source_ids"].to(self.device)
            tgt_input = batch["target_ids"][:, :-1].to(self.device)
            tgt_expected = batch["target_ids"][:, 1:].to(self.device)
            
            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                output = self.model(src, tgt_input)
                output_flat = output.reshape(-1, output.size(-1))
                tgt_expected_flat = tgt_expected.reshape(-1)
                
                loss = self.criterion(output_flat, tgt_expected_flat)
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(self.val_loader)
        self.writer.add_scalar("Validation/Loss", avg_loss, self.global_step)
        
        return avg_loss

    def train(self):
        """Starts or resumes training."""
        self.logger.info("Starting training run...")
        start_epoch = self.current_epoch
        
        for epoch in range(start_epoch, self.epochs):
            self.current_epoch = epoch
            
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            self.logger.info(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save checkpoint at end of epoch
            self._save_state(val_loss)
            
        self.logger.info("Training complete!")
        self.writer.close()
