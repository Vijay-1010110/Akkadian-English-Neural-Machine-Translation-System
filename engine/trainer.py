import torch
from tqdm import tqdm
import logging
from utils.timer import Timer

logger = logging.getLogger(__name__)

class Trainer:
    """
    Core research training loop handling Curriculum phases, AMP, Gradient Accumulation,
    and Validation scoring.
    """
    def __init__(self, model, optimizer, scheduler, criterion, 
                 train_loaders, val_loader, config, checkpoint_engine, tb_logger, device):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        
        # Dictionary of dataloaders for curriculum (e.g., {"phase1": loader1, "phase2": loader2})
        self.train_loaders = train_loaders 
        self.val_loader = val_loader
        
        self.config = config
        self.ckpt_engine = checkpoint_engine
        self.tb = tb_logger
        self.device = device
        
        tc = config["training"]
        self.epochs = tc["epochs"]
        self.grad_accum_steps = tc.get("gradient_accumulation_steps", 1)
        self.mixed_precision = tc.get("mixed_precision", True)
        self.clip_norm = tc.get("clip_norm", 1.0)
        
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)
        
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
    def load_state(self, state_dict):
        # Handle DataParallel unwrapping if loading into a single GPU later, or vice versa
        model_state = state_dict["model_state_dict"]
        if hasattr(self.model, "module") and not any(k.startswith("module.") for k in model_state.keys()):
            model_state = {"module."+k: v for k, v in model_state.items()}
            
        self.model.load_state_dict(model_state, strict=False)
        self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        self.scheduler.load_state_dict(state_dict["scheduler_state_dict"])
        if "scaler_state_dict" in state_dict and self.mixed_precision:
            self.scaler.load_state_dict(state_dict["scaler_state_dict"])
            
        self.current_epoch = state_dict.get("epoch", 0)
        self.global_step = state_dict.get("step", 0)
        self.best_val_loss = state_dict.get("best_val_loss", float('inf'))
        
    def _save_state(self, val_loss):
        is_best = val_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = val_loss
            
        state_dict = {
            "model_state_dict": self.model.state_dict(),
            "epoch": self.current_epoch,
            "step": self.global_step,
            "best_val_loss": self.best_val_loss
        }
        
        if self.config["training"].get("save_optimizer_state", False):
            state_dict["optimizer_state_dict"] = self.optimizer.state_dict()
            state_dict["scheduler_state_dict"] = self.scheduler.state_dict()
            state_dict["scaler_state_dict"] = self.scaler.state_dict() if self.mixed_precision else None
            
            "epoch": self.current_epoch,
            "step": self.global_step,
            "best_val_loss": self.best_val_loss
        }
        self.ckpt_engine.save(is_best, state_dict, self.current_epoch, self.global_step)
        
    def _get_current_loader(self):
        curr_cfg = self.config.get("curriculum", {})
        if not curr_cfg.get("enabled", False) or "phase2" not in self.train_loaders:
            return self.train_loaders.get("default") or self.train_loaders.get("phase1")
            
        p1_epochs = curr_cfg.get("phase1_epochs", 3)
        if self.current_epoch < p1_epochs:
            return self.train_loaders["phase1"]
        else:
            return self.train_loaders["phase2"]

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        train_loader = self._get_current_loader()
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch+1}/{self.epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            src = batch["source_ids"].to(self.device)
            # tgt for input shifted right
            tgt_input = batch["target_ids"][:, :-1].to(self.device)
            # expected output shifted left
            tgt_expected = batch["target_ids"][:, 1:].to(self.device)
            
            src_mask = batch.get("source_attention_mask")
            tgt_mask = batch.get("target_attention_mask")
            if src_mask is not None: src_mask = src_mask.to(self.device)
            if tgt_mask is not None: tgt_mask = tgt_mask[:, :-1].to(self.device)
            
            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                if self.config["model"]["type"] == "custom":
                    output = self.model(src, tgt_input)
                else:
                    output = self.model(src, tgt_input, src_attention_mask=src_mask, tgt_attention_mask=tgt_mask)
                    
                loss = self.criterion(output.reshape(-1, output.size(-1)), tgt_expected.reshape(-1))
                loss = loss / self.grad_accum_steps
                
            self.scaler.scale(loss).backward()
            total_loss += loss.item() * self.grad_accum_steps
            
            if ((batch_idx + 1) % self.grad_accum_steps == 0) or (batch_idx + 1 == len(train_loader)):
                if self.clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
                    
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                
                self.scheduler.step()
                self.global_step += 1
                
                lr = self.optimizer.param_groups[0]["lr"]
                self.tb.log_scalar("Train/Loss", loss.item() * self.grad_accum_steps, self.global_step)
                self.tb.log_scalar("Train/LearningRate", lr, self.global_step)
                pbar.set_postfix({"loss": f"{loss.item() * self.grad_accum_steps:.4f}"})
                
        return total_loss / len(train_loader)
        
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        pbar = tqdm(self.val_loader, desc="Validating")
        
        for batch in pbar:
            src, tgt_input, tgt_expected = batch["source_ids"].to(self.device), batch["target_ids"][:, :-1].to(self.device), batch["target_ids"][:, 1:].to(self.device)
            src_mask = batch.get("source_attention_mask")
            tgt_mask = batch.get("target_attention_mask")
            if src_mask is not None: src_mask = src_mask.to(self.device)
            if tgt_mask is not None: tgt_mask = tgt_mask[:, :-1].to(self.device)
            
            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                if self.config["model"]["type"] == "custom":
                    output = self.model(src, tgt_input)
                else:
                    output = self.model(src, tgt_input, src_attention_mask=src_mask, tgt_attention_mask=tgt_mask)
                    
                loss = self.criterion(output.reshape(-1, output.size(-1)), tgt_expected.reshape(-1))
                
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(self.val_loader)
        self.tb.log_scalar("Validation/Loss", avg_loss, self.global_step)
        return avg_loss

    def train(self):
        logger.info("Starting training loop...")
        for epoch in range(self.current_epoch, self.epochs):
            self.current_epoch = epoch
            
            with Timer(f"Epoch {epoch+1} Training", logger):
                train_loss = self.train_epoch()
                
            with Timer(f"Epoch {epoch+1} Validation", logger):
                val_loss = self.validate()
                
            logger.info(f"End of Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            self._save_state(val_loss)
