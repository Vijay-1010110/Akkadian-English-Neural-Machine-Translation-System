import os
import glob
import torch
import shutil
import logging

logger = logging.getLogger(__name__)

class CheckpointEngine:
    """
    Manages saving and loading of models, optimizers, and schedulers.
    Maintains a maximum number of historical checkpoints to save disk space.
    """
    def __init__(self, checkpoint_dir, max_keep=3):
        self.checkpoint_dir = checkpoint_dir
        self.max_keep = max_keep
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
    def save(self, is_best, state_dict, epoch, step):
        filename = f"checkpoint_epoch{epoch}_step{step}.pt"
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        torch.save(state_dict, filepath)
        
        latest_path = os.path.join(self.checkpoint_dir, "latest_checkpoint.pt")
        shutil.copyfile(filepath, latest_path)
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_checkpoint.pt")
            shutil.copyfile(filepath, best_path)
            
        self._cleanup_old_checkpoints()
        logger.info(f"Saved checkpoint: {filename} (Best: {is_best})")
        
    def _cleanup_old_checkpoints(self):
        all_checkpoints = glob.glob(os.path.join(self.checkpoint_dir, "checkpoint_epoch*_step*.pt"))
        all_checkpoints.sort(key=os.path.getmtime)
        
        if len(all_checkpoints) > self.max_keep:
            for ckpt_to_remove in all_checkpoints[:-self.max_keep]:
                try:
                    os.remove(ckpt_to_remove)
                except Exception as e:
                    logger.warning(f"Could not remove old checkpoint {ckpt_to_remove}: {e}")
                
    def load_latest(self, device="cpu"):
        latest_path = os.path.join(self.checkpoint_dir, "latest_checkpoint.pt")
        if os.path.exists(latest_path):
            logger.info("Found latest checkpoint, loading...")
            return torch.load(latest_path, map_location=device)
        return None
        
    def load_best(self, device="cpu"):
        best_path = os.path.join(self.checkpoint_dir, "best_checkpoint.pt")
        if os.path.exists(best_path):
            return torch.load(best_path, map_location=device)
        return None
