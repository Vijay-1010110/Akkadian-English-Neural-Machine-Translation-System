import os
import glob
import torch
import shutil

class CheckpointManager:
    """
    Manages saving and loading of PyTorch models along with optimizer,
    scheduler, and training states. Keeps a limited number of recent 
    checkpoints and tracks the best model based on a metric.
    """
    
    def __init__(self, checkpoint_dir, max_keep=3):
        self.checkpoint_dir = checkpoint_dir
        self.max_keep = max_keep
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
    def save(self, is_best, state_dict, epoch, step):
        """
        Saves the training state.
        
        Args:
            is_best (bool): If true, also saves as the best model.
            state_dict (dict): Contains model, optimizer, scheduler, epoch, step, and seed.
            epoch (int): Current training epoch.
            step (int): Current training step.
        """
        filename = f"checkpoint_epoch{epoch}_step{step}.pt"
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        # Save current checkpoint
        torch.save(state_dict, filepath)
        
        # Maintain symlink or copy to 'latest_checkpoint.pt'
        latest_path = os.path.join(self.checkpoint_dir, "latest_checkpoint.pt")
        shutil.copyfile(filepath, latest_path)
        
        # If it's the best model, save separately
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_checkpoint.pt")
            shutil.copyfile(filepath, best_path)
            
        # Remove older checkpoints to maintain max_keep limit
        self._cleanup_old_checkpoints(exclude=["latest_checkpoint.pt", "best_checkpoint.pt"])
        
    def _cleanup_old_checkpoints(self, exclude):
        """Keeps only the the max_keep most recent specific checkpoint files."""
        # Find all step/epoch checkpoints
        all_checkpoints = glob.glob(os.path.join(self.checkpoint_dir, "checkpoint_epoch*_step*.pt"))
        
        # Sort by modification time, oldest first
        all_checkpoints.sort(key=os.path.getmtime)
        
        # Remove oldest if we exceed limit
        if len(all_checkpoints) > self.max_keep:
            for ckpt_to_remove in all_checkpoints[:-self.max_keep]:
                os.remove(ckpt_to_remove)
                
    def load_latest(self):
        """
        Loads the 'latest' checkpoint if it exists.
        
        Returns:
            dict or None: The state dict if found, else None.
        """
        latest_path = os.path.join(self.checkpoint_dir, "latest_checkpoint.pt")
        if os.path.exists(latest_path):
            return torch.load(latest_path, map_location="cpu")
        return None
        
    def load_best(self):
        """
        Loads the 'best' checkpoint if it exists.
        
        Returns:
            dict or None: The state dict if found, else None.
        """
        best_path = os.path.join(self.checkpoint_dir, "best_checkpoint.pt")
        if os.path.exists(best_path):
            return torch.load(best_path, map_location="cpu")
        return None
