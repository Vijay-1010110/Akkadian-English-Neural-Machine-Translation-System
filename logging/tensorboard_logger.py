import os
from torch.utils.tensorboard import SummaryWriter

class TensorBoardLogger:
    """
    Wrapper for TensorBoard SummaryWriter to track metrics and experiment states.
    """
    def __init__(self, log_dir):
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        
    def log_scalar(self, tag, value, step):
        """Logs a single scalar value."""
        self.writer.add_scalar(tag, value, step)
        
    def log_scalars(self, main_tag, tag_scalar_dict, step):
        """Logs multiple scalars under the same main tag."""
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)
        
    def close(self):
        """Closes the TensorBoard writer."""
        self.writer.close()
