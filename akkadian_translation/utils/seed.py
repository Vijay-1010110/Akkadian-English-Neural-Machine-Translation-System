import random
import numpy as np
import torch
import os

def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    
    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # GPU operations
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    # CuDNN reproducibility (might slow down training slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # OS level
    os.environ['PYTHONHASHSEED'] = str(seed)
