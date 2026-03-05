import random
import numpy as np
import torch
import os

def set_seed(seed=42):
    """
    Sets the random seed across multiple libraries for full reproducibility.
    
    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # GPU operations
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    # CuDNN reproducibility (essential for deterministic research runs)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # OS level python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
