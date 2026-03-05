import torch

def get_device():
    """
    Determines the best available compute device. Support for multi-GPU
    will be handled via Accelerate/DDP in the distributed engine.
    
    Returns:
        torch.device: The selected device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    else:
        return torch.device("cpu")
