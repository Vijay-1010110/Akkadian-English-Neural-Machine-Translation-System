import torch.nn as nn

def get_criterion(pad_id, label_smoothing=0.1):
    """
    Returns the loss function.
    CrossEntropyLoss with label smoothing to prevent overconfidence.
    
    Args:
        pad_id (int): Padding token id to ignore in loss calculation.
        label_smoothing (float): Amount of label smoothing.
        
    Returns:
        nn.Module: Loss criterion.
    """
    # Use standard CrossEntropyLoss with ignore_index for padding
    criterion = nn.CrossEntropyLoss(
        ignore_index=pad_id, 
        label_smoothing=label_smoothing
    )
    return criterion
