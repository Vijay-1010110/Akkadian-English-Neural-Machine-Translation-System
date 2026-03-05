import torch.optim as optim

def get_optimizer(model, learning_rate=3e-5, weight_decay=0.01):
    """
    Initializes AdamW optimizer.
    
    Args:
        model (nn.Module): The model containing parameters to optimize.
        learning_rate (float): Base learning rate.
        weight_decay (float): Weight decay for regularization.
        
    Returns:
        optim.Optimizer: Configured AdamW optimizer.
    """
    # Separate parameters that shouldn't typically be decayed (e.g., LayerNorm biases)
    no_decay = ["bias", "LayerNorm.weight", "norm"]
    
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
    return optimizer
