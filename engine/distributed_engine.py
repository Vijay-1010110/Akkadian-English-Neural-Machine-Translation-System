import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import math

class DistributedEngine:
    """
    Wrapper handling distributed Multi-GPU environments.
    Currently uses standard PyTorch capabilities (DDP / DataParallel).
    Can be expanded to use HuggingFace Accelerate if required.
    """
    def __init__(self, device):
        self.device = device
        self.is_distributed = False
        
        if torch.cuda.device_count() > 1:
            self.is_distributed = True
            
    def prepare_model(self, model):
        model = model.to(self.device)
        if self.is_distributed:
            # For Kaggle notebooks simply returning DataParallel is often the easiest robust method
            # without requiring `torchrun` launch scripts for true DDP.
            model = nn.DataParallel(model)
        return model

# ----------------- Helper functions for Trainer initialization ----------------- #

def get_optimizer(model, learning_rate, weight_decay=0.01):
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
    return torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)

def get_scheduler(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    return LambdaLR(optimizer, lr_lambda)

def get_criterion(pad_id, label_smoothing=0.1):
    return nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=label_smoothing)
