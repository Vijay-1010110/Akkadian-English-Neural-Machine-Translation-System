import torch

class DynamicPadCollator:
    """
    Collates a batch of dataset examples and dynamically pads them
    to the maximum sequence length in the *current batch*, significantly
    saving memory and compute compared to padding to a global max length.
    """
    def __init__(self, pad_id):
        self.pad_id = pad_id
        
    def __call__(self, batch):
        source_lengths = [len(item["source_ids"]) for item in batch]
        target_lengths = [len(item["target_ids"]) for item in batch]
        
        max_source_len = max(source_lengths)
        max_target_len = max(target_lengths)
        
        padded_sources = []
        padded_targets = []
        
        # Attention masks (1 for real tokens, 0 for padding) required by HF models
        source_masks = []
        target_masks = []
        
        for item in batch:
            src = item["source_ids"]
            pad_len_src = max_source_len - len(src)
            padded_sources.append(src + [self.pad_id] * pad_len_src)
            source_masks.append([1] * len(src) + [0] * pad_len_src)
            
            tgt = item["target_ids"]
            pad_len_tgt = max_target_len - len(tgt)
            padded_targets.append(tgt + [self.pad_id] * pad_len_tgt)
            target_masks.append([1] * len(tgt) + [0] * pad_len_tgt)
            
        return {
            "source_ids": torch.tensor(padded_sources, dtype=torch.long),
            "target_ids": torch.tensor(padded_targets, dtype=torch.long),
            "source_attention_mask": torch.tensor(source_masks, dtype=torch.long),
            "target_attention_mask": torch.tensor(target_masks, dtype=torch.long)
        }
