import torch
import torch.nn as nn
import logging
from inference.decoder import DecoderEngine

logger = logging.getLogger(__name__)

class EnsembleModel(nn.Module):
    """
    Wraps multiple models and averages their raw output logits during decoding 
    forward passes. Used natively with Beam Search engine.
    """
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.num_models = len(models)
        logger.info(f"Initialized Ensemble with {self.num_models} active checkpoints.")
        
    def encode(self, input_ids, src_key_padding_mask=None, src_attention_mask=None):
        # We need to maintain separate memories for each model since hidden sizes 
        # or architectures might subtly differ, but realistically ensembling works 
        # best over identical architectures with different random seeds or CV folds.
        memories = []
        for model in self.models:
            if hasattr(model, "encode"): # custom transformer
                mem, mask = model.encode(input_ids, src_key_padding_mask=src_key_padding_mask)
            else: # HF AutoModelForSeq2Seq
                encoder_outputs = model.get_encoder()(input_ids=input_ids, attention_mask=src_attention_mask)
                mem = encoder_outputs.last_hidden_state
                mask = src_attention_mask
            memories.append((mem, mask))
            
        return memories, src_key_padding_mask

    def decode(self, decoder_input_ids, encoded_memories, memory_key_padding_mask=None):
        all_logits = []
        
        for i, model in enumerate(self.models):
            mem, _ = encoded_memories[i]
            
            if hasattr(model, "decode"): # custom transformer
                logits = model.decode(decoder_input_ids, mem, memory_key_padding_mask=memory_key_padding_mask)
            else: # HF wrapper standard
                outputs = model(encoder_outputs=(mem,), decoder_input_ids=decoder_input_ids)
                logits = outputs.logits
                
            all_logits.append(logits)
            
        # Average probabilities (Logits sum -> softmax conceptually, simple average works fine)
        avg_logits = sum(all_logits) / self.num_models
        return avg_logits
