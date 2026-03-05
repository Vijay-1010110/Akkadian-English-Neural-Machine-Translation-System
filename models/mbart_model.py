import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM
import logging
from models.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

class PretrainedSeq2SeqWrapper(nn.Module):
    """
    Wraps HuggingFace PreTrained Seq2Seq models (e.g., mBART, mT5).
    Exposes a unified interface similar to our Custom Transformer.
    """
    def __init__(self, config, tokenizer):
        super().__init__()
        model_path = config["model"]["pretrained_path"]
        logger.info(f"Loading pretrained model from: {model_path}")
        
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        
        # If tokenizer has added new special words, resize embeddings
        if hasattr(tokenizer, "__len__") and len(tokenizer) > self.model.config.vocab_size:
            logger.info("Resizing token embeddings to fit new tokenizer vocabulary.")
            self.model.resize_token_embeddings(len(tokenizer))

    def forward(self, input_ids, decoder_input_ids, src_attention_mask=None, tgt_attention_mask=None):
        # HuggingFace models typically handle the causal masking internally
        # We just need to pass the correctly shifted labels or decoder_input_ids
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=src_attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=tgt_attention_mask
        )
        # return logits: [batch, seq_len, vocab_size]
        return outputs.logits

    def encode(self, input_ids, src_attention_mask=None):
        # Using HF encode hooks
        encoder_outputs = self.model.get_encoder()(
            input_ids=input_ids, 
            attention_mask=src_attention_mask
        )
        return encoder_outputs.last_hidden_state, src_attention_mask

    def decode(self, decoder_input_ids, memory, memory_key_padding_mask=None):
        # Not all HF models implement a clean manual decode step cleanly this way,
        # usually they provide `.generate()`. However if we write custom beam search,
        # we can interface here for standard architectures like BART.
        outputs = self.model(
            encoder_outputs=(memory,),
            decoder_input_ids=decoder_input_ids,
        )
        return outputs.logits

@ModelRegistry.register("mbart")
class MBartModel(PretrainedSeq2SeqWrapper):
    pass

@ModelRegistry.register("mt5")
class MT5Model(PretrainedSeq2SeqWrapper):
    pass

@ModelRegistry.register("t5")
class T5Model(PretrainedSeq2SeqWrapper):
    pass
