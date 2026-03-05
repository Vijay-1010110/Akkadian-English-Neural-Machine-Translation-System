import torch
from inference.beam_search import beam_search_decode

class DecoderEngine:
    """
    Abstractions for performing batched inference wrapping both 
    Custom architectures and HuggingFace AutoModels.
    """
    def __init__(self, model, config, tokenizer, device):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.device = device
        
        self.model_type = config["model"]["type"]
        self.beam_size = config["generation"].get("beam_size", 5)
        self.max_len = config["generation"].get("max_output_length", 64)
        self.length_penalty = config["generation"].get("length_penalty", 1.0)
        
        # Depending on Tokenizer type (Sentencepiece vs HF), get special tokens
        if hasattr(self.tokenizer, "bos_token_id"):
            self.bos_id = self.tokenizer.bos_token_id
            self.eos_id = self.tokenizer.eos_token_id
            self.pad_id = self.tokenizer.pad_token_id
        else:
            self.bos_id = self.tokenizer.bos_id()
            self.eos_id = self.tokenizer.eos_id()
            self.pad_id = self.tokenizer.pad_id()
            
    @torch.no_grad()
    def generate_batch(self, input_ids, attention_mask=None):
        """
        Generates sequences for a batch of encoded inputs.
        """
        self.model.eval()
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
            
        if self.model_type in ["mbart", "mt5", "t5"]:
            # Leverage robust HuggingFace generation hooks
            outputs = self.model.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.max_len,
                num_beams=self.beam_size,
                early_stopping=self.config["generation"].get("early_stopping", True),
                length_penalty=self.length_penalty
            )
            # HF generate returns padded tensor batch, convert to explicit list of lists
            return [out.cpu().tolist() for out in outputs]
            
        else: # custom transformer model
            # For our custom transformer we compute explicit padding masks as bools
            src_key_padding_mask = (input_ids == self.pad_id)
            
            output_seqs = beam_search_decode(
                self.model, 
                src=input_ids, 
                src_mask=src_key_padding_mask,
                max_len=self.max_len,
                start_symbol=self.bos_id,
                end_symbol=self.eos_id,
                pad_symbol=self.pad_id,
                beam_size=self.beam_size,
                length_penalty=self.length_penalty
            )
            return [seq.cpu().tolist() for seq in output_seqs]
            
    def decode_tokens(self, token_lists):
        """
        Decodes lists of output token IDs back into string sentences.
        """
        sentences = []
        for tokens in token_lists:
            if hasattr(self.tokenizer, "decode"):
                # Either HF or Sentencepiece native decode handles lists
                decoded = self.tokenizer.decode(tokens, skip_special_tokens=True)
                sentences.append(decoded)
        return sentences
