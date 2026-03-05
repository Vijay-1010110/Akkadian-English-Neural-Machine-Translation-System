import torch
import torch.nn as nn
from models.transformer_encoder import TransformerEncoder
from models.transformer_decoder import TransformerDecoder

class Seq2SeqTransformer(nn.Module):
    """
    Translates sequences by encoding source tokens and decoding target tokens 
    with a complete Transformer architecture.
    """
    def __init__(self, encoder_layers, decoder_layers, vocab_size, d_model=512, 
                 nhead=8, dim_feedforward=2048, dropout=0.1, pad_id=0, max_sequence_length=5000):
        super(Seq2SeqTransformer, self).__init__()
        
        self.pad_id = pad_id
        
        self.encoder = TransformerEncoder(
            num_layers=encoder_layers,
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=max_sequence_length
        )
        
        self.decoder = TransformerDecoder(
            num_layers=decoder_layers,
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=max_sequence_length
        )
        
        self._reset_parameters()

    def forward(self, src, tgt):
        """
        src: [batch_size, src_seq_len]
        tgt: [batch_size, tgt_seq_len]
        """
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.create_masks(src, tgt)
        
        memory = self.encoder(src, src_mask=src_mask, src_key_padding_mask=src_padding_mask)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=None,
                              tgt_key_padding_mask=tgt_padding_mask,
                              memory_key_padding_mask=src_padding_mask)
        return output

    def encode(self, src, src_mask=None, src_key_padding_mask=None):
        return self.encoder(src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)

    def decode(self, tgt, memory, tgt_mask=None, memory_key_padding_mask=None):
        dummy_tgt_padding_mask = torch.zeros((tgt.shape[0], tgt.shape[1]), dtype=torch.bool, device=tgt.device)
        return self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=None,
                            tgt_key_padding_mask=dummy_tgt_padding_mask,
                            memory_key_padding_mask=memory_key_padding_mask)

    def create_masks(self, src, tgt):
        """
        Creates padding and causal masks for transformer.
        """
        src_seq_len = src.size(1)
        tgt_seq_len = tgt.size(1)

        # Causal mask for decoder: prevent attending to future tokens
        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len).to(src.device)
        
        # Encoder doesn't need causal mask
        src_mask = torch.zeros((src_seq_len, src_seq_len), device=src.device).type(torch.bool)
        
        # Padding masks
        src_padding_mask = (src == self.pad_id)
        tgt_padding_mask = (tgt == self.pad_id)
        
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _reset_parameters(self):
        """
        Initialize parameters according to Xavier uniform (common for Transformers)
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
