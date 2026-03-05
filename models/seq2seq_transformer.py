import torch
import torch.nn as nn
import math
from models.model_registry import ModelRegistry

# Reuse existing encoder/decoder blocks but put them in one file for concise modularity
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

@ModelRegistry.register("custom")
class Seq2SeqTransformer(nn.Module):
    """
    Translates sequences using a complete custom Transformer architecture.
    """
    def __init__(self, config, tokenizer):
        super(Seq2SeqTransformer, self).__init__()
        
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_id()
        
        c = config["model"]["custom_config"]
        vocab_size = config["tokenizer"]["vocab_size"]
        d_model = c["hidden_size"]
        nhead = c["attention_heads"]
        dim_feedforward = c["feedforward_dim"]
        dropout = c["dropout"]
        max_seq_len = config["data"]["max_seq_length"]
        encoder_layers = c["encoder_layers"]
        decoder_layers = c["decoder_layers"]
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)
        
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=encoder_layers,
            num_decoder_layers=decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.out = nn.Linear(d_model, vocab_size)
        self._reset_parameters()

    def forward(self, input_ids, decoder_input_ids, src_attention_mask=None, tgt_attention_mask=None):
        src = self.embedding(input_ids) * math.sqrt(self.transformer.d_model)
        src = self.pos_encoder(src)
        
        tgt = self.embedding(decoder_input_ids) * math.sqrt(self.transformer.d_model)
        tgt = self.pos_encoder(tgt)
        
        tgt_seq_len = tgt.size(1)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_seq_len).to(src.device)
        
        src_key_padding_mask = (input_ids == self.pad_id)
        tgt_key_padding_mask = (decoder_input_ids == self.pad_id)
        
        out = self.transformer(
            src, tgt, 
            src_mask=None, 
            tgt_mask=tgt_mask, 
            memory_mask=None,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask # Memory uses same padding as src
        )
        
        logits = self.out(out)
        return logits

    def encode(self, input_ids, src_key_padding_mask=None):
        src = self.embedding(input_ids) * math.sqrt(self.transformer.d_model)
        src = self.pos_encoder(src)
        if src_key_padding_mask is None:
            src_key_padding_mask = (input_ids == self.pad_id)
        memory = self.transformer.encoder(src, src_key_padding_mask=src_key_padding_mask)
        return memory, src_key_padding_mask

    def decode(self, decoder_input_ids, memory, memory_key_padding_mask):
        tgt = self.embedding(decoder_input_ids) * math.sqrt(self.transformer.d_model)
        tgt = self.pos_encoder(tgt)
        tgt_seq_len = tgt.size(1)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_seq_len).to(tgt.device)
        tgt_key_padding_mask = (decoder_input_ids == self.pad_id)
        
        out = self.transformer.decoder(
            tgt, memory, 
            tgt_mask=tgt_mask, 
            memory_mask=None,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        return self.out(out)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
