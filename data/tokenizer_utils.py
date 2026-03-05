import sentencepiece as spm
import os
import argparse
import pandas as pd
from .preprocessing import normalize_akkadian, preprocess_english

def train_spm(input_file, model_prefix, vocab_size=12000, character_coverage=1.0, model_type="unigram", extra_specials=None):
    """
    Trains a SentencePiece model on a raw text file containing all sentences.
    Supports integration of domain-specific special vocab (like <DIVINE>).
    """
    user_defined_symbols = ["<MASK>"]
    if extra_specials:
        user_defined_symbols.extend(extra_specials)
        
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        model_type=model_type,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",
        user_defined_symbols=user_defined_symbols
    )
    print(f"SentencePiece model saved to {model_prefix}.model")

def load_tokenizer(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Tokenizer model not found at {model_path}.")
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(model_path)
    return tokenizer

def encode(tokenizer, text, add_bos=True, add_eos=True):
    tokens = tokenizer.encode(text)
    if add_bos:
        tokens = [tokenizer.bos_id()] + tokens
    if add_eos:
        tokens = tokens + [tokenizer.eos_id()]
    return tokens

def decode(tokenizer, tokens, ignore_special=True):
    if ignore_special:
        special_ids = {tokenizer.pad_id(), tokenizer.bos_id(), tokenizer.eos_id(), tokenizer.unk_id()}
        tokens = [t for t in tokens if t not in special_ids]
    return tokenizer.decode(tokens)
