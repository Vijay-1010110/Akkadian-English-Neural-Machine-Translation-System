import sentencepiece as spm
import os

def load_tokenizer(model_path):
    """
    Loads a trained SentencePiece tokenizer from the given path.
    
    Args:
        model_path (str): Path to the .model file.
        
    Returns:
        spm.SentencePieceProcessor: An initialized tokenizer instance.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Tokenizer model not found at {model_path}.")
        
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(model_path)
    return tokenizer

def encode(tokenizer, text, add_bos=True, add_eos=True):
    """
    Encodes a string into tokens using the tokenizer.
    
    Args:
        tokenizer (spm.SentencePieceProcessor): The tokenizer instance.
        text (str): Input string.
        add_bos (bool): Whether to prepend <s>.
        add_eos (bool): Whether to append </s>.
        
    Returns:
        list[int]: List of token IDs.
    """
    tokens = tokenizer.encode(text)
    if add_bos:
        tokens = [tokenizer.bos_id()] + tokens
    if add_eos:
        tokens = tokens + [tokenizer.eos_id()]
    return tokens

def decode(tokenizer, tokens, ignore_special=True):
    """
    Decodes token IDs back into a string.
    
    Args:
        tokenizer (spm.SentencePieceProcessor): The tokenizer instance.
        tokens (list[int]): List of token IDs.
        ignore_special (bool): Whether to ignore special tokens padding/bos/eos.
        
    Returns:
        str: Decoded string.
    """
    if ignore_special:
        special_ids = {tokenizer.pad_id(), tokenizer.bos_id(), tokenizer.eos_id(), tokenizer.unk_id()}
        tokens = [t for t in tokens if t not in special_ids]
        
    return tokenizer.decode(tokens)
