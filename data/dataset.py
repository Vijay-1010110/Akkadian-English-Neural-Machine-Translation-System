import torch
from torch.utils.data import Dataset
import pandas as pd
import logging

from .preprocessing import normalize_akkadian, preprocess_english, DataAugmenter
from .tokenizer_utils import load_tokenizer

logger = logging.getLogger(__name__)

class AkkadianDataset(Dataset):
    """
    PyTorch Dataset for loading Akkadian-English pairs.
    Supports on-the-fly augmentation and tokenization.
    """
    def __init__(self, data_path, tokenizer, max_seq_length, augment_config=None, is_training=False, task_prefix=""):
        self.df = pd.read_csv(data_path)
        
        # Determine column names dynamically if standard ones aren't found
        akk_candidates = ['transliteration', 'akkadian', 'source', 'text']
        eng_candidates = ['translation', 'english', 'target']
        self.akk_col = next((c for c in akk_candidates if c in self.df.columns), self.df.columns[0])
        self.eng_col = next((c for c in eng_candidates if c in self.df.columns), self.df.columns[1] if len(self.df.columns) > 1 else self.df.columns[0])
        
        # Clean data inline
        original_len = len(self.df)
        self.df = self.df.dropna(subset=[self.akk_col, self.eng_col])
        self.df = self.df[self.df[self.akk_col].str.strip() != '']
        self.df = self.df[self.df[self.eng_col].str.strip() != '']
        self.df = self.df.reset_index(drop=True)
        
        if len(self.df) < original_len:
            logger.info(f"Dropped {original_len - len(self.df)} empty rows from {data_path}")
            
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_length
        self.is_training = is_training
        self.task_prefix = task_prefix  # e.g. "translate Akkadian to English: "
        
        self.augmenter = None
        if self.is_training and augment_config:
            self.augmenter = DataAugmenter(
                mask_prob=augment_config.get("mask_probability", 0.0),
                syllable_dropout_prob=augment_config.get("syllable_dropout", 0.0),
                noise_injection_prob=augment_config.get("noise_injection", 0.0)
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        akk_text = normalize_akkadian(str(row[self.akk_col]))
        eng_text = preprocess_english(str(row[self.eng_col]))
        
        if self.is_training and self.augmenter:
            akk_text = self.augmenter(akk_text)
            
        # Ensure HF tokenizer compatibility vs custom SPM tokenizer
        if hasattr(self.tokenizer, "encode_plus"):
            # HuggingFace PreTrainedTokenizer - prepend task prefix for T5/mT5
            src_text = self.task_prefix + akk_text
            src_enc = self.tokenizer(src_text, truncation=True, max_length=self.max_seq_len)
            tgt_enc = self.tokenizer(eng_text, truncation=True, max_length=self.max_seq_len)
            return {
                "source_ids": src_enc["input_ids"],
                "target_ids": tgt_enc["input_ids"]
            }
        else:
            # Custom SentencePiece
            bos_id = getattr(self.tokenizer, "bos_id", lambda: None)()
            eos_id = getattr(self.tokenizer, "eos_id", lambda: None)()
            
            # encode output
            encoded_akk = self.tokenizer.encode(akk_text)
            encoded_eng = self.tokenizer.encode(eng_text)
            
            source_ids = ([bos_id] if bos_id is not None else []) + encoded_akk + ([eos_id] if eos_id is not None else [])
            target_ids = ([bos_id] if bos_id is not None else []) + encoded_eng + ([eos_id] if eos_id is not None else [])
            
            if len(source_ids) > self.max_seq_len:
                source_ids = source_ids[:self.max_seq_len-1] + ([eos_id] if eos_id is not None else [])
            if len(target_ids) > self.max_seq_len:
                target_ids = target_ids[:self.max_seq_len-1] + ([eos_id] if eos_id is not None else [])
                
            return {
                "source_ids": source_ids,
                "target_ids": target_ids
            }
