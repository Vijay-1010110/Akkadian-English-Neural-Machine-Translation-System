import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from data.preprocess import normalize_akkadian, preprocess_english
from tokenizer.tokenizer_utils import load_tokenizer

class AkkadianDataset(Dataset):
    """
    PyTorch Dataset for Akkadian-English pairs.
    """
    def __init__(self, data_path, tokenizer, max_seq_length):
        """
        Args:
            data_path (str): Path to the CSV file containing pairs.
            tokenizer (spm.SentencePieceProcessor): Trained SPM BPE Tokenizer.
            max_seq_length (int): Maximum sequence length for truncation/padding.
        """
        self.df = pd.read_csv(data_path)
        # Assume columns are 'akkadian' and 'english'
        if 'akkadian' not in self.df.columns or 'english' not in self.df.columns:
            # Fallback if names differ 
            self.akk_col = self.df.columns[0]
            self.eng_col = self.df.columns[1]
        else:
            self.akk_col = 'akkadian'
            self.eng_col = 'english'
            
        # Filter out completely empty rows
        self.df = self.df.dropna(subset=[self.akk_col, self.eng_col])
        self.df = self.df[self.df[self.akk_col].str.strip() != '']
        self.df = self.df[self.df[self.eng_col].str.strip() != '']
        self.df = self.df.reset_index(drop=True)
        
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_length
        # Assume standard special tokens are added during SPM training: <s>, </s>, <pad>
        self.pad_id = tokenizer.pad_id()
        self.bos_id = tokenizer.bos_id()
        self.eos_id = tokenizer.eos_id()
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        akk_text = normalize_akkadian(str(row[self.akk_col]))
        eng_text = preprocess_english(str(row[self.eng_col]))
        
        # Tokenize (SentencePiece directly converts string to token IDs)
        # Note: We prepend BOS and append EOS
        source_ids = [self.bos_id] + self.tokenizer.encode(akk_text) + [self.eos_id]
        target_ids = [self.bos_id] + self.tokenizer.encode(eng_text) + [self.eos_id]
        
        # Truncate if necessary (keeping BOS and EOS)
        if len(source_ids) > self.max_seq_len:
            source_ids = source_ids[:self.max_seq_len-1] + [self.eos_id]
        if len(target_ids) > self.max_seq_len:
            target_ids = target_ids[:self.max_seq_len-1] + [self.eos_id]
            
        return {
            "source_ids": source_ids,
            "target_ids": target_ids
        }

def collate_fn(batch, pad_id):
    """
    Pads dynamically to the maximum sequence length in the current batch.
    """
    source_lengths = [len(item["source_ids"]) for item in batch]
    target_lengths = [len(item["target_ids"]) for item in batch]
    
    max_source_len = max(source_lengths)
    max_target_len = max(target_lengths)
    
    padded_sources = []
    padded_targets = []
    
    for item in batch:
        src = item["source_ids"]
        pad_len_src = max_source_len - len(src)
        padded_sources.append(src + [pad_id] * pad_len_src)
        
        tgt = item["target_ids"]
        pad_len_tgt = max_target_len - len(tgt)
        padded_targets.append(tgt + [pad_id] * pad_len_tgt)
        
    return {
        "source_ids": torch.tensor(padded_sources, dtype=torch.long),
        "target_ids": torch.tensor(padded_targets, dtype=torch.long)
    }

def get_dataloaders(config):
    """
    Creates DataLoaders for train and validation datasets.
    """
    tokenizer = load_tokenizer(config["data"]["tokenizer_model_path"])
    pad_id = tokenizer.pad_id()
    
    train_dataset = AkkadianDataset(
        data_path=config["data"]["train_path"],
        tokenizer=tokenizer,
        max_seq_length=config["model"]["max_sequence_length"]
    )
    
    val_dataset = AkkadianDataset(
        data_path=config["data"]["val_path"],
        tokenizer=tokenizer,
        max_seq_length=config["model"]["max_sequence_length"]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size_per_gpu"],
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, pad_id),
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size_per_gpu"] * 2, # can be larger for validation
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, pad_id),
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader
