import re
import random

def normalize_akkadian(text):
    """
    Normalizes Akkadian transliterated text based on competition rules.
    """
    if not isinstance(text, str):
        return ""
        
    text = text.lower()
    
    # Standardize determinatives and missing text to special tokens 
    # that the tokenizer will explicitly recognize.
    text = text.replace("{d}", " <DIVINE> ")
    text = text.replace("<gap>", " <MISSING> ")
    text = text.replace("<big_gap>", " <MISSING> ")
    
    # Syllable splitting: replace meaningful dashes with space 
    # to handle e.g. a-na-ku -> a na ku
    text = text.replace("-", " ")
    
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_english(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class DataAugmenter:
    """
    Handles training-time data augmentations to improve generalization on small datasets.
    """
    def __init__(self, mask_prob=0.1, syllable_dropout_prob=0.05, noise_injection_prob=0.02, mask_token="<MASK>"):
        self.mask_prob = mask_prob
        self.syll_prob = syllable_dropout_prob
        self.noise_prob = noise_injection_prob
        self.mask_token = mask_token
        
    def __call__(self, text):
        """
        Applies augmentations to an already normalized Akkadian string containing space-separated syllables/words.
        """
        if not text:
            return text
            
        tokens = text.split()
        augmented_tokens = []
        
        for token in tokens:
            # Skip augmenting special structural tokens
            if token in ["<DIVINE>", "<MISSING>", "<LOGOGRAM>"]:
                augmented_tokens.append(token)
                continue
                
            r = random.random()
            
            # 1. Syllable Dropout (Entire token is removed)
            if r < self.syll_prob:
                continue
                
            # 2. Token Masking (Replaced with <MASK> token)
            elif r < self.syll_prob + self.mask_prob:
                augmented_tokens.append(self.mask_token)
                
            # 3. Noise Injection (Slight spelling perturbation - simple random char replacement for robustness)
            elif r < self.syll_prob + self.mask_prob + self.noise_prob:
                if len(token) > 1:
                    chars = list(token)
                    idx = random.randint(0, len(chars)-1)
                    # Simple alphabetic noise
                    chars[idx] = random.choice('abcdefghijklmnopqrstuvwxyzšțș')
                    augmented_tokens.append("".join(chars))
                else:
                    augmented_tokens.append(token)
            else:
                augmented_tokens.append(token)
                
        return " ".join(augmented_tokens)
