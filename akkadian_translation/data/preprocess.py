import re

def normalize_akkadian(text):
    """
    Normalizes Akkadian transliterated text based on competition rules:
    1. Lowercase
    2. Replace determinatives: {d} -> <DIVINE> etc., but more generally, 
       we can map known things, or just keep `{x}` as separate tokens if trained, 
       but standardizing it makes sense. The requirements say:
       {d} -> <DIVINE>
    3. Missing text markers:
       <gap> -> <MISSING>
       <big_gap> -> <MISSING>
    4. Syllable splitting:
       a-na-ku -> a na ku
    
    Args:
        text (str): Raw Akkadian transliteration string.
        
    Returns:
        str: Normalized and token-separated string.
    """
    if not isinstance(text, str):
        return ""
        
    # 1. Lowercase everything (logograms are often uppercase like KÙ.BABBAR, 
    # but the prompt says to lowercase text. We will lowercase everything 
    # and optionally handle logograms separately if needed.
    # The requirement specifically says: "Lowercase text")
    text = text.lower()
    
    # 2. Replace known determinatives
    text = text.replace("{d}", " <DIVINE> ")
    # Depending on the dataset, other determinatives like {m}, {f}, {ki} exist.
    # For now, following strict prompt requirements.
    
    # 3. Normalize missing text
    text = text.replace("<gap>", " <MISSING> ")
    text = text.replace("<big_gap>", " <MISSING> ")
    
    # Optional: Logograms (e.g., KÙ.BABBAR). Often represented in uppercase. 
    # Since we lowercased, it becomes kù.babbar. We can leave it as is or handle it.
    # We will leave it as normalized lowercase text for the tokenizer to learn.
    
    # 4. Split syllables (replace dashes with spaces so `a-na-ku` -> `a na ku`)
    text = text.replace("-", " ")
    
    # Special fix for Akkadian specific punctuations if needed, but SentencePiece 
    # will handle word boundaries well. Let's just normalize whitespace.
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_english(text):
    """
    Basic preprocessing for English target side.
    
    Args:
        text (str): Raw English translation string.
        
    Returns:
        str: Normalized English string.
    """
    if not isinstance(text, str):
        return ""
        
    # Lowercase the English side as well to simplify vocabulary matching, 
    # or keep casing depending on BLEU expectations. Standard NMT often lowercases.
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text
