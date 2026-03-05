import argparse
import os
import sys

# Ensure akkadian_research is in the path when run as a script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.tokenizer_utils import train_spm
from utils.config_loader import load_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train custom SentencePiece Tokenizer")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    parser.add_argument("--data_csv", type=str, default="data/raw/train.csv")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # 1. Prepare temp text dump
    import pandas as pd
    from data.preprocessing import normalize_akkadian, preprocess_english
    
    df = pd.read_csv(args.data_csv)
    akk_col = df.columns[0] if 'akkadian' not in df.columns else 'akkadian'
    eng_col = df.columns[1] if 'english' not in df.columns else 'english'
    df = df.dropna(subset=[akk_col, eng_col])
    
    temp_txt = "temp_corpus.txt"
    with open(temp_txt, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            if str(row[akk_col]).strip(): f.write(normalize_akkadian(str(row[akk_col])) + "\n")
            if str(row[eng_col]).strip(): f.write(preprocess_english(str(row[eng_col])) + "\n")
            
    # 2. Train with domain specials
    specials = config["tokenizer"].get("special_tokens", ["<DIVINE>", "<MISSING>", "<LOGOGRAM>"])
    os.makedirs(os.path.dirname(config["tokenizer"]["model_path"]), exist_ok=True)
    
    train_spm(
        input_file=temp_txt,
        model_prefix=config["tokenizer"]["model_path"].replace(".model", ""),
        vocab_size=config["tokenizer"]["vocab_size"],
        character_coverage=1.0,
        extra_specials=specials
    )
    
    os.remove(temp_txt)
