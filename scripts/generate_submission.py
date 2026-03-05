import argparse
import os
import sys
import torch
import pandas as pd
from tqdm import tqdm

# Ensure framework is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_loader import load_config
from utils.device import get_device
from custom_logging.logger import get_logger
from data.tokenizer_utils import load_tokenizer
from data.preprocessing import normalize_akkadian
from models.model_registry import ModelRegistry
import models.seq2seq_transformer
import models.mbart_model
from inference.decoder import DecoderEngine

def main():
    parser = argparse.ArgumentParser(description="Generate Kaggle Submission CSV")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Explicit path to best model checkpoint")
    parser.add_argument("--test_data", type=str, default="data/raw/test.csv")
    parser.add_argument("--output", type=str, default="submission.csv")
    args = parser.parse_args()
    
    config = load_config(args.config)
    device = get_device()
    logger = get_logger(__name__)
    
    # Init Tokenizer
    model_type = config["model"]["type"]
    if model_type == "custom":
        tokenizer = load_tokenizer(config["tokenizer"]["model_path"])
        pad_id = tokenizer.pad_id()
        bos_id = tokenizer.bos_id()
        eos_id = tokenizer.eos_id()
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config["model"]["pretrained_path"])
        tokenizer.add_tokens(config["tokenizer"].get("special_tokens", []))
    
    # Init Model
    model = ModelRegistry.build_model(model_type, config, tokenizer)
    model.to(device)
    
    logger.info(f"Loading checkpoint from: {args.checkpoint}")
    state = torch.load(args.checkpoint, map_location=device)
    model_state = state["model_state_dict"] if "model_state_dict" in state else state
    if not hasattr(model, "module") and any(k.startswith("module.") for k in model_state.keys()):
        model_state = {k.replace("module.", ""): v for k, v in model_state.items()}
    model.load_state_dict(model_state, strict=False)
    model.eval()
    
    decoder = DecoderEngine(model, config, tokenizer, device)
    
    test_path = args.test_data if args.test_data != "data/raw/test.csv" else config["data"].get("test_path", args.test_data)
    logger.info(f"Reading test data from {test_path}")
    df = pd.read_csv(test_path)
    
    # Kaggle expects output format: id, translation
    id_col = 'id' if 'id' in df.columns else df.columns[0]
    akk_candidates = ['transliteration', 'akkadian', 'source', 'text']
    akk_col = next((c for c in akk_candidates if c in df.columns), df.columns[1] if len(df.columns) > 1 else df.columns[0])
    
    predictions = []
    batch_size = config["training"].get("batch_size", 8) * 2
    
    with torch.no_grad():
        for i in tqdm(range(0, len(df), batch_size), desc="Translating..."):
            batch_df = df.iloc[i:i+batch_size]
            raw_sources = [normalize_akkadian(str(x)) for x in batch_df[akk_col].values]
            
            if model_type == "custom":
                source_ids = [[bos_id] + tokenizer.encode(text) + [eos_id] for text in raw_sources]
                max_l = max(len(seq) for seq in source_ids)
                padded_src = [seq + [pad_id]*(max_l-len(seq)) for seq in source_ids]
                input_tensor = torch.tensor(padded_src, dtype=torch.long)
                output_tokens = decoder.generate_batch(input_tensor, attention_mask=None)
            else:
                inputs = tokenizer(raw_sources, return_tensors="pt", padding=True, truncation=True, max_length=config["data"]["max_seq_length"])
                output_tokens = decoder.generate_batch(inputs["input_ids"], attention_mask=inputs["attention_mask"])
                
            decoded_preds = decoder.decode_tokens(output_tokens)
            predictions.extend(decoded_preds)
            
    # Save Submission
    submission_df = pd.DataFrame({
        id_col: df[id_col].values,
        "translation": predictions
    })
    
    submission_df.to_csv(args.output, index=False)
    logger.info(f"Successfully generated {len(predictions)} translations at {args.output}")

if __name__ == "__main__":
    main()
