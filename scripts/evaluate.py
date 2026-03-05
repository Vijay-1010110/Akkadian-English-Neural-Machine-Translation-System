import argparse
import os
import sys
import torch
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_loader import load_config
from utils.device import get_device
from custom_logging.logger import get_logger

from data.tokenizer_utils import load_tokenizer
from data.preprocessing import normalize_akkadian

from models.model_registry import ModelRegistry
import models.seq2seq_transformer
import models.mbart_model

from engine.checkpoint_engine import CheckpointEngine
from inference.decoder import DecoderEngine
from evaluation.metrics import evaluate_metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained NMT model.")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Explicit path to checkpoint file")
    parser.add_argument("--eval_data", type=str, default="data/raw/val.csv")
    args = parser.parse_args()
    
    config = load_config(args.config)
    device = get_device()
    logger = get_logger(__name__)
    
    # Basic loads
    model_type = config["model"]["type"]
    if model_type == "custom":
        tokenizer = load_tokenizer(config["tokenizer"]["model_path"])
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config["model"]["pretrained_path"])
        tokenizer.add_tokens(config["tokenizer"].get("special_tokens", []))
        
    model = ModelRegistry.build_model(model_type, config, tokenizer)
    model.to(device)
    
    # Load Weights
    if args.checkpoint and os.path.exists(args.checkpoint):
        state = torch.load(args.checkpoint, map_location=device)
    else:
        ckpt_engine = CheckpointEngine(config["paths"]["checkpoint_dir"])
        state = ckpt_engine.load_best(device)
        
    if not state:
        logger.error("No checkpoint found to evaluate!")
        return
        
    # Handle DDP wrapping mismatches
    model_state = state["model_state_dict"]
    if not hasattr(model, "module") and any(k.startswith("module.") for k in model_state.keys()):
        model_state = {k.replace("module.", ""): v for k, v in model_state.items()}
    model.load_state_dict(model_state, strict=False)
    
    # Inference setup
    decoder = DecoderEngine(model, config, tokenizer, device)
    
    # Read validation data
    eval_path = args.eval_data if args.eval_data != "data/raw/val.csv" else config["data"].get("val_path", args.eval_data)
    
    # If using auto-split from training
    if not os.path.exists(eval_path) and os.path.exists("temp_val.csv"):
        eval_path = "temp_val.csv"
        
    df = pd.read_csv(eval_path)
    akk_col = df.columns[0] if 'akkadian' not in df.columns else 'akkadian'
    eng_col = df.columns[1] if 'english' not in df.columns else 'english'
    df = df.dropna(subset=[akk_col, eng_col])
    
    references = []
    predictions = []
    
    batch_size = config["training"].get("batch_size", 4) * 2 # Can use higher batch size for inference
    
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(df), batch_size), desc="Evaluating..."):
            batch_df = df.iloc[i:i+batch_size]
            
            # Predict
            raw_sources = [normalize_akkadian(str(x)) for x in batch_df[akk_col].values]
            if model_type == "custom":
                bos_id = tokenizer.bos_id()
                eos_id = tokenizer.eos_id()
                pad_id = tokenizer.pad_id()
                source_ids = [[bos_id] + tokenizer.encode(text) + [eos_id] for text in raw_sources]
                
                # Manual pad batch
                max_l = max(len(seq) for seq in source_ids)
                padded_src = [seq + [pad_id]*(max_l-len(seq)) for seq in source_ids]
                input_tensor = torch.tensor(padded_src, dtype=torch.long)
                
                output_tokens = decoder.generate_batch(input_tensor, attention_mask=None)
            else:
                inputs = tokenizer(raw_sources, return_tensors="pt", padding=True, truncation=True, max_length=config["data"]["max_seq_length"])
                output_tokens = decoder.generate_batch(inputs["input_ids"], attention_mask=inputs["attention_mask"])
            
            # Decode predictions
            decoded_preds = decoder.decode_tokens(output_tokens)
            predictions.extend(decoded_preds)
            
            # Save raw references exactly as they appear in the dataset for SACREBLEU
            references.extend([str(x) for x in batch_df[eng_col].values])
            
    # Calculate scores
    metrics = evaluate_metrics(predictions, references)
    logger.info(f"Final Evaluation Metrics: {metrics}")
    
if __name__ == "__main__":
    main()
