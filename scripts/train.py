import argparse
import os
import sys
from torch.utils.data import DataLoader

# Add the project root (akkadian_research) to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_loader import load_config
from utils.seed import set_seed
from utils.device import get_device
from custom_logging.logger import get_logger
from custom_logging.tensorboard_logger import TensorBoardLogger

from data.tokenizer_utils import load_tokenizer
from data.dataset import AkkadianDataset
from data.collator import DynamicPadCollator

from models.model_registry import ModelRegistry
import models.seq2seq_transformer
import models.mbart_model

from engine.checkpoint_engine import CheckpointEngine
from engine.distributed_engine import DistributedEngine, get_optimizer, get_scheduler, get_criterion
from engine.trainer import Trainer
from experiments.experiment_runner import ExperimentTracker

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # 1. Setup Environment
    set_seed(42)
    device = get_device()
    logger = get_logger(__name__, config["paths"]["log_dir"])
    tb_logger = TensorBoardLogger(config["paths"]["log_dir"])
    tracker = ExperimentTracker(config["paths"]["experiment_log_path"])
    
    logger.info(f"Using device: {device}")
    
    # 2. Tokenizer
    model_type = config["model"]["type"]
    if model_type == "custom":
        tokenizer = load_tokenizer(config["tokenizer"]["model_path"])
        pad_id = tokenizer.pad_id()
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config["model"]["pretrained_path"])
        # Add requested special tokens if they don't exist
        num_added = tokenizer.add_tokens(config["tokenizer"].get("special_tokens", []))
        logger.info(f"Added {num_added} special tokens to pretrained tokenizer.")
        pad_id = tokenizer.pad_token_id

    # 3. Data Loaders (Curriculum Support)
    collator = DynamicPadCollator(pad_id)
    max_len = config["data"]["max_seq_length"]
    bs = config["training"]["batch_size"]
    
    # Auto-generate val and trim train if val doesn't exist natively
    # This prevents FileNotFoundError during dataloading
    val_path = config["data"]["val_path"]
    train_path = config["data"]["train_path"]
    
    if not os.path.exists(val_path) and os.path.exists(train_path):
        import pandas as pd
        from sklearn.model_selection import train_test_split
        logger.info(f"Validation file {val_path} not found. Creating a 90/10 split from {train_path}...")
        
        df = pd.read_csv(train_path)
        train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
        
        # We need a local place to save these overriding the Kaggle read-only input
        tmp_train = "temp_train.csv"
        tmp_val = "temp_val.csv"
        train_df.to_csv(tmp_train, index=False)
        val_df.to_csv(tmp_val, index=False)
        
        train_path = tmp_train
        val_path = tmp_val
    
    # Task prefix for T5/mT5 models (tells the model what task to perform)
    task_prefix = config["data"].get("task_prefix", "")
    
    # Phase 2 (Noisy)
    train_ds_phase2 = AkkadianDataset(train_path, tokenizer, max_len, 
                                      augment_config=config.get("augmentation"), is_training=True, task_prefix=task_prefix)
    loader_p2 = DataLoader(train_ds_phase2, batch_size=bs, shuffle=True, collate_fn=collator, num_workers=4)
    
    # Phase 1 (Clean) - disable augmentations for curriculum phase 1
    train_ds_phase1 = AkkadianDataset(train_path, tokenizer, max_len, 
                                      augment_config=None, is_training=True, task_prefix=task_prefix)
    loader_p1 = DataLoader(train_ds_phase1, batch_size=bs, shuffle=True, collate_fn=collator, num_workers=4)
    
    val_ds = AkkadianDataset(val_path, tokenizer, max_len, is_training=False, task_prefix=task_prefix)
    val_loader = DataLoader(val_ds, batch_size=bs*2, shuffle=False, collate_fn=collator, num_workers=4)
    
    train_loaders = {"phase1": loader_p1, "phase2": loader_p2}
    
    # 4. Model & Engine setup
    model = ModelRegistry.build_model(model_type, config, tokenizer)
    dist_engine = DistributedEngine(device)
    model = dist_engine.prepare_model(model)
    
    # 5. Optimizer & Criterion
    epochs = config["training"]["epochs"]
    total_steps = epochs * len(loader_p1) // config["training"].get("gradient_accumulation_steps", 1)
    
    optimizer = get_optimizer(model, config["training"]["learning_rate"], config["training"].get("weight_decay", 0.01))
    scheduler = get_scheduler(optimizer, config["training"]["warmup_steps"], total_steps)
    criterion = get_criterion(pad_id, config["training"].get("label_smoothing", 0.1))
    
    # 6. Checkpoints & Trainer
    ckpt_engine = CheckpointEngine(config["paths"]["checkpoint_dir"], max_keep=config["training"].get("save_total_limit", 3))
    
    trainer = Trainer(
        model=model, optimizer=optimizer, scheduler=scheduler, criterion=criterion,
        train_loaders=train_loaders, val_loader=val_loader, config=config,
        checkpoint_engine=ckpt_engine, tb_logger=tb_logger, device=device
    )
    
    # Auto-resume functionality for Kaggle continuity
    latest_state = ckpt_engine.load_latest(device)
    if latest_state:
        trainer.load_state(latest_state)
        
    # 7. Execute Training
    import time
    start_time = time.time()
    
    try:
        trainer.train()
    except Exception as e:
        logger.error(f"Training interrupted: {e}")
        raise e
    finally:
        tb_logger.close()
        # Log final metrics
        best_loss = trainer.best_val_loss
        total_time = time.time() - start_time
        tracker.log_experiment(config, {"val_loss": best_loss}, total_time)
        logger.info(f"Experiment logged to {tracker.log_path}.")

if __name__ == "__main__":
    main()
