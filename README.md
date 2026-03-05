# Akkadian Research NMT Framework

A research-grade Neural Machine Translation framework for the Kaggle Deep Past Initiative.

## Structure Overview

*   `configs/`: YAML configuration files.
*   `data/`: Datasets, collation, tokenization, and augmentation.
*   `engine/`: Checkpointing, distributed training, and the core Trainer.
*   `models/`: Custom Seq2Seq and HuggingFace AutoModel wrappers.
*   `evaluation/`: BLEU and chrF++ computation.
*   `inference/`: Beam search and decoding abstractions.
*   `experiments/`: Experiment tracking and logging.
*   `utils/`: Seeding, config loading, and runtime utilities.
*   `logging/`: Standard and TensorBoard logging.
*   `scripts/`: Executable entrypoints (train, evaluate, submit).

## Setup

1.  Enable the environment:
    ```bash
    ../AI_lab/Scripts/activate
    ```
2.  Install requirements:
    ```bash
    pip install -r requirements.txt
    ```

## Execution

Run a training experiment overriding the default config:
```bash
python scripts/train.py --config configs/experiment.yaml
```

Generate a submission:
```bash
python scripts/generate_submission.py --config configs/experiment.yaml --checkpoint /kaggle/working/checkpoints/best_checkpoint.pt
```
