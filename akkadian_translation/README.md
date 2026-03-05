# Akkadian → English Neural Machine Translation

This repository contains a complete, production-ready implementation of a Sequence-to-Sequence Transformer model for translating Akkadian transliteration to English, designed for the Kaggle Deep Past Initiative competition.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Prepare Data (Optional, if running locally):
   ```bash
   python scripts/prepare_dataset.py
   ```

3. Train Tokenizer:
   ```bash
   python scripts/train_tokenizer.py
   ```

4. Train Model:
   ```bash
   python scripts/train.py
   ```

5. Inference/Translation:
   ```bash
   python scripts/translate.py --input "a-na-ku"
   ```

## Features

- Custom SentencePiece tokenization
- Multi-head attention Transformer Encoder-Decoder architecture
- Mixed precision training for Kaggle T4 GPUs
- Robust checkpointing system for 9-hour Kaggle session limits (resume from latest)
- Beam search generation

## Configuration

Edit `configs/train.yaml` to modify hyperparameters for training, architectures, and inference.

## License

MIT
