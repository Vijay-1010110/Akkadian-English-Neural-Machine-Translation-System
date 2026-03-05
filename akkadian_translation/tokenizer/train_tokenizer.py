import sentencepiece as spm
import os
import argparse
import pandas as pd
from data.preprocess import normalize_akkadian, preprocess_english

def train_spm(input_file, model_prefix, vocab_size=12000, character_coverage=1.0, model_type="unigram"):
    """
    Trains a SentencePiece model on a raw text file containing all sentences.
    
    Args:
        input_file (str): Path to raw text file containing sentences (one per line).
        model_prefix (str): Prefix of the output model and vocab files.
        vocab_size (int): Size of the vocabulary.
        character_coverage (float): Character coverage (1.0 for languages with rich characters).
        model_type (str): Tokenizer model type (unigram, bpe, char, or word).
    """
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        model_type=model_type,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",
        user_defined_symbols=["<DIVINE>", "<MISSING>"]
    )
    print(f"SentencePiece model saved to {model_prefix}.model and {model_prefix}.vocab")

def prepare_text_file(csv_path, output_txt_path):
    """
    Extracts texts from the datasets, normalizes them, and saves to a plain text file 
    for SentencePiece to train on. Both source and target languages are mixed 
    to create a shared vocabulary.
    """
    print(f"Reading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Assume 1st column is Akkadian, 2nd column is English
    akk_col = df.columns[0] if 'akkadian' not in df.columns else 'akkadian'
    eng_col = df.columns[1] if 'english' not in df.columns else 'english'
    
    df = df.dropna(subset=[akk_col, eng_col])
    
    sentences = []
    
    for _, row in df.iterrows():
        akk_text = normalize_akkadian(str(row[akk_col]))
        eng_text = preprocess_english(str(row[eng_col]))
        
        if akk_text:
            sentences.append(akk_text)
        if eng_text:
            sentences.append(eng_text)
            
    # Write to a plain text file
    with open(output_txt_path, "w", encoding="utf-8") as f:
        for sentence in sentences:
            f.write(sentence + "\n")
            
    print(f"Wrote {len(sentences)} lines to {output_txt_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", type=str, required=True, help="Path to the training CSV data")
    parser.add_argument("--model_prefix", type=str, default="tokenizer/tokenizer", help="Prefix for the saved model")
    parser.add_argument("--vocab_size", type=int, default=12000, help="Vocabulary size")
    
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.model_prefix), exist_ok=True)
    
    temp_txt_path = "temp_spm_train.txt"
    prepare_text_file(args.data_csv, temp_txt_path)
    
    print("Training SentencePiece model...")
    # Per instructions: unigram, 12000 vocab size, 1.0 character coverage
    train_spm(
        input_file=temp_txt_path,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        character_coverage=1.0,
        model_type="unigram"
    )
    
    # Cleanup temp file
    if os.path.exists(temp_txt_path):
        os.remove(temp_txt_path)
