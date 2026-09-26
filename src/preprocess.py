import os
import sys
import pandas as pd
from tqdm import tqdm
from tokenizers import Tokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from configs import Config

def preprocess_files(file_paths, save_path, tokenizer, max_chunk=100000, max_samples=None):
    print(f"\n--- Preprocessing {file_paths} -> {save_path} ---")
    df_list = []
    for f in file_paths:
        if os.path.exists(f):
            print(f"Reading {f}...")
            df_part = pd.read_parquet(f, columns=['en', 'vi'])
            df_list.append(df_part)
        else:
            print(f"Warning: File {f} does not exist.")

    if not df_list:
        print(f"No files found for {save_path}, skipping.")
        return

    df = pd.concat(df_list, ignore_index=True)
    df = df.dropna(subset=['en', 'vi'])
    df = df[(df['en'].str.len() > 1) & (df['vi'].str.len() > 1)]

    if max_samples and len(df) > max_samples:
        df = df.iloc[:max_samples]

    print(f"Total valid samples to encode: {len(df):,}")

    processed_chunks = []
    for i in tqdm(range(0, len(df), max_chunk), desc=f"Encoding -> {os.path.basename(save_path)}"):
        chunk = df.iloc[i : i + max_chunk]

        src_texts = chunk['vi'].astype(str).tolist()
        tar_texts = chunk['en'].astype(str).tolist()

        # Batch encode with Rust multi-threading
        vi_encodings = tokenizer.encode_batch(src_texts)
        en_encodings = tokenizer.encode_batch(tar_texts)

        vi_ids = [e.ids for e in vi_encodings]
        en_ids = [e.ids for e in en_encodings]

        chunk_df = pd.DataFrame({
            'vi_ids': vi_ids,
            'en_ids': en_ids
        })
        processed_chunks.append(chunk_df)

    final_df = pd.concat(processed_chunks, ignore_index=True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    final_df.to_parquet(save_path, index=False)
    print(f"[✓] Saved {len(final_df):,} preprocessed samples to {save_path} ({os.path.getsize(save_path)/(1024*1024):.2f} MB)")

if __name__ == '__main__':
    tokenizer_path = Config.TOKENIZER_JSON if os.path.exists(Config.TOKENIZER_JSON) else Config.TOKENIZER_PATH
    tokenizer = Tokenizer.from_file(tokenizer_path)

    # Preprocess validation set
    val_raw = [os.path.join(Config.RAW_DATA_PATH, 'val.parquet')]
    val_proc = os.path.join(Config.PROCESSED_DATA_PATH, 'val_ids.parquet')
    preprocess_files(val_raw, val_proc, tokenizer)

    # Preprocess test set
    test_raw = [os.path.join(Config.RAW_DATA_PATH, 'test.parquet')]
    test_proc = os.path.join(Config.PROCESSED_DATA_PATH, 'test_ids.parquet')
    preprocess_files(test_raw, test_proc, tokenizer)

    # Preprocess training set (train_0 and train_1)
    train_raw = [
        os.path.join(Config.RAW_DATA_PATH, 'train_0.parquet'),
        os.path.join(Config.RAW_DATA_PATH, 'train_1.parquet')
    ]
    train_proc = os.path.join(Config.PROCESSED_DATA_PATH, 'train_ids.parquet')
    preprocess_files(train_raw, train_proc, tokenizer)

    print("\nData preprocessing finished successfully!")