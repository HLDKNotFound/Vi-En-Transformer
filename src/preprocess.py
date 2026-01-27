import pandas as pd
from tokenizers import Tokenizer

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from configs.config import Config

def preprocess_txt(file_paths, save_path, tokenizer, max_chunk=100000):

    # Load Data
    df_list = [pd.read_parquet(f) for f in file_paths]
    df = pd.concat(df_list, ignore_index=True)
    
    processed_chunks = []
    for i in range(0, len(df), max_chunk):
        chunk = df.iloc[i : i + max_chunk].copy()
            
        src_encodings = tokenizer.encode_batch(chunk['vi'].tolist())
        tar_encodings = tokenizer.encode_batch(chunk['en'].tolist())

        chunk['vi_ids'] = [e.ids for e in src_encodings]
        chunk['en_ids'] = [e.ids for e in tar_encodings]

        processed_chunks.append(chunk[['vi_ids', 'en_ids']])

    final_df = pd.concat(processed_chunks, ignore_index=True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    final_df.to_parquet(save_path, index=False)

if __name__ == '__main__':
    tokenizer = Tokenizer.from_file(Config.TOKENIZER_PATH)

    print('Start save train data')
    train_paths = [f'{Config.RAW_DATA_PATH}train.parquet']
    train_save_path = f'{Config.PROCESSED_DATA_PATH}train_ids.parquet'

    preprocess_txt(train_paths, train_save_path, tokenizer)
    print(f'Train indices are save to {train_save_path}')

    print('Start save val data')
    val_paths = [f'{Config.RAW_DATA_PATH}val.parquet']
    val_save_path = f'{Config.PROCESSED_DATA_PATH}val_ids.parquet'

    preprocess_txt(val_paths, val_save_path, tokenizer)
    print(f'Train indices are save to {val_save_path}')

    print('Start save test data')
    test_paths = [f'{Config.RAW_DATA_PATH}test.parquet']
    test_save_path = f'{Config.PROCESSED_DATA_PATH}test_ids.parquet'

    preprocess_txt(test_paths, test_save_path, tokenizer)
    print(f'Train indices are save to {test_save_path}')