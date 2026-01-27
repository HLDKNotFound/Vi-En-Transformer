import pandas as pd

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from configs.config import Config

def get_training_data(file_paths, chunk_size=100000):
    # read parquet file columns = ('en', 'vi', 'source')
    for file_path in file_paths:
        df = pd.read_parquet(file_path)

        for i in range(0, len(df), chunk_size):
            chunk_en = df['en'].iloc[i:i + chunk_size].astype(str).tolist()
            chunk_vi = df['vi'].iloc[i:i + chunk_size].astype(str).tolist()
            yield chunk_en + chunk_vi

if __name__ == '__main__':
    # initialize tokenizer with model BPE
    tokenizer = Tokenizer(BPE(unk_token='[UNK]'))
    tokenizer.pre_tokenizer = Whitespace()

    # Trainer Config
    trainer = BpeTrainer(
        vocab_size=Config.VOCAB_SIZE,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        min_frequency=Config.MIN_FREQ,
        show_progress=True
    )

    # Training Tokenizer
    file_paths = [f'{Config.RAW_DATA_PATH}train_1.parquet',
                  f'{Config.RAW_DATA_PATH}train_2.parquet']
    training_data = get_training_data(file_paths)
    tokenizer.train_from_iterator(training_data, trainer=trainer)

    # Save tokens
    tokenizer.save(Config.TOKENIZER_PATH)
    print(f'The tokenizer with a vocabulary size of {Config.VOCAB_SIZE} and a minimum frequency {Config.MIN_FREQ} has been saved.')