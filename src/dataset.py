import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tokenizers import Tokenizer
import random
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from configs import Config

class MultiTaskTranslationDataset(Dataset):
    """
    Multi-task Seq2Seq Dataset supporting:
    1. <sos> <en> {en} <eos> -> <sos> <vi> {vi} <eos> (EN -> VI)
    2. <sos> <vi> {vi} <eos> -> <sos> <en> {en} <eos> (VI -> EN)
    3. <sos> <vi> {vi} <eos> -> <sos> <vi> {vi} <eos> (VI -> VI autoencoding)
    4. <sos> <en> {en} <eos> -> <sos> <en> {en} <eos> (EN -> EN autoencoding)
    """
    def __init__(self, file_path, tokenizer_path, context_length=256, is_train=True,
                 max_samples=None, epoch_offset=0, task_weights=(0.35, 0.35, 0.15, 0.15)):
        df = pd.read_parquet(file_path)
        all_vi = df['vi_ids'].tolist()
        all_en = df['en_ids'].tolist()

        if max_samples and max_samples < len(all_vi):
            # Select slice for current epoch so each epoch sees fresh data
            total = len(all_vi)
            start_i = (epoch_offset * max_samples) % total
            end_i = start_i + max_samples
            if end_i <= total:
                self.vi_data = all_vi[start_i:end_i]
                self.en_data = all_en[start_i:end_i]
            else:
                self.vi_data = all_vi[start_i:] + all_vi[:end_i - total]
                self.en_data = all_en[start_i:] + all_en[:end_i - total]
        else:
            self.vi_data = all_vi
            self.en_data = all_en

        self.context_length = context_length
        self.is_train = is_train
        self.task_weights = task_weights

        tokenizer = Tokenizer.from_file(tokenizer_path)
        self.pad_id = tokenizer.token_to_id(Config.PAD_TOKEN)
        self.sos_id = tokenizer.token_to_id(Config.SOS_TOKEN)
        self.eos_id = tokenizer.token_to_id(Config.EOS_TOKEN)
        self.en_id = tokenizer.token_to_id(Config.EN_TOKEN)
        self.vi_id = tokenizer.token_to_id(Config.VI_TOKEN)

    def __len__(self):
        return len(self.vi_data)

    def __getitem__(self, idx):
        vi_tokens = self.vi_data[idx]
        en_tokens = self.en_data[idx]

        if self.is_train:
            task = random.choices([0, 1, 2, 3], weights=self.task_weights, k=1)[0]
        else:
            task = idx % 4

        max_content = self.context_length - 3

        if task == 0:
            # 1. EN -> VI Translation
            src = [self.sos_id, self.en_id] + list(en_tokens[:max_content]) + [self.eos_id]
            tar = [self.sos_id, self.vi_id] + list(vi_tokens[:max_content]) + [self.eos_id]
        elif task == 1:
            # 2. VI -> EN Translation
            src = [self.sos_id, self.vi_id] + list(vi_tokens[:max_content]) + [self.eos_id]
            tar = [self.sos_id, self.en_id] + list(en_tokens[:max_content]) + [self.eos_id]
        elif task == 2:
            # 3. VI -> VI Autoencoding / Reconstruction
            src = [self.sos_id, self.vi_id] + list(vi_tokens[:max_content]) + [self.eos_id]
            tar = [self.sos_id, self.vi_id] + list(vi_tokens[:max_content]) + [self.eos_id]
        else:
            # 4. EN -> EN Autoencoding / Reconstruction
            src = [self.sos_id, self.en_id] + list(en_tokens[:max_content]) + [self.eos_id]
            tar = [self.sos_id, self.en_id] + list(en_tokens[:max_content]) + [self.eos_id]

        return torch.tensor(src, dtype=torch.long), torch.tensor(tar, dtype=torch.long)

def collate_fn(batch, pad_id=0):
    src_list, tar_list = [], []
    for item in batch:
        src_list.append(item[0])
        tar_list.append(item[1])

    src_padded = torch.nn.utils.rnn.pad_sequence(src_list, batch_first=True, padding_value=pad_id)
    tar_padded = torch.nn.utils.rnn.pad_sequence(tar_list, batch_first=True, padding_value=pad_id)

    return src_padded, tar_padded

def get_dataloader(file_path, config, shuffle=True, is_train=True, epoch=0):
    tok_path = config.TOKENIZER_JSON if os.path.exists(config.TOKENIZER_JSON) else config.TOKENIZER_PATH
    max_samples = config.MAX_TRAIN_SAMPLES_PER_EPOCH if is_train else None
    dataset = MultiTaskTranslationDataset(
        file_path,
        tok_path,
        context_length=config.CONTEXT_LENGTH,
        is_train=is_train,
        max_samples=max_samples,
        epoch_offset=epoch
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=lambda x: collate_fn(x, dataset.pad_id)
    )
    return dataloader