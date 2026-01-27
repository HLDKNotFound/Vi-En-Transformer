import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tokenizers import Tokenizer

class TranslationDataset(Dataset):

    def __init__(self, file_path, tokenizer_path, context_length):
        # Load Data
        df = pd.read_parquet(file_path)

        self.src_data = df['vi_ids'].tolist()
        self.tar_data = df['en_ids'].tolist()

        # Load Tokenizer
        tokenizer = Tokenizer.from_file(tokenizer_path)
        self.context_length = context_length

        # Get special ID
        self.pad_id = tokenizer.token_to_id("[PAD]")
        self.cls_id = tokenizer.token_to_id("[CLS]")
        self.sep_id = tokenizer.token_to_id("[SEP]")

    def __len__(self):
        return len(self.src_data)
    
    def __getitem__(self, idx):
        # Read and convert text to ids
        src_ids = self.src_data[idx]
        tar_ids = self.tar_data[idx]
        
        src_ids = [self.cls_id] + list(src_ids[:self.context_length - 2]) + [self.sep_id]
        tar_ids = [self.cls_id] + list(tar_ids[:self.context_length - 2]) + [self.sep_id]

        src_ids = torch.tensor(src_ids, dtype=torch.long)
        tar_ids = torch.tensor(tar_ids, dtype=torch.long)

        return src_ids, tar_ids
    
def collate_fn(batch, pad_id):
    src_list, tar_list = [], []
    for item in batch:
        src_list.append(item[0])
        tar_list.append(item[1]) 

    # Pad to the longest sentence in the batch
    src_padded = torch.nn.utils.rnn.pad_sequence(src_list, batch_first=True, padding_value=pad_id)
    tar_padded = torch.nn.utils.rnn.pad_sequence(tar_list, batch_first=True, padding_value=pad_id)

    return src_padded, tar_padded
    
def get_dataloader(file_path, config, shuffle=True):

    dataset = TranslationDataset(file_path,
                                 config.TOKENIZER_PATH,
                                 config.CONTEXT_LENGTH)
    
    dataloader = DataLoader(dataset=dataset,
                            batch_size=config.BATCH_SIZE,
                            shuffle=shuffle,
                            collate_fn=lambda x: collate_fn(x, dataset.pad_id))
    
    return dataloader