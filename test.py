import torch

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from configs.config import Config
from src.utils import calc_loss_loader
from src.dataset import get_dataloader
from model.transformer_model import TransformerModel

if __name__ == '__main__':
    test_path = f'{Config.PROCESSED_DATA_PATH}test_ids.parquet'
    test_loader = get_dataloader(test_path, Config, shuffle=False)

    print(f'len train data loader: {len(test_loader)}')

    model = TransformerModel(Config.VOCAB_SIZE,
                             Config.MODEL_DIM,
                             Config.N_HEADS,
                             Config.CONTEXT_LENGTH,
                             Config.FF_DIM,
                             Config.N_ENCODERS,
                             Config.N_DECODERS,
                             dropout_rate=Config.DROPOUT_RATE,
                             device=Config.DEVICE)
    
    if os.path.exists(Config.CHECKPOINT_MODEL):
        checkpoint = torch.load(Config.CHECKPOINT_MODEL, map_location=Config.DEVICE, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(Config.DEVICE)

        total_params = sum(param.numel() for param in model.parameters())
        print(f'total number of parameters: {total_params}')

        model.eval()
        with torch.inference_mode():
            test_loss = calc_loss_loader(test_loader, model, Config.DEVICE, num_batches=5)

            print(f"Test loss {test_loss:.5f}")

    else:
        print(f'There is no checkpoint in path {Config.CHECKPOINT_MODEL}')