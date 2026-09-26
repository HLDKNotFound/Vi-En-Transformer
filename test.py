import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from configs import Config
from src.utils import calc_loss_loader
from src.dataset import get_dataloader
from model.transformer_model import TransformerModel

if __name__ == '__main__':
    test_path = f'{Config.PROCESSED_DATA_PATH}test_ids.parquet'
    test_loader = get_dataloader(test_path, Config, shuffle=False, is_train=False)

    print(f'Test dataset batches: {len(test_loader)}')

    model = TransformerModel(
        vocab_size=Config.VOCAB_SIZE,
        model_dim=Config.MODEL_DIM,
        n_heads=Config.N_HEADS,
        context_length=Config.CONTEXT_LENGTH,
        ff_dim=Config.FF_DIM,
        n_encoders=Config.N_ENCODERS,
        n_decoders=Config.N_DECODERS,
        num_experts=Config.NUM_EXPERTS,
        top_k=Config.TOP_K_EXPERTS,
        dropout_rate=0.0,
        pad_id=0,
        tie_embeddings=Config.TIE_EMBEDDINGS,
        device=Config.DEVICE
    ).to(Config.DEVICE)

    total_params, trainable_params = model.count_parameters()
    print(f'Total parameters: {total_params:,} ({total_params / 1e6:.2f}M)')

    ckpt_path = Config.CHECKPOINT_BEST if os.path.exists(Config.CHECKPOINT_BEST) else Config.CHECKPOINT_LATEST
    if os.path.exists(ckpt_path):
        print(f'Loading checkpoint: {ckpt_path}')
        checkpoint = torch.load(ckpt_path, map_location=Config.DEVICE)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        print(f"Resumed from step {checkpoint.get('step', 'N/A')}")
    else:
        print(f'No checkpoint found at {ckpt_path}. Evaluating with initial weights.')

    model.eval()
    with torch.inference_mode():
        test_loss, test_ce = calc_loss_loader(test_loader, model, Config.DEVICE, num_batches=20, aux_loss_coef=Config.AUX_LOSS_COEF)
        print(f"\nEvaluation Results:")
        print(f"  Test Total Loss: {test_loss:.5f}")
        print(f"  Test CE Loss:    {test_ce:.5f}")