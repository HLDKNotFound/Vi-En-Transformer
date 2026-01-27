import torch

from time import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from configs.config import Config
from src.utils import Logger, calc_loss_batch, calc_loss_loader
from src.dataset import get_dataloader
from model.transformer_model import TransformerModel

def training_model(model, train_loader, val_loader,
                   optimizer, scheduler, device, epochs, 
                   start_epoch, start_step, best_val_loss,
                   eval_freq, save_path, log):
    train_losses = []
    val_losses = []
    step = start_step

    for epoch in range(start_epoch, epochs):
        model.train()
        
        start_time = time()
        for src_batch, tar_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(model, src_batch, tar_batch, device)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            step += 1
            if step % eval_freq == 0:
                model.eval()
                with torch.inference_mode():
                    train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
                    val_loss = calc_loss_loader(val_loader, model, device)

                    train_losses.append(train_loss)
                    val_losses.append(val_loss)

                    elapsed = time() - start_time
                    log.print(f"Ep {epoch + 1} (Step {step:07d}) | Train loss {train_loss:.5f} | Val loss {val_loss:.5f} | Time {elapsed:.2f}s")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint = {
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'step': step,
                        'epoch': epoch,
                        'best_val_loss': best_val_loss # Lưu kèm để khi resume không bị reset
                    }
                    torch.save(checkpoint, save_path)
                    log.print(f"--- Model saved (Best loss: {best_val_loss:.5f}) ---")

                start_time = time()
                model.train()

if __name__ == '__main__':
    # READ AND LOAD DATA
    train_path = f'{Config.PROCESSED_DATA_PATH}train_ids.parquet'
    val_path = f'{Config.PROCESSED_DATA_PATH}val_ids.parquet'

    train_loader = get_dataloader(train_path, Config)
    val_loader = get_dataloader(val_path, Config)

    print(f'len train data loader: {len(train_loader)}, len val data loader: {len(val_loader)}')

    # Definition
    logger = Logger(Config.CHECKPOINT_LOG)
    model = TransformerModel(Config.VOCAB_SIZE, 
                             Config.MODEL_DIM, 
                             Config.N_HEADS,
                             Config.CONTEXT_LENGTH, 
                             Config.FF_DIM,
                             Config.N_ENCODERS, 
                             Config.N_DECODERS,
                             dropout_rate=Config.DROPOUT_RATE, 
                             device=Config.DEVICE)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=Config.WEIGHT_DECAY,
        foreach=False
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(
            (step + 1) ** -0.5,
            (step + 1) * (Config.WARMUP_STEP ** -1.5)
        )
    )

    start_step = 0
    start_epoch = 0
    best_val_loss = float('inf')
    if os.path.exists(Config.CHECKPOINT_MODEL):
        checkpoint = torch.load(Config.CHECKPOINT_MODEL, map_location=Config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=Config.DEVICE, dtype=Config.DTYPE)

        start_step = checkpoint.get('step', 0)
        start_epoch = checkpoint.get('epoch', -1) + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f'Resumed from step {start_step}')
    else:
        print('Start training model')
    
    model = model.to(device=Config.DEVICE, dtype=Config.DTYPE)
    
    total_params = sum(param.numel() for param in model.parameters())
    print(f'total number of parameters: {total_params}')

    training_model(model=model,
                   train_loader=train_loader, 
                   val_loader=val_loader, 
                   optimizer=optimizer, 
                   scheduler=scheduler, 
                   device=Config.DEVICE,
                   epochs=Config.EPOCHS,
                   start_step=start_step,
                   start_epoch=start_epoch,
                   best_val_loss=best_val_loss,
                   eval_freq=Config.EVAL_FREQ, 
                   save_path=Config.CHECKPOINT_MODEL, 
                   log=logger)
