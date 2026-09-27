import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch
from torch.cuda.amp import autocast, GradScaler
from time import time, strftime, localtime
import sys
import os
import glob
import math
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from configs import Config
from src.utils import Logger, calc_loss_batch, calc_loss_loader
from src.dataset import get_dataloader
from model.transformer_model import TransformerModel

def log_batch_results_to_txt(file_path, epoch, total_epochs, batch_idx, total_batches,
                             step, train_loss, ce_loss, aux_loss, lr, elapsed_s):
    """
    Appends training results to a formatted text file every several batches.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    ts = strftime("%Y-%m-%d %H:%M:%S", localtime())
    pct = (batch_idx / total_batches) * 100.0
    line = (
        f"[{ts}] Epoch {epoch}/{total_epochs} | "
        f"Batch {batch_idx:06d}/{total_batches:06d} ({pct:5.1f}%) | "
        f"Step {step:07d} | "
        f"Loss: {train_loss:7.4f} | "
        f"CE_Loss: {ce_loss:7.4f} | "
        f"MoE_Aux: {aux_loss:6.4f} | "
        f"LR: {lr:9.7f} | "
        f"Time: {elapsed_s:6.1f}s"
    )
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(line + "\n")

def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, step, best_val_loss, log):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'epoch': epoch,
        'step': step,
        'best_val_loss': best_val_loss,
        'config': {
            'vocab_size': Config.VOCAB_SIZE,
            'model_dim': Config.MODEL_DIM,
            'n_heads': Config.N_HEADS,
            'context_length': Config.CONTEXT_LENGTH,
            'ff_dim': Config.FF_DIM,
            'n_encoders': Config.N_ENCODERS,
            'n_decoders': Config.N_DECODERS,
            'num_experts': Config.NUM_EXPERTS,
            'top_k': Config.TOP_K_EXPERTS,
        }
    }
    torch.save(checkpoint, path)
    log.print(f"  >>> Checkpoint saved to: {path}")

def find_latest_checkpoint():
    """
    Finds the most recent checkpoint file for automatic resume.
    """
    if os.path.exists(Config.CHECKPOINT_LATEST):
        return Config.CHECKPOINT_LATEST
    if os.path.exists(Config.CHECKPOINT_BEST):
        return Config.CHECKPOINT_BEST

    pattern = os.path.join(Config.CHECKPOINT_DIR, "model_epoch_*.pt")
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]

def train_model(model, train_path, val_loader, optimizer, scheduler, scaler,
                device, epochs, start_epoch, start_step, best_val_loss,
                eval_freq, log):
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    step = start_step

    # Header for training results text file
    if not os.path.exists(Config.TRAINING_RESULTS_TXT):
        with open(Config.TRAINING_RESULTS_TXT, 'w', encoding='utf-8') as f:
            f.write("=" * 115 + "\n")
            f.write(f"150M MoE Transformer Accelerated Training Log (5 Experts, Top-2 Routing)\n")
            f.write(f"Initialized at: {strftime('%Y-%m-%d %H:%M:%S', localtime())}\n")
            f.write("=" * 115 + "\n")

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loader = get_dataloader(train_path, Config, shuffle=True, is_train=True, epoch=epoch)
        total_batches = len(train_loader)

        # Milestone batches for 25%, 50%, 75%, 100% of epoch
        milestones = {
            max(1, int(0.25 * total_batches)): "25pct",
            max(1, int(0.50 * total_batches)): "50pct",
            max(1, int(0.75 * total_batches)): "75pct",
            total_batches: "100pct"
        }
        log.print(f"\n--- Starting Epoch {epoch + 1}/{epochs} (Total Batches: {total_batches:,}) ---")
        active_weights = train_loader.dataset.task_weights
        log.print(f"Task Weights: EN->VI: {active_weights[0]*100:.0f}%, VI->EN: {active_weights[1]*100:.0f}%, VI->VI: {active_weights[2]*100:.0f}%, EN->EN: {active_weights[3]*100:.0f}%")
        log.print(f"Logging metrics every {Config.LOG_FREQ_BATCHES} batches to: {Config.TRAINING_RESULTS_TXT}")

        epoch_start_time = time()
        running_loss = 0.0
        running_ce = 0.0
        running_aux = 0.0
        batch_count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (src_batch, tar_batch) in enumerate(pbar, start=1):
            with torch.amp.autocast('cuda', dtype=Config.DTYPE):
                loss, ce_loss, aux_loss = calc_loss_batch(
                    model, src_batch, tar_batch, device,
                    aux_loss_coef=Config.AUX_LOSS_COEF,
                    label_smoothing=Config.LABEL_SMOOTHING
                )
                if Config.GRAD_ACCUM_STEPS > 1:
                    loss = loss / Config.GRAD_ACCUM_STEPS

            if scaler and scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running_loss += loss.item() * (Config.GRAD_ACCUM_STEPS if Config.GRAD_ACCUM_STEPS > 1 else 1)
            running_ce += ce_loss.item()
            running_aux += aux_loss.item()
            batch_count += 1

            if batch_idx % Config.GRAD_ACCUM_STEPS == 0 or batch_idx == total_batches:
                if scaler and scaler.is_enabled():
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=Config.MAX_GRAD_NORM)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=Config.MAX_GRAD_NORM)
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                if scheduler:
                    scheduler.step()
                step += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{running_loss / batch_count:.4f}",
                'ce': f"{running_ce / batch_count:.4f}",
                'aux': f"{running_aux / batch_count:.2f}"
            })

            # Requirement: Write results to txt file every several batches
            if batch_idx % Config.LOG_FREQ_BATCHES == 0 or batch_idx == total_batches:
                elapsed_s = time() - epoch_start_time
                current_lr = optimizer.param_groups[0]['lr']
                log_batch_results_to_txt(
                    file_path=Config.TRAINING_RESULTS_TXT,
                    epoch=epoch + 1,
                    total_epochs=epochs,
                    batch_idx=batch_idx,
                    total_batches=total_batches,
                    step=step,
                    train_loss=running_loss / batch_count,
                    ce_loss=running_ce / batch_count,
                    aux_loss=running_aux / batch_count,
                    lr=current_lr,
                    elapsed_s=elapsed_s
                )

            # Checkpoint at 25%, 50%, 75%, and 100% of epoch
            if batch_idx in milestones:
                pct_tag = milestones[batch_idx]
                log.print(f"\n[Milestone reached: Epoch {epoch + 1} - {pct_tag} ({batch_idx}/{total_batches} batches)]")

                # Fast validation evaluation at milestone
                val_loss, val_ce = calc_loss_loader(val_loader, model, device, num_batches=10, aux_loss_coef=Config.AUX_LOSS_COEF)
                log.print(f"  Milestone Eval -> Val Loss: {val_loss:.4f} | Val CE: {val_ce:.4f}")

                with open(Config.TRAINING_RESULTS_TXT, 'a', encoding='utf-8') as f:
                    f.write(f"--- MILESTONE: Epoch {epoch + 1} ({pct_tag}) | Val Loss: {val_loss:.4f} | Val CE: {val_ce:.4f} ---\n")

                # Save milestone checkpoint
                milestone_ckpt_path = os.path.join(
                    Config.CHECKPOINT_DIR, f"model_epoch_{epoch + 1}_{pct_tag}.pt"
                )
                save_epoch_idx = (epoch + 1) if pct_tag == "100pct" else epoch
                save_checkpoint(milestone_ckpt_path, model, optimizer, scheduler, scaler, save_epoch_idx, step, best_val_loss, log)

                # Check and save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(Config.CHECKPOINT_BEST, model, optimizer, scheduler, scaler, save_epoch_idx, step, best_val_loss, log)
                    log.print(f"  ★ NEW BEST MODEL SAVED! (Val loss: {best_val_loss:.4f})")

                save_checkpoint(Config.CHECKPOINT_LATEST, model, optimizer, scheduler, scaler, save_epoch_idx, step, best_val_loss, log)
                model.train()

            # Periodic validation and logging
            if step > 0 and step % eval_freq == 0 and batch_idx % Config.GRAD_ACCUM_STEPS == 0:
                val_loss, val_ce = calc_loss_loader(val_loader, model, device, num_batches=8, aux_loss_coef=Config.AUX_LOSS_COEF)
                elapsed = time() - epoch_start_time
                current_lr = optimizer.param_groups[0]['lr']
                log.print(f"Step {step:07d} (Ep {epoch+1} {batch_idx}/{total_batches}) | Train: {running_loss/batch_count:.4f} | Val: {val_loss:.4f} | LR: {current_lr:.6f} | Time: {elapsed:.1f}s")
                save_checkpoint(Config.CHECKPOINT_LATEST, model, optimizer, scheduler, scaler, epoch, step, best_val_loss, log)
                model.train()

        epoch_time = time() - epoch_start_time
        log.print(f"\n=======================================================")
        log.print(f"Completed Epoch {epoch + 1}/{epochs} in {epoch_time:.2f}s | Avg Train Loss: {running_loss / batch_count:.4f}")
        log.print(f"=======================================================\n")

        save_checkpoint(os.path.join(Config.CHECKPOINT_DIR, f"model_epoch_{epoch + 1}_100pct.pt"),
                        model, optimizer, scheduler, scaler, epoch + 1, step, best_val_loss, log)
        save_checkpoint(Config.CHECKPOINT_LATEST, model, optimizer, scheduler, scaler, epoch + 1, step, best_val_loss, log)

if __name__ == '__main__':
    train_path = f'{Config.PROCESSED_DATA_PATH}train_ids.parquet'
    val_path = f'{Config.PROCESSED_DATA_PATH}val_ids.parquet'

    val_loader = get_dataloader(val_path, Config, shuffle=False, is_train=False)

    logger = Logger(Config.CHECKPOINT_LOG)
    logger.print("=" * 65)
    logger.print("  Accelerated 150M MoE Transformer (FlashAttention + Fused AdamW)")
    logger.print("=" * 65)

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
        dropout_rate=Config.DROPOUT_RATE,
        pad_id=0,
        tie_embeddings=Config.TIE_EMBEDDINGS,
        device=Config.DEVICE
    ).to(Config.DEVICE)

    total_params, trainable_params = model.count_parameters()
    logger.print(f"Model Parameters: {total_params:,} ({total_params / 1e6:.2f}M)")
    logger.print(f"Batch Size: {Config.BATCH_SIZE} | Precision: {Config.DTYPE}")

    # Use fused AdamW on CUDA for significant GPU kernel speedup
    use_fused = (Config.DEVICE == 'cuda')
    try:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            betas=(0.9, 0.98),
            eps=1e-8,
            weight_decay=Config.WEIGHT_DECAY,
            fused=use_fused
        )
    except Exception:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            betas=(0.9, 0.98),
            eps=1e-8,
            weight_decay=Config.WEIGHT_DECAY
        )

    # Calculate steps based on configured samples per epoch
    samples_per_epoch = Config.MAX_TRAIN_SAMPLES_PER_EPOCH or 2884451
    batches_per_epoch = samples_per_epoch // Config.BATCH_SIZE
    total_steps = (batches_per_epoch // Config.GRAD_ACCUM_STEPS) * max(1, Config.EPOCHS)

    # Cosine learning rate schedule with linear warmup
    def lr_lambda(current_step):
        if current_step < Config.WARMUP_STEPS:
            return float(current_step) / float(max(1, Config.WARMUP_STEPS))
        progress = float(current_step - Config.WARMUP_STEPS) / float(max(1, total_steps - Config.WARMUP_STEPS))
        return max(Config.MIN_LEARNING_RATE / Config.LEARNING_RATE, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    # Scaler only needed for FP16, not for native BF16
    use_scaler = (Config.DTYPE == torch.float16 and Config.DEVICE == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)

    start_step = 0
    start_epoch = 0
    best_val_loss = float('inf')

    # Automatic Autoload of Previous Epoch Checkpoints
    latest_ckpt_path = find_latest_checkpoint()
    if latest_ckpt_path and os.path.exists(latest_ckpt_path):
        logger.print(f"Autoloading previous checkpoint from: {latest_ckpt_path}")
        ckpt = torch.load(latest_ckpt_path, map_location=Config.DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            try:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            except Exception as e:
                logger.print(f"Warning loading optimizer state: {e}")
        if 'scheduler_state_dict' in ckpt and ckpt['scheduler_state_dict']:
            try:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            except Exception as e:
                logger.print(f"Warning loading scheduler state: {e}")
        if 'scaler_state_dict' in ckpt and ckpt['scaler_state_dict'] and use_scaler:
            try:
                scaler.load_state_dict(ckpt['scaler_state_dict'])
            except Exception as e:
                logger.print(f"Warning loading scaler state: {e}")

        start_step = ckpt.get('step', 0)
        start_epoch = ckpt.get('epoch', 0)
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        logger.print(f"[✓] Successfully autoloaded: Resumed from Epoch {start_epoch + 1}, Step {start_step}, Best Val Loss: {best_val_loss:.4f}")
    else:
        logger.print("No previous checkpoint found. Starting training from scratch (Epoch 1).")

    # Check if requested epochs are already completed
    if start_epoch >= Config.EPOCHS:
        logger.print("\n" + "=" * 65)
        logger.print(f"[INFO] Model has already completed {start_epoch} epoch(s).")
        logger.print(f"Current Config.EPOCHS is set to {Config.EPOCHS}.")
        logger.print(f"To continue training further:")
        logger.print(f"  1. Open configs/config.py")
        logger.print(f"  2. Increase EPOCHS (e.g. EPOCHS = {start_epoch + 1})")
        logger.print(f"  3. Re-run 'python train.py'")
        logger.print("The model will automatically autoload the parameters from epoch "
                     f"{start_epoch} and continue training.")
        logger.print("=" * 65)
    else:
        logger.print(f"Training will run from Epoch {start_epoch + 1} to Epoch {Config.EPOCHS}...")
        train_model(
            model=model,
            train_path=train_path,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=Config.DEVICE,
            epochs=Config.EPOCHS,
            start_epoch=start_epoch,
            start_step=start_step,
            best_val_loss=best_val_loss,
            eval_freq=Config.EVAL_FREQ_STEPS,
            log=logger
        )
