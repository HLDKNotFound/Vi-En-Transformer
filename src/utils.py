import torch
import torch.nn.functional as F
import os

class Logger(object):
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def print(self, msg):
        print(msg)
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(msg + "\n")

def calc_loss_batch(model, src_batch, tar_batch, device, aux_loss_coef=0.01, label_smoothing=0.1):
    src_batch = src_batch.to(device, non_blocking=True)
    tar_batch = tar_batch.to(device, non_blocking=True)

    # Decoder inputs: tokens from index 0 to -2
    tar_input = tar_batch[:, :-1]
    # Prediction targets: tokens from index 1 to end
    tar_target = tar_batch[:, 1:].contiguous()

    logits, aux_loss = model(src_batch, tar_input)

    ce_loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        tar_target.view(-1),
        ignore_index=0,  # PAD token ID
        label_smoothing=label_smoothing
    )

    total_loss = ce_loss + aux_loss_coef * aux_loss
    return total_loss, ce_loss, aux_loss

def calc_loss_loader(data_loader, model, device, num_batches=None, aux_loss_coef=0.01):
    model.eval()
    if len(data_loader) == 0:
        return float("nan"), float("nan")

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    total_loss = 0.0
    total_ce = 0.0

    with torch.inference_mode():
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i >= num_batches:
                break
            loss, ce, _ = calc_loss_batch(
                model, input_batch, target_batch, device,
                aux_loss_coef=aux_loss_coef, label_smoothing=0.0
            )
            total_loss += loss.item()
            total_ce += ce.item()

    return total_loss / num_batches, total_ce / num_batches