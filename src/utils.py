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

def chunked_cross_entropy(logits, targets, ignore_index=0, chunk_size=2048, label_smoothing=0.1):
    """
    Computes cross entropy in memory-safe chunks over the token dimension.
    Prevents CUDA OOM on laptop GPUs (e.g. RTX 4050 6GB) when vocabulary is large (32,000).
    """
    flat_logits = logits.view(-1, logits.size(-1))
    flat_targets = targets.view(-1)

    total_tokens = (flat_targets != ignore_index).sum()
    if total_tokens == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    total_loss = 0.0
    for i in range(0, flat_logits.size(0), chunk_size):
        c_logits = flat_logits[i:i + chunk_size]
        c_targets = flat_targets[i:i + chunk_size]

        c_valid = (c_targets != ignore_index).sum()
        if c_valid == 0:
            continue

        c_loss = F.cross_entropy(
            c_logits, c_targets,
            ignore_index=ignore_index,
            reduction='sum',
            label_smoothing=label_smoothing
        )
        total_loss = total_loss + c_loss

    return total_loss / total_tokens

def calc_loss_batch(model, src_batch, tar_batch, device, aux_loss_coef=0.01, label_smoothing=0.1):
    src_batch = src_batch.to(device, non_blocking=True)
    tar_batch = tar_batch.to(device, non_blocking=True)

    tar_input = tar_batch[:, :-1]
    tar_target = tar_batch[:, 1:].contiguous()

    logits, aux_loss = model(src_batch, tar_input)

    ce_loss = chunked_cross_entropy(
        logits, tar_target,
        ignore_index=0,  # PAD token ID
        chunk_size=2048,
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