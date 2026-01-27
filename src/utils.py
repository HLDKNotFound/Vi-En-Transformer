import torch.nn.functional as F

class Logger(object):
    def __init__(self, path):
        self.path = path
    
    def print(self, msg):
        print(msg)
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(msg + "\n")

def calc_loss_batch(model, src_batch, tar_batch, device):
    src_batch = src_batch.to(device)
    tar_batch = tar_batch.to(device)

    logits = model(src_batch, tar_batch[:, :-1])
    loss = F.cross_entropy(logits.flatten(0, 1),
                           tar_batch[:, 1:].flatten(),
                           ignore_index=0)

    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    model.eval()

    if len(data_loader) == 0:
        return float("nan")

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    total_loss = 0

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break

        loss = calc_loss_batch(model, input_batch, target_batch, device)
        total_loss += loss.item()

    return total_loss / num_batches