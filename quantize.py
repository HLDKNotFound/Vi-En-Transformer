import os
import sys
import time
import torch
import torch.nn as nn
import bitsandbytes as bnb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from configs import Config
from model.transformer_model import TransformerModel

def replace_linear_with_8bit(module, target_classes=(nn.Linear,)):
    """
    Recursively replaces nn.Linear modules with bitsandbytes Linear8bitLt.
    Leaves embeddings and final norm as standard precision for numerical stability.
    """
    for name, child in module.named_children():
        if isinstance(child, target_classes) and name not in ['tokens_emb', 'position_emb']:
            # Create 8-bit linear replacement
            has_bias = child.bias is not None
            in_features = child.in_features
            out_features = child.out_features

            linear_8bit = bnb.nn.Linear8bitLt(
                in_features,
                out_features,
                bias=has_bias,
                has_fp16_weights=False,
                threshold=6.0  # Outlier threshold for vector-wise INT8 quantization
            )

            with torch.no_grad():
                linear_8bit.weight.copy_(child.weight)
                if has_bias:
                    linear_8bit.bias.copy_(child.bias)

            setattr(module, name, linear_8bit)
        else:
            replace_linear_with_8bit(child, target_classes)
    return module

def quantize_model_int8(model):
    """
    Applies 8-bit quantization to the Transformer model.
    """
    model.eval()
    model = replace_linear_with_8bit(model)
    return model

def benchmark_quantization(checkpoint_path=None):
    device = Config.DEVICE
    print("=" * 65)
    print("  Benchmarking INT8 Quantization vs FP32/BF16 on RTX 4050")
    print("=" * 65)

    # 1. Load Original Model
    print("\n1. Loading Base Precision Model...")
    base_model = TransformerModel(
        vocab_size=Config.VOCAB_SIZE,
        model_dim=Config.MODEL_DIM,
        n_heads=Config.N_HEADS,
        context_length=Config.CONTEXT_LENGTH,
        ff_dim=Config.FF_DIM,
        n_encoders=Config.N_ENCODERS,
        n_decoders=Config.N_DECODERS,
        num_experts=Config.NUM_EXPERTS,
        top_k=Config.TOP_K_EXPERTS,
        tie_embeddings=Config.TIE_EMBEDDINGS,
        device=device
    )

    ckpt_path = checkpoint_path or Config.CHECKPOINT_BEST
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Loading checkpoint from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location='cpu')
        sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        base_model.load_state_dict(sd)

    base_model = base_model.to(device)
    base_model.eval()

    # Measure Base Memory
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    dummy_src = torch.randint(2, 1000, (4, 64), device=device)
    dummy_tar = torch.randint(2, 1000, (4, 64), device=device)

    # Cast to half precision to match Linear8bitLt expectations
    base_model = base_model.to(dtype=torch.float16)

    with torch.inference_mode():
        _ = base_model(dummy_src, dummy_tar)
    base_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # Measure Base Latency
    times = []
    with torch.inference_mode():
        for _ in range(5):
            t0 = time.time()
            _ = base_model(dummy_src, dummy_tar)
            torch.cuda.synchronize()
            times.append(time.time() - t0)
    base_latency = (sum(times) / len(times)) * 1000.0

    print(f"Base Precision (FP16):")
    print(f"  Peak VRAM: {base_vram:.1f} MB")
    print(f"  Forward Latency: {base_latency:.2f} ms")

    # 2. Apply INT8 Quantization
    print("\n2. Quantizing Model to INT8 with bitsandbytes...")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    int8_model = quantize_model_int8(base_model)
    int8_model = int8_model.to(device)

    with torch.inference_mode():
        _ = int8_model(dummy_src, dummy_tar)
    int8_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)

    int8_times = []
    with torch.inference_mode():
        for _ in range(5):
            t0 = time.time()
            _ = int8_model(dummy_src, dummy_tar)
            torch.cuda.synchronize()
            int8_times.append(time.time() - t0)
    int8_latency = (sum(int8_times) / len(int8_times)) * 1000.0

    print(f"INT8 Quantized Model:")
    print(f"  Peak VRAM: {int8_vram:.1f} MB ({(1.0 - int8_vram / base_vram) * 100:.1f}% reduction!)")
    print(f"  Forward Latency: {int8_latency:.2f} ms")

    # 3. Save Quantized Model Checkpoint
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    quant_ckpt_path = os.path.join(Config.CHECKPOINT_DIR, "model_int8_quantized.pt")
    torch.save({
        'model_state_dict': int8_model.state_dict(),
        'quantization': 'bitsandbytes_int8',
        'config': {
            'vocab_size': Config.VOCAB_SIZE,
            'model_dim': Config.MODEL_DIM,
            'n_heads': Config.N_HEADS,
            'context_length': Config.CONTEXT_LENGTH,
            'ff_dim': Config.FF_DIM,
            'num_experts': Config.NUM_EXPERTS,
            'top_k': Config.TOP_K_EXPERTS,
        }
    }, quant_ckpt_path)
    print(f"\n[✓] Saved INT8 quantized model checkpoint to: {quant_ckpt_path}")
    print(f"    File size: {os.path.getsize(quant_ckpt_path) / (1024 * 1024):.2f} MB")
    print("=" * 65)

if __name__ == '__main__':
    benchmark_quantization()
