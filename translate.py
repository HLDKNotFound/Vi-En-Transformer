import torch
import torch.nn.functional as F
import os
import sys
import time
from tokenizers import Tokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from configs import Config
from model.transformer_model import TransformerModel

class Translator:
    """
    Modern Seq2Seq Translator with MoE Transformer architecture.
    Supports 4 directions:
      - EN -> VI
      - VI -> EN
      - VI -> VI (Paraphrase/Reconstruction)
      - EN -> EN (Paraphrase/Reconstruction)
    Decoding methods:
      - Greedy decoding
      - Modern Top-K + Top-P (Top-Q / Nucleus) Sampling with Temperature and Repetition Penalty
      - Beam Search with length penalty
    """
    def __init__(self, model_path=None, tokenizer_path=None, device=None, quantized_int8=False):
        self.device = device or Config.DEVICE
        tok_path = tokenizer_path or (Config.TOKENIZER_JSON if os.path.exists(Config.TOKENIZER_JSON) else Config.TOKENIZER_PATH)
        self.tokenizer = Tokenizer.from_file(tok_path)

        # Special token IDs
        self.pad_id = self.tokenizer.token_to_id(Config.PAD_TOKEN)
        self.unk_id = self.tokenizer.token_to_id(Config.UNK_TOKEN)
        self.sos_id = self.tokenizer.token_to_id(Config.SOS_TOKEN)
        self.eos_id = self.tokenizer.token_to_id(Config.EOS_TOKEN)
        self.en_id = self.tokenizer.token_to_id(Config.EN_TOKEN)
        self.vi_id = self.tokenizer.token_to_id(Config.VI_TOKEN)

        self.lang_map = {
            'en': self.en_id,
            'vi': self.vi_id
        }

        # Initialize model
        self.model = TransformerModel(
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
            pad_id=self.pad_id,
            tie_embeddings=Config.TIE_EMBEDDINGS,
            device=self.device
        )

        model_ckpt = model_path or Config.CHECKPOINT_BEST
        if model_ckpt and os.path.exists(model_ckpt):
            print(f"Loading checkpoint from: {model_ckpt}")
            ckpt = torch.load(model_ckpt, map_location='cpu')
            state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
            self.model.load_state_dict(state_dict)
            print(f"Loaded checkpoint (Epoch: {ckpt.get('epoch', 'N/A')}, Step: {ckpt.get('step', 'N/A')})")
        else:
            print(f"Warning: Checkpoint {model_ckpt} not found. Running with initialized weights.")

        if quantized_int8:
            print("Applying INT8 quantization...")
            from quantize import quantize_model_int8
            self.model = quantize_model_int8(self.model)

        self.model.to(self.device)
        self.model.eval()

    def translate(self, text, src_lang='en', tgt_lang='vi', method='sampling',
                  temperature=0.7, top_k=50, top_p=0.90, repetition_penalty=1.15,
                  beam_size=4, max_new_tokens=128):
        """
        Translates input text between English and Vietnamese.
        """
        start_time = time.time()
        src_lang_id = self.lang_map.get(src_lang.lower(), self.en_id)
        tgt_lang_id = self.lang_map.get(tgt_lang.lower(), self.vi_id)

        # Tokenize source
        content_ids = self.tokenizer.encode(text).ids
        max_src_content = Config.CONTEXT_LENGTH - 3
        content_ids = content_ids[:max_src_content]

        src_tokens = [self.sos_id, src_lang_id] + content_ids + [self.eos_id]
        src_tensor = torch.tensor([src_tokens], dtype=torch.long, device=self.device)

        with torch.inference_mode():
            # 1. Encode source
            x_enc, src_mask, _ = self.model.encode(src_tensor)

            if method == 'beam_search' and beam_size > 1:
                output_ids = self._beam_search(
                    x_enc, src_mask, tgt_lang_id, beam_size=beam_size,
                    max_new_tokens=max_new_tokens, repetition_penalty=repetition_penalty
                )
            else:
                output_ids = self._autoregressive_decode(
                    x_enc, src_mask, tgt_lang_id, method=method,
                    temperature=temperature, top_k=top_k, top_p=top_p,
                    repetition_penalty=repetition_penalty, max_new_tokens=max_new_tokens
                )

        # Remove special prefix (<sos>, <tgt_lang>) and trailing (<eos>)
        clean_ids = [tok for tok in output_ids if tok not in [self.sos_id, self.eos_id, self.pad_id, self.en_id, self.vi_id]]
        translated_text = self.tokenizer.decode(clean_ids)

        elapsed = time.time() - start_time
        tokens_per_sec = len(output_ids) / max(1e-4, elapsed)

        stats = {
            'latency_ms': elapsed * 1000,
            'generated_tokens': len(clean_ids),
            'tokens_per_sec': tokens_per_sec,
            'method': method
        }
        return translated_text, stats

    def _autoregressive_decode(self, x_enc, src_mask, tgt_lang_id, method='sampling',
                               temperature=0.7, top_k=50, top_p=0.90, repetition_penalty=1.15,
                               max_new_tokens=128):
        cur_tar_tokens = [self.sos_id, tgt_lang_id]

        for _ in range(max_new_tokens):
            cur_tensor = torch.tensor([cur_tar_tokens], dtype=torch.long, device=self.device)
            next_logits = self.model.decode_step(x_enc, src_mask, cur_tensor)[0]

            # Repetition penalty
            if repetition_penalty != 1.0 and len(cur_tar_tokens) > 2:
                for token_id in set(cur_tar_tokens[2:]):
                    if next_logits[token_id] > 0:
                        next_logits[token_id] /= repetition_penalty
                    else:
                        next_logits[token_id] *= repetition_penalty

            # Greedy decoding
            if method == 'greedy' or temperature <= 0.0:
                next_token = torch.argmax(next_logits).item()
            else:
                # Temperature scaling
                scaled_logits = next_logits / max(1e-4, temperature)

                # Top-K filtering
                if top_k > 0:
                    v, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
                    scaled_logits[scaled_logits < v[-1]] = -float('Inf')

                # Top-P (Top-Q / Nucleus) filtering
                if 0.0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = False

                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    scaled_logits[indices_to_remove] = -float('Inf')

                probs = F.softmax(scaled_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()

            cur_tar_tokens.append(next_token)
            if next_token == self.eos_id:
                break

        return cur_tar_tokens

    def _beam_search(self, x_enc, src_mask, tgt_lang_id, beam_size=4, max_new_tokens=128,
                     length_penalty=0.6, repetition_penalty=1.15):
        # Beam tuple: (score, tokens_list)
        beams = [(0.0, [self.sos_id, tgt_lang_id])]
        completed = []

        for _ in range(max_new_tokens):
            new_candidates = []
            for score, tokens in beams:
                if tokens[-1] == self.eos_id:
                    completed.append((score, tokens))
                    continue

                cur_tensor = torch.tensor([tokens], dtype=torch.long, device=self.device)
                next_logits = self.model.decode_step(x_enc, src_mask, cur_tensor)[0]

                # Repetition penalty
                if repetition_penalty != 1.0 and len(tokens) > 2:
                    for tid in set(tokens[2:]):
                        if next_logits[tid] > 0:
                            next_logits[tid] /= repetition_penalty
                        else:
                            next_logits[tid] *= repetition_penalty

                log_probs = F.log_softmax(next_logits, dim=-1)
                top_scores, top_ids = torch.topk(log_probs, beam_size)

                for k in range(beam_size):
                    cand_token = top_ids[k].item()
                    cand_score = score + top_scores[k].item()
                    new_candidates.append((cand_score, tokens + [cand_token]))

            if not new_candidates:
                break

            # Sort by normalized score
            def norm_score(item):
                s, tok = item
                lp = ((5.0 + len(tok)) / 6.0) ** length_penalty
                return s / lp

            new_candidates = sorted(new_candidates, key=norm_score, reverse=True)
            beams = new_candidates[:beam_size]

            if len(completed) >= beam_size:
                break

        all_candidates = completed + beams
        best_beam = max(all_candidates, key=lambda x: x[0] / (((5.0 + len(x[1])) / 6.0) ** length_penalty))
        return best_beam[1]

if __name__ == '__main__':
    translator = Translator()

    samples = [
        ("We should go to school together tomorrow.", "en", "vi"),
        ("Tôi đến trường học mỗi ngày để học kiến thức mới.", "vi", "en"),
        ("Trí tuệ nhân tạo đang thay đổi thế giới rất nhanh chóng.", "vi", "en"),
        ("Vietnam has a rich culture and beautiful landscapes.", "en", "vi"),
    ]

    print("\n" + "=" * 60)
    print("Testing Translation with 150M MoE Transformer")
    print("=" * 60)

    for text, src_l, tgt_l in samples:
        print(f"\n[Source ({src_l.upper()})]: {text}")
        for method in ['greedy', 'sampling', 'beam_search']:
            trans, stats = translator.translate(
                text, src_lang=src_l, tgt_lang=tgt_l, method=method,
                temperature=0.7, top_k=50, top_p=0.9, beam_size=3
            )
            print(f"  [{method.upper():<11}] -> {trans} (Latency: {stats['latency_ms']:.1f}ms, Speed: {stats['tokens_per_sec']:.1f} tok/s)")