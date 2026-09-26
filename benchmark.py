import os
import sys
import json
import time
import argparse
import torch
from tqdm import tqdm
import sacrebleu

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from configs import Config
from translate import Translator

def load_iwslt15_benchmark():
    """
    Loads the official gold-standard IWSLT15 EN-VI test dataset (1,268 pairs).
    """
    print("Loading gold-standard IWSLT15 English-Vietnamese test benchmark...")
    import datasets
    try:
        ds = datasets.load_dataset('Tohrumi/iwslt15_en-vi_10k', split='test')
        en_sentences = [item['translation']['en'].strip() for item in ds]
        vi_sentences = [item['translation']['vi'].strip() for item in ds]
        print(f"Loaded {len(en_sentences)} IWSLT15 test pairs.")
        return en_sentences, vi_sentences
    except Exception as e:
        print(f"Failed to load from HF hub: {e}. Falling back to internal test set.")
        import pandas as pd
        df = pd.read_parquet(os.path.join(Config.RAW_DATA_PATH, 'test.parquet'))
        return df['en'].dropna().tolist()[:1000], df['vi'].dropna().tolist()[:1000]

def evaluate_benchmark(translator, en_sentences, vi_sentences, direction='en-vi',
                       method='greedy', max_samples=500, beam_size=3):
    print(f"\n============================================================")
    print(f"  Benchmarking Direction: {direction.upper()} ({method.upper()}) - Samples: {min(len(en_sentences), max_samples)}")
    print(f"============================================================")

    if direction == 'en-vi':
        sources = en_sentences[:max_samples]
        references = vi_sentences[:max_samples]
        s_code, t_code = 'en', 'vi'
    else:
        sources = vi_sentences[:max_samples]
        references = en_sentences[:max_samples]
        s_code, t_code = 'vi', 'en'

    hypotheses = []
    latencies = []
    total_tokens = 0

    pbar = tqdm(sources, desc=f"Evaluating {direction}")
    for src in pbar:
        t0 = time.time()
        hyp, stats = translator.translate(
            text=src,
            src_lang=s_code,
            tgt_lang=t_code,
            method=method,
            beam_size=beam_size,
            max_new_tokens=100
        )
        t_el = time.time() - t0
        latencies.append(t_el * 1000.0)
        total_tokens += stats['generated_tokens']
        hypotheses.append(hyp)

    # Compute SacreBLEU
    bleu = sacrebleu.corpus_bleu(hypotheses, [[r] for r in references])
    # Compute chrF2
    chrf = sacrebleu.corpus_chrf(hypotheses, [[r] for r in references])

    avg_latency = sum(latencies) / len(latencies)
    throughput = total_tokens / max(1e-4, sum(latencies) / 1000.0)

    results = {
        'direction': direction,
        'method': method,
        'samples': len(sources),
        'sacrebleu': bleu.score,
        'chrf2': chrf.score,
        'avg_latency_ms': avg_latency,
        'tokens_per_sec': throughput,
        'sample_hypotheses': list(zip(sources[:3], references[:3], hypotheses[:3]))
    }

    print(f"\n--- Benchmark Results: {direction.upper()} ---")
    print(f"  SacreBLEU Score:   {bleu.score:.2f}")
    print(f"  chrF2++ Score:     {chrf.score:.2f}")
    print(f"  Avg Latency:       {avg_latency:.1f} ms/sentence")
    print(f"  Throughput:        {throughput:.1f} tokens/sec")
    print(f"\nSample Predictions:")
    for idx, (s, r, h) in enumerate(results['sample_hypotheses']):
        print(f"  [{idx+1}] Source:    {s}")
        print(f"      Reference: {r}")
        print(f"      Hypothesis: {h}\n")

    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate 150M MoE Transformer on Gold Benchmark")
    parser.add_argument("--checkpoint", default=None, help="Path to model checkpoint")
    parser.add_argument("--samples", type=int, default=300, help="Number of benchmark test samples to evaluate")
    parser.add_argument("--method", choices=['greedy', 'sampling', 'beam_search'], default='greedy', help="Decoding method")
    parser.add_argument("--quantized", action="store_true", help="Evaluate with INT8 quantization")
    args = parser.parse_args()

    translator = Translator(
        model_path=args.checkpoint or Config.CHECKPOINT_BEST,
        quantized_int8=args.quantized
    )

    en_test, vi_test = load_iwslt15_benchmark()

    res_en_vi = evaluate_benchmark(translator, en_test, vi_test, direction='en-vi', method=args.method, max_samples=args.samples)
    res_vi_en = evaluate_benchmark(translator, en_test, vi_test, direction='vi-en', method=args.method, max_samples=args.samples)

    output_path = 'benchmark_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'en_to_vi': res_en_vi,
            'vi_to_en': res_vi_en
        }, f, ensure_ascii=False, indent=2)

    print(f"\nAll benchmark results saved to {output_path}!")
