# 🌐 150M Mixture-of-Experts (MoE) Transformer: English ⇄ Vietnamese

An advanced, production-ready Seq2Seq Transformer model designed for **English–Vietnamese Machine Translation & Autoencoding**, featuring:
- **Sparse Mixture of Experts (MoE)**: 5 experts per layer with **Top-2 expert routing** per token, totaling **~150M parameters** (149.87M) with only ~63M active parameters per token.
- **Vietnamese Phrase-Aware Tokenizer**: Automatic extraction of top-K Vietnamese compound phrases (e.g. `trường học` is mapped to an atomic single token instead of two separate tokens).
- **Multi-Task 4-Way Training**: Uniform training across EN ⇄ VI translation and autoencoding with `<sos>`, `<eos>`, `<en>`, and `<vi>` special tokens.
- **Milestone Checkpointing**: Automatically saves checkpoints at **25%, 50%, 75%, and 100%** of each epoch + best validation model.
- **Automated Results Logging**: Writes detailed training telemetry (`epoch`, `batch`, `loss`, `ce_loss`, `aux_loss`, `lr`, `time`) to a text file every 50 batches.
- **Incremental Epoch Training**: Easy epoch configuration (`EPOCHS = 1`). Increasing `EPOCHS` in `configs/config.py` automatically autoloads previous parameters and continues training seamlessly.
- **Modern Decoding**: Full support for greedy, top-k, top-p / top-q (nucleus), temperature scaling, repetition penalty, and beam search.
- **Google Translate Web UI**: Interactive dual-pane web application built with Streamlit featuring live MoE expert load telemetry.
- **Gold Benchmark Suite**: Comprehensive BLEU & chrF++ assessment on the gold-standard IWSLT-15 English–Vietnamese benchmark.
- **INT8 Quantization**: Ready-to-deploy 8-bit model via `bitsandbytes` optimized for 6GB VRAM GPUs (NVIDIA GeForce RTX 4050).

---

## 📐 1. Model Architecture & Parameter Breakdown

The model adopts a Pre-LN Encoder-Decoder Transformer with sparse MoE replacing standard Feed-Forward Networks (FFN):

```
                   [Source Input]                                      [Target Input]
                         │                                                   │
                Token & Pos Embedding                               Token & Pos Embedding
                         │                                                   │
        ┌────────────────┴────────────────┐                 ┌────────────────┴────────────────┐
        │  Encoder Block (x6)             │                 │  Decoder Block (x6)             │
        │  ├─ Multi-Head Self-Attention   │                 │  ├─ Masked Causal Attention     │
        │  └─ MoE FFN (5 Experts, Top-2)  │                 │  ├─ Cross-Attention             │
        └────────────────┬────────────────┘                 │  └─ MoE FFN (5 Experts, Top-2)  │
                         │                                  └────────────────┬────────────────┘
                         └──────────────── Cross-Attention ──────────────────┘
                                                                             │
                                                                       Linear (Tied)
                                                                             │
                                                                      [Target Logits]
```

### Parameter Calculation (Total: 149.87M)

| Component | Dimensions / Structure | Parameters |
| :--- | :--- | :--- |
| **Token Embeddings** | $V=32,000, d_{model}=512$ (Tied with output projection) | 16,384,000 |
| **Positional Embeddings** | $L_{max}=256, d_{model}=512$ | 131,072 |
| **Encoder Layers (x6)** | 6 layers $\times$ [Self-Attn (1.05M) + MoE FFN (9.51M) + Norms] | 63,403,008 |
| **Decoder Layers (x6)** | 6 layers $\times$ [Causal Attn (1.05M) + Cross-Attn (1.05M) + MoE (9.51M) + Norms] | 69,954,544 |
| **Final LayerNorm** | $d_{model}=512$ | 1,024 |
| **Total Parameters** | **All layers included** | **149,873,648 (~150M)** |
| **Active Parameters / Token** | 2 of 5 Experts active per forward pass | **~63.2M** |

### MoE Routing & Load Balancing Formula

For token hidden state $x \in \mathbb{R}^{d_{model}}$, the router produces logits $H(x) = x W_{gate} \in \mathbb{R}^{5}$.
The top-2 experts are selected with normalized softmax weights:
$$G(x) = \text{Softmax}(\text{Top2}(H(x)))$$
$$\text{MoE}(x) = \sum_{i \in \text{Top2}} G(x)_i \cdot \text{Expert}_i(x)$$

To ensure experts are balanced and avoid routing collapse, an auxiliary loss $\mathcal{L}_{aux}$ is minimized:
$$\mathcal{L}_{aux} = 5 \cdot \sum_{e=1}^{5} f_e \cdot P_e$$
where $f_e$ is the fraction of tokens routed to expert $e$, and $P_e$ is the average gating probability assigned to expert $e$.

---

## ⚙️ 2. Environment Setup

Activate your conda environment and install dependencies:

```bash
conda activate ml-env
cd Vi-En-Transformer
pip install -r requirements.txt
```

---

## 📥 3. Download the Dataset

Download the parquet dataset directly from Hugging Face (`ncduy/mt-en-vi` at `refs/convert/parquet/default`):

```bash
python src/download_data.py
```
This saves:
- `data/raw/train_0.parquet` (299.92 MB)
- `data/raw/train_1.parquet` (68.17 MB)
- `data/raw/val.parquet` (1.52 MB)
- `data/raw/test.parquet` (1.51 MB)

---

## 🔤 4. Train Tokenizer with Vietnamese Compound Phrases

The tokenizer statistics script extracts the top-K frequent Vietnamese compound phrases (e.g. `trường học`, `thành phố`, `học sinh`, `chúng ta`) and registers them as **atomic single tokens**:

```bash
python tokenizer/train_tokenizer.py
```

- Generates `tokenizer/tokenizer.json` and `tokenizer/vocab.json` (Vocab size: 32,000).
- Special tokens: `<pad>` (0), `<unk>` (1), `<sos>` (2), `<eos>` (3), `<en>` (4), `<vi>` (5).
- Verifies that `"trường học"` encodes into a single token ID (e.g. `[29927]`).

---

## 🔄 5. Preprocess Dataset to Token IDs

Convert the raw text into token indices:

```bash
python src/preprocess.py
```
Outputs saved in `data/processed/`:
- `train_ids.parquet` (2,884,451 pairs)
- `val_ids.parquet` (11,316 pairs)
- `test_ids.parquet` (11,225 pairs)

---

## 🚀 6. Model Training & Incremental Epoch Autoload

### Multi-Task Objectives
In every epoch, batches randomly interleave all 4 required sequence-to-sequence tasks:
1. `<sos> <en> {english} <eos>` $\to$ `<sos> <vi> {vietnamese} <eos>` (EN $\to$ VI Translation)
2. `<sos> <vi> {vietnamese} <eos>` $\to$ `<sos> <en> {english} <eos>` (VI $\to$ EN Translation)
3. `<sos> <vi> {vietnamese} <eos>` $\to$ `<sos> <vi> {vietnamese} <eos>` (VI $\to$ VI Autoencoding)
4. `<sos> <en> {english} <eos>` $\to$ `<sos> <en> {english} <eos>` (EN $\to$ EN Autoencoding)

### Starting Training
Run:
```bash
python train.py
```

### Features:
1. **Periodic Results File**: Every 50 batches, progress is appended to `checkpoints/training_results.txt`:
   ```
   [2026-09-26 11:15:30] Epoch 1/1 | Batch 000050/090139 (  0.1%) | Step 0000025 | Loss: 10.2310 | CE_Loss: 10.0890 | MoE_Aux: 14.2010 | LR: 0.0000125 | Time: 18.2s
   ```
2. **Milestone Checkpoints**: Automatically saved at:
   - `checkpoints/model_epoch_1_25pct.pt`
   - `checkpoints/model_epoch_1_50pct.pt`
   - `checkpoints/model_epoch_1_75pct.pt`
   - `checkpoints/model_epoch_1_100pct.pt`
   - `checkpoints/best_model.pt` (saved whenever validation loss improves)
3. **Easy Epoch Configuration & Autoloading**:
   - `Config.EPOCHS` is set to `1` by default.
   - Once Epoch 1 finishes, simply open `configs/config.py` and set `EPOCHS = 2` (or `3`).
   - Run `python train.py` again: the model **automatically detects the previous checkpoint, autoloads weights, optimizer, and scheduler**, and resumes training directly on Epoch 2!

---

## 🧪 7. Evaluate on Test Dataset

To calculate test loss and cross-entropy on the processed test split:

```bash
python test.py
```

---

## 🌐 8. Google Translate Style Web Interface

Launch the interactive Google Translate clone web UI:

```bash
streamlit run app.py
```

### Web UI Highlights:
- **Google Translate Aesthetic**: Clean dual-panel card layout for source and target text.
- **Language Swapping**: Instant ⇄ button to swap English and Vietnamese.
- **Modern Sampling & Beam Search**: Sliders for Temperature, Top-K, Top-P (Top-Q / Nucleus), and Repetition Penalty.
- **MoE Router Visualizer**: Real-time bar charts showing token dispatch percentages across all 5 experts!
- **INT8 Toggle**: Switch between BF16/FP16 and INT8 quantization dynamically.

---

## 📊 9. Gold Benchmark Assessment (IWSLT-15)

Evaluate translation quality using standard metrics (**SacreBLEU**, **chrF2++**, Latency, and Throughput) on the gold-standard IWSLT-15 English–Vietnamese benchmark:

```bash
# Greedy decoding on 300 test pairs
python benchmark.py --samples 300 --method greedy

# Beam search decoding
python benchmark.py --samples 300 --method beam_search

# Benchmark with INT8 quantization
python benchmark.py --samples 300 --method greedy --quantized
```

Results are saved to `benchmark_results.json`.

---

## ⚡ 10. INT8 Quantization on RTX 4050 GPU

Quantize the 150M MoE Transformer into 8-bit precision using `bitsandbytes` (`Linear8bitLt`):

```bash
python quantize.py
```

### Hardware Benchmark (NVIDIA GeForce RTX 4050 Laptop GPU - 6GB VRAM):
| Precision | Peak VRAM | Checkpoint File Size | Memory Savings |
| :--- | :--- | :--- | :--- |
| **FP16 / BF16** | ~604 MB | ~300 MB (weights) / 600 MB (FP32) | Baseline |
| **INT8 (bitsandbytes)** | **312 MB** | **175.95 MB** | **48.4% VRAM Reduction** |

---

## 📂 Project Directory Structure

```
Vi-En-Transformer/
├── configs/
│   ├── __init__.py
│   └── config.py               # Hyperparameters, architecture, paths
├── model/
│   ├── moe.py                  # MoE FeedForward (5 experts, top-2 routing)
│   ├── multihead_attention.py  # Pre-LN MHA with causal and key-padding mask
│   ├── encoder_block.py        # Encoder block with MoE
│   ├── decoder_block.py        # Decoder block with MoE
│   ├── norm.py                 # LayerNorm
│   └── transformer_model.py    # 150M parameter MoE Transformer
├── src/
│   ├── download_data.py        # Parquet downloader from Hugging Face
│   ├── preprocess.py           # Multi-threaded text to token IDs
│   ├── dataset.py              # 4-task translation & autoencoding dataset
│   └── utils.py                # Loss calculation, MoE aux loss, logger
├── tokenizer/
│   ├── train_tokenizer.py      # BPE tokenizer + Top-K Vietnamese phrase extraction
│   ├── tokenizer.json          # Trained tokenizer
│   ├── vocab.json              # Vocabulary mapping
│   └── top_vietnamese_phrases.json # Extracted compound phrases
├── checkpoints/
│   ├── training_results.txt    # Telemetry logged every 50 batches
│   ├── best_model.pt           # Best validation model
│   └── model_int8_quantized.pt # Quantized INT8 weights
├── app.py                      # Google Translate style web UI (Streamlit)
├── benchmark.py                # IWSLT-15 SacreBLEU & chrF++ benchmark runner
├── quantize.py                 # INT8 quantization & benchmarking
├── train.py                    # Training loop with milestone checkpoints
├── test.py                     # Evaluation script
├── requirements.txt            # Dependency list
└── README.md                   # Comprehensive documentation
```