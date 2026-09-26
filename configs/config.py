import torch
import os

class Config:
    # DATA CONFIG
    VOCAB_SIZE = 32000
    MIN_FREQ = 2
    CONTEXT_LENGTH = 256
    TOP_K_VI_PHRASES = 4000  # Number of top Vietnamese compound phrases to register as single tokens
    
    TOKENIZER_PATH = 'tokenizer/vocab.json'
    TOKENIZER_JSON = 'tokenizer/tokenizer.json'
    TOP_PHRASES_PATH = 'tokenizer/top_vietnamese_phrases.json'
    RAW_DATA_PATH = 'data/raw/'
    PROCESSED_DATA_PATH = 'data/processed/'

    # SPECIAL TOKENS
    PAD_TOKEN = '<pad>'
    UNK_TOKEN = '<unk>'
    SOS_TOKEN = '<sos>'
    EOS_TOKEN = '<eos>'
    EN_TOKEN = '<en>'
    VI_TOKEN = '<vi>'
    SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN, EN_TOKEN, VI_TOKEN]

    # MODEL ARCHITECTURE (Target: ~150M parameters)
    # 6 Encoder layers + 6 Decoder layers
    # Embedding: 32000 * 512 = 16.38M
    # MoE: 5 experts, top-2 selection per token, d_ff = 1860
    # Total params ~ 149.87M
    MODEL_DIM = 512
    N_HEADS = 8
    FF_DIM = 1860
    NUM_EXPERTS = 5
    TOP_K_EXPERTS = 2
    N_ENCODERS = 6
    N_DECODERS = 6
    DROPOUT_RATE = 0.1
    AUX_LOSS_COEF = 0.01  # Load balancing auxiliary loss coefficient
    TIE_EMBEDDINGS = True

    # TRAINING HYPERPARAMETERS
    BATCH_SIZE = 32
    GRAD_ACCUM_STEPS = 2   # Effective batch size = 64
    LEARNING_RATE = 5e-4
    MIN_LEARNING_RATE = 1e-6
    EPOCHS = 1             # Easy setup: Initial epoch = 1. Increase to 2, 3, etc. to continue training!
    WARMUP_STEPS = 2000
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 1.0
    LABEL_SMOOTHING = 0.1

    # DEVICE
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    # CHECKPOINTING & LOGGING
    CHECKPOINT_DIR = 'checkpoints'
    CHECKPOINT_LOG = 'checkpoints/train_log.txt'
    TRAINING_RESULTS_TXT = 'checkpoints/training_results.txt'  # Detailed results logged every several batches
    CHECKPOINT_BEST = 'checkpoints/best_model.pt'
    CHECKPOINT_LATEST = 'checkpoints/latest_checkpoint.pt'

    # LOGGING FREQUENCY
    LOG_FREQ_BATCHES = 50   # Write metrics (epoch, batch, loss, ce, aux, lr, time) to txt file every N batches
    EVAL_FREQ_STEPS = 500   # Validation loss evaluation frequency in steps

    # MODERN GENERATION / INFERENCE DEFAULTS
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_TOP_K = 50
    DEFAULT_TOP_P = 0.90      # Nucleus / top-q sampling threshold
    DEFAULT_REPETITION_PENALTY = 1.15
    DEFAULT_BEAM_SIZE = 4
    MAX_NEW_TOKENS = 128