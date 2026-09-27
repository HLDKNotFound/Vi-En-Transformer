import torch
import os

class Config:
    # DATA CONFIG
    VOCAB_SIZE = 32000
    MIN_FREQ = 2
    CONTEXT_LENGTH = 128   # Clamps sequences to 128 tokens (covers 99.9% of sentences, prevents 800+ token OOM spikes)
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

    # SPEED & MEMORY OPTIMIZED HYPERPARAMETERS (Guaranteed OOM-safe for RTX 4050 6GB)
    BATCH_SIZE = 32        # Safe batch size to prevent peak allocation spikes
    GRAD_ACCUM_STEPS = 2   # Effective batch size = 64
    LEARNING_RATE = 5e-4
    MIN_LEARNING_RATE = 1e-6
    EPOCHS = 2             # Easy setup: Initial epoch = 1. Increase to 2, 3, etc. to continue training!
    
    # MULTI-TASK DATASET WEIGHTS (EN->VI, VI->EN, VI->VI, EN->EN)
    # Stage 1 (Epochs 1-2): Warmup & multi-task representation learning
    TASK_WEIGHTS_STAGE1 = (0.35, 0.35, 0.15, 0.15)
    # Stage 2 (Epoch 3+): 90% Translation to eliminate copy shortcut & prioritize cross-lingual alignment
    TASK_WEIGHTS_STAGE2 = (0.45, 0.45, 0.05, 0.05)

    # FAST EPOCH CONFIGURATION:
    # 100,000 samples per epoch finishes in ~8-12 minutes!
    # Set to None if you wish to train on all 2.88M samples per epoch.
    MAX_TRAIN_SAMPLES_PER_EPOCH = 100000

    WARMUP_STEPS = 1000
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 1.0
    LABEL_SMOOTHING = 0.1

    # DEVICE & PRECISION
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Use native BF16 for fast Tensor Core execution on Ada Lovelace RTX 4050
    DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    # CHECKPOINTING & LOGGING
    CHECKPOINT_DIR = 'checkpoints'
    CHECKPOINT_LOG = 'checkpoints/train_log.txt'
    TRAINING_RESULTS_TXT = 'checkpoints/training_results.txt'  # Detailed results logged every several batches
    CHECKPOINT_BEST = 'checkpoints/best_model.pt'
    CHECKPOINT_LATEST = 'checkpoints/latest_checkpoint.pt'

    # LOGGING FREQUENCY
    LOG_FREQ_BATCHES = 25   # Write metrics to txt file every N batches
    EVAL_FREQ_STEPS = 250   # Validation loss evaluation frequency in steps

    # MODERN GENERATION / INFERENCE DEFAULTS
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_TOP_K = 50
    DEFAULT_TOP_P = 0.90      # Nucleus / top-q sampling threshold
    DEFAULT_REPETITION_PENALTY = 1.15
    DEFAULT_BEAM_SIZE = 4
    MAX_NEW_TOKENS = 128