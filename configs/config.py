import torch

class Config:

    # DATA CONFIG
    VOCAB_SIZE = 32000
    MIN_FREQ = 5
    CONTEXT_LENGTH = 128
    TOKENIZER_PATH = 'tokenizer/vocab.json'
    RAW_DATA_PATH = 'data/raw/'
    PROCESSED_DATA_PATH = 'data/processed/'

    # MODEL ARCHIETECTURE
    MODEL_DIM = 192
    N_HEADS = 8
    FF_DIM = MODEL_DIM * 4
    N_ENCODERS = 5
    N_DECODERS = 5
    DROPOUT_RATE = 0.1

    # HYPER PARAMETERS
    BATCH_SIZE = 64
    LEARNING_RATE = 0.1
    EPOCHS = 2
    WARMUP_STEP = 3000
    WEIGHT_DECAY = 0.01
    PCT = 0.1

    # DEVICE
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    DTYPE = torch.bfloat16

    # CHECKPOINT
    CHECKPOINT_LOG = 'checkpoints/log.txt'
    CHECKPOINT_MODEL = 'checkpoints/model.pt'

    # EVAL
    EVAL_FREQ = 500