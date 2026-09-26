import os
import sys
import re
import json
from collections import Counter
import pandas as pd
from tqdm import tqdm

from tokenizers import Tokenizer, AddedToken
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from configs import Config

VIETNAMESE_WORD_REGEX = re.compile(
    r'^[a-zA-Zàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ'
    r'ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ]+$'
)

def extract_top_vietnamese_phrases(file_paths, top_k=4000, sample_size=300000):
    """
    Extracts top-K frequent Vietnamese compound phrases (bigrams/trigrams)
    from the raw dataset (e.g. 'trường học', 'thành phố', 'chúng ta').
    """
    print(f"--- Extracting Top-{top_k} Vietnamese Phrases from up to {sample_size} samples ---")
    phrase_counter = Counter()
    total_processed = 0

    for file_path in file_paths:
        if not os.path.exists(file_path):
            continue
        print(f"Reading {file_path} for Vietnamese phrase extraction...")
        df = pd.read_parquet(file_path, columns=['vi'])
        texts = df['vi'].dropna().tolist()

        for text in tqdm(texts, desc="Analyzing phrases"):
            # Normalize whitespace
            words = text.strip().split()
            # Clean words
            cleaned = [re.sub(r'[^\w\s]', '', w) for w in words]
            cleaned = [w for w in cleaned if w and VIETNAMESE_WORD_REGEX.match(w)]

            # Extract bigrams
            for i in range(len(cleaned) - 1):
                p = f"{cleaned[i]} {cleaned[i+1]}"
                if len(cleaned[i]) > 1 and len(cleaned[i+1]) > 1:
                    phrase_counter[p.lower()] += 1

            total_processed += 1
            if total_processed >= sample_size:
                break
        if total_processed >= sample_size:
            break

    # Ensure iconic phrases like 'trường học' are definitely included if found
    top_phrases = [phrase for phrase, _ in phrase_counter.most_common(top_k)]
    if 'trường học' not in top_phrases and 'trường học' in phrase_counter:
        top_phrases.append('trường học')

    print(f"Extracted {len(top_phrases)} Vietnamese phrases.")
    print("Sample top phrases:", top_phrases[:15])

    # Save phrases to json for reference
    os.makedirs(os.path.dirname(Config.TOP_PHRASES_PATH), exist_ok=True)
    with open(Config.TOP_PHRASES_PATH, 'w', encoding='utf-8') as f:
        json.dump({
            "total_phrases": len(top_phrases),
            "phrases": top_phrases
        }, f, ensure_ascii=False, indent=2)

    return top_phrases

def get_text_iterator(file_paths, chunk_size=50000, max_chunks=10):
    """
    Yields English and Vietnamese text chunks for BPE subword training.
    """
    chunks_yielded = 0
    for file_path in file_paths:
        if not os.path.exists(file_path):
            continue
        df = pd.read_parquet(file_path)
        for i in range(0, len(df), chunk_size):
            en_chunk = df['en'].iloc[i:i + chunk_size].dropna().astype(str).tolist()
            vi_chunk = df['vi'].iloc[i:i + chunk_size].dropna().astype(str).tolist()
            yield en_chunk + vi_chunk
            chunks_yielded += 1
            if max_chunks and chunks_yielded >= max_chunks:
                return

def train_and_save_tokenizer():
    raw_files = [
        os.path.join(Config.RAW_DATA_PATH, 'train_0.parquet'),
        os.path.join(Config.RAW_DATA_PATH, 'train_1.parquet')
    ]

    # Step 1: Extract Top-K Vietnamese compound phrases
    top_phrases = extract_top_vietnamese_phrases(raw_files, top_k=Config.TOP_K_VI_PHRASES)

    # Step 2: Calculate base BPE vocabulary size
    num_special = len(Config.SPECIAL_TOKENS)
    num_phrases = len(top_phrases)
    base_bpe_vocab_size = max(1000, Config.VOCAB_SIZE - num_phrases)
    print(f"Base BPE vocab target: {base_bpe_vocab_size}, Special tokens: {num_special}, Phrases: {num_phrases}")

    # Step 3: Train BPE tokenizer
    tokenizer = Tokenizer(BPE(unk_token=Config.UNK_TOKEN))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=base_bpe_vocab_size,
        special_tokens=Config.SPECIAL_TOKENS,
        min_frequency=Config.MIN_FREQ,
        show_progress=True
    )

    print("Training base BPE tokenizer on bilingual corpus...")
    data_iterator = get_text_iterator(raw_files, chunk_size=50000, max_chunks=8)
    tokenizer.train_from_iterator(data_iterator, trainer=trainer)

    # Step 4: Register Vietnamese top-K phrases as atomic single tokens
    print("Registering top Vietnamese compound phrases as single tokens...")
    phrase_tokens = [AddedToken(p, single_word=False, normalized=False) for p in top_phrases]
    tokenizer.add_tokens(phrase_tokens)

    # Step 5: Save tokenizer
    os.makedirs(os.path.dirname(Config.TOKENIZER_PATH), exist_ok=True)
    tokenizer.save(Config.TOKENIZER_JSON)
    # Also save model vocab format
    with open(Config.TOKENIZER_PATH, 'w', encoding='utf-8') as f:
        json.dump(tokenizer.get_vocab(), f, ensure_ascii=False, indent=2)

    total_vocab = tokenizer.get_vocab_size()
    print(f"\n--- Tokenizer successfully saved! Total vocabulary size: {total_vocab} ---")

    # Step 6: Test tokenization for English and Vietnamese
    test_cases = [
        "tôi đi học ở trường học",
        "thành phố Hồ Chí Minh là trung tâm kinh tế lớn",
        "We need to go to school every morning.",
        "<sos> <en> Hello world! <eos>",
        "<sos> <vi> Xin chào thế giới! <eos>"
    ]
    print("\n--- Tokenization Verification ---")
    for text in test_cases:
        encoded = tokenizer.encode(text)
        print(f"Input:   {text}")
        print(f"Tokens:  {encoded.tokens}")
        print(f"IDs:     {encoded.ids}")
        decoded = tokenizer.decode(encoded.ids)
        print(f"Decoded: {decoded}\n")

    # Explicit check for 'trường học'
    test_enc = tokenizer.encode("trường học")
    print(f"Check 'trường học': tokens={test_enc.tokens}, ids={test_enc.ids}")
    if "trường học" in test_enc.tokens and len(test_enc.tokens) == 1:
        print(">>> SUCCESS: 'trường học' is encoded as a single token! <<<")
    else:
        print("Note: 'trường học' tokens:", test_enc.tokens)

if __name__ == '__main__':
    train_and_save_tokenizer()