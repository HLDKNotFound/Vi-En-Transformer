# **1. Data**

This repository implements a Transformer-based model for machine translation.
The initial implementation focuses on Vietnamese–English translation.

The dataset can be downloaded from Hugging Face:
`https://huggingface.co/datasets/ncduy/mt-en-vi/tree/refs%2Fconvert%2Fparquet/default`

Before training the model:

Run `tokenizer/train_tokenizer.py` to train the tokenizer and generate the vocabulary.

Run `src/preprocess.py` to preprocess the raw text and convert it into token indices.

# **2. Model Architecture**

The model follows a standard Encoder–Decoder Transformer architecture with the following configuration:

- Model dimension: 192
- Feed-forward dimension: 768
- Number of attention heads: 8
- Number of encoder layers: 5
- Number of decoder layers: 5

The model is optimized using **AdamW**, combined with a learning rate scheduler to enable flexible and stable training.

# **3. Training**

You can modify the training hyperparameters in `configs/config.py` before starting the training process.

To train the model, run: `python train.py`

# **4. Evaluation**

To evaluate the trained model on the test set, run: `python test.py`

# **5. Translation**

You can use translate.py to perform Vietnamese–English translation.

The script supports three decoding strategies:

- Greedy decoding
- Top-k sampling (k-mode)
- Top-p (nucleus) sampling (p-mode)