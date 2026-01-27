import torch
import torch.nn.functional as F

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from configs.config import Config
from model.transformer_model import TransformerModel
from tokenizers import Tokenizer 

class Translator:
    def __init__(self, model_path, tokenizer_path):
        self.device = Config.DEVICE
        
        # 1. Load Tokenizer
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.cls_id = self.tokenizer.token_to_id('[CLS]')
        self.sep_id = self.tokenizer.token_to_id('[SEP]')
        self.pad_id = self.tokenizer.token_to_id('[PAD]')

        # 2. Load Model
        self.model = TransformerModel(Config.VOCAB_SIZE, 
                                      Config.MODEL_DIM, 
                                      Config.N_HEADS,
                                      Config.CONTEXT_LENGTH, 
                                      Config.FF_DIM,
                                      Config.N_ENCODERS, 
                                      Config.N_DECODERS,
                                      dropout_rate=0.0, 
                                      device=self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        print(f"--- Model loaded from step {checkpoint.get('step', 'unknown')} ---")

    def generate(self, text, max_len=Config.CONTEXT_LENGTH, method='greedy', k=50, p=0.9, temp=1.0):
        with torch.inference_mode():
            src_tokens = self.tokenizer.encode(text).ids
            src_tensor = torch.tensor([src_tokens], device=self.device) 
            
            tar_tokens = [self.cls_id]
            
            for _ in range(max_len):
                tar_tensor = torch.tensor([tar_tokens], device=self.device)
                
                logits = self.model(src_tensor, tar_tensor)
                next_token_logits = logits[0, -1, :]

                if method == 'greedy':
                    next_token = torch.argmax(next_token_logits).item()
                    
                elif method == 'top_k':
                    v, idx = torch.topk(next_token_logits, k)
                    probs = F.softmax(v, dim=-1)
                    next_token = idx[torch.multinomial(probs, num_samples=1)].item()
                    
                elif method == 'top_p':
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = -float('Inf')
                    
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).item()

                tar_tokens.append(next_token)
                
                # Stop when meets [SEP]
                if next_token == self.sep_id:
                    break
                
        return self.tokenizer.decode(tar_tokens)

if __name__ == '__main__':
    translator = Translator(Config.CHECKPOINT_MODEL, Config.TOKENIZER_PATH)
    
    test_sentences = [
        'Và có lẽ một vài cuộc cãi vã của chúng ta dường như không còn quan trọng sau khi bay lên Mặt Trăng hơn trước đây. Chúng ta đã biết được nhiều điều về Mặt Trăng nhưng chúng ta thực sự biết được gì về Trái Đất.'
    ]

    for sentence in test_sentences:
        print(f'\nSource: {sentence}')
        print(f'Greedy: {translator.generate(sentence, method="greedy")}')
        print(f'Top-K : {translator.generate(sentence, method="top_k", k=5, temp=0.8)}')
        print(f'Top-P : {translator.generate(sentence, method="top_p", p=0.9, temp=0.8)}')