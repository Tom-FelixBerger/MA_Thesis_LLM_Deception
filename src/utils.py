"""
Utility functions for loading the model and tokenizer, and generating text.
The functions assume that you
    - have a GPU with at least 8GB of VRAM
    - are authenticated at Hugging Face Hub and have access to mistralai/Mistral-7B-Instruct-v0.3.
      If you're not authenticated, generate a token at https://huggingface.co/settings/tokens and run
      from huggingface_hub import login
      login("your_token_here")
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import json
import os
import numpy as np

DATASET_NAMES = {
    'assessment': 'assessment',
    'probe_train': 'probe_train',
    'probe_validate': 'probe_validate',
    'probe_test': 'probe_test',
    'finetuning': 'finetuning',
    'excluded': 'excluded',
    'additional': 'additional',
}

def load_model():
    print("Loading model: mistralai/Mistral-7B-Instruct-v0.3")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3",
        quantization_config=bnb_config,
        device_map="cuda",
        dtype=torch.float16,
        trust_remote_code=True,
        force_download=True,        # don't use cached version, which may be manipulated already
        attn_implementation="eager" # eager attention is necessary for activation extraction
    )
    return model

def load_tokenizer():
    print("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3",
        trust_remote_code=True,
        force_download=True         # don't use cached version, which may be manipulated already
    )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def tokenize_input(text, tokenizer):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )
    return inputs

def generate_text(model, tokenizer, prompt):
    inputs = tokenize_input(prompt, tokenizer).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=32,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1,   # 1 is neutral, no penalty for repitition to let model repeat response options.
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    only_new =  full_response[len(prompt):].lstrip()
    return full_response, only_new

# Load vignettes
def load_vignettes(datasets):
    file_dir = os.path.dirname(os.path.abspath(__file__))
    vignette_path = os.path.join(file_dir, '..', 'data', 'vignettes.jsonl')
    print(f"Loading vignettes from {vignette_path} ...")
    vignettes = []
    with open(vignette_path, 'r', encoding='utf-8') as f:
        for line in f:
            v = json.loads(line)
            if v['dataset'] in datasets:
                vignettes.append(v)
    print(f"Loaded {len(vignettes)} vignettes")
    return vignettes

# Clean the responses(remove everything but alphanumeric and spaces)
def clean_text(text):
    return ' '.join(c for c in text if c.isalnum() or c.isspace()).lower().strip()

# Classify the model response as deceptive, honest, or invalid
def classify_response(response, response_a, response_b):
    # Clean
    target_dec = clean_text(response_a)
    target_hon = clean_text(response_b)
    response_clean = clean_text(response)
    
    # Classify
    if target_dec == response_clean:
        return 'deceptive'
    elif target_hon == response_clean:
        return 'honest'
    else:
        return 'invalid'
    
class MistralAttentionHeadExtractor:
    def __init__(self):
        print("Loading extractor model and tokenizer ...")
        self.tokenizer = load_tokenizer()
        self.model = load_model()
        self.model.eval()
        self.num_layers = self.model.config.num_hidden_layers
        self.num_heads = self.model.config.num_attention_heads
        self.head_dim = self.model.config.hidden_size // self.num_heads

    def get_num_layers(self):
        return self.num_layers
    
    def get_num_heads(self):
        return self.num_heads
    
    def get_head_dim(self):
        return self.head_dim

    def extract_head_activations(self, text: str):
        """
        Returns a list of size L x H x D containing attention head activations
        at the final token.
        """
        # Tokenize the input text
        inputs = tokenize_input(text, self.tokenizer).to(self.model.device)
        
        # Hook function to capture attention head activations
        head_activations = {}
        
        def attention_hook(module, input, output, layer_idx):
            # For Mistral, the attention output is a tuple: (attention_output, attention_weights, ...)
            # We want the attention output after the head projection
            attn_output = output[0]  # Shape: (batch_size, seq_len, hidden_size)
            
            # Reshape to separate heads: (batch_size, seq_len, num_heads, head_dim)
            batch_size, seq_len, hidden_size = attn_output.shape
            attn_output_heads = attn_output.view(batch_size, seq_len, self.num_heads, self.head_dim)
            
            # Get activations at the final token position
            final_token_idx = inputs['attention_mask'].sum(dim=1) - 1  # Last non-padding token
            final_activations = attn_output_heads[0, final_token_idx[0]]  # Shape: (num_heads, head_dim)
            
            head_activations[layer_idx] = final_activations.detach().cpu().numpy()
        
        # Register hooks for all attention layers
        hooks = []
        for layer_idx in range(self.num_layers):
            layer = self.model.model.layers[layer_idx].self_attn
            hook = layer.register_forward_hook(
                lambda module, input, output, idx=layer_idx: attention_hook(module, input, output, idx)
            )
            hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            _ = self.model(**inputs)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Flatten activations into a single array
        flattened_activations = []
        for layer_idx in range(self.num_layers):
            layer_activations = head_activations[layer_idx]  # Shape: (num_heads, head_dim)
            for head_idx in range(self.num_heads):
                for dim_idx in range(self.head_dim):
                    flattened_activations.append(float(layer_activations[head_idx, dim_idx]))
        
        return np.array(flattened_activations, dtype=np.float32)


def top_heads(target):
    file_dir = os.path.dirname(os.path.abspath(__file__))
    top_heads_file = 'top_heads_targets_c.json' if target == 'c' else 'top_heads_targets_p.json'
    path = os.path.join(file_dir, '..', 'data', top_heads_file)
    with open(path, "r", encoding="utf-8") as f:
        heads_list = json.load(f)
    return [(int(layer), int(head)) for layer, head in heads_list]

