"""
This script forward-passes the vignettes from training_vignettes.jsonl and testing_vignettes.jsonl
through Mistral-7B-v0.3, captures activations, and saves them with targets as a .csv file.
"""
import json
import numpy as np
import pandas as pd
import torch
import random
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple, Dict

MAX_LENGTH = 256
MODEL_NAME = "mistralai/Mistral-7B-v0.3"

class MistralAttentionHeadExtractor:
    def __init__(self, model_name=MODEL_NAME):
        print("Loading tokenizer and model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager"
        )
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
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH
        )
        
        # Move inputs to the same device as the model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
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
        
        # Flatten activations into a single list
        flattened_activations = []
        for layer_idx in range(self.num_layers):
            layer_activations = head_activations[layer_idx]  # Shape: (num_heads, head_dim)
            for head_idx in range(self.num_heads):
                for dim_idx in range(self.head_dim):
                    flattened_activations.append(float(layer_activations[head_idx, dim_idx]))
        
        return flattened_activations

def load_vignettes(filename: str) -> Tuple[List[str], List[float], List[float]]:
    """Load vignettes from JSONL file with optional sampling."""
    ids, vignettes, targets_p, targets_c, datasets = [], [], [], [], []
    
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    for line in lines:
        data = json.loads(line)
        ids.append(data["id"])
        vignettes.append(data["vignette"])
        targets_p.append(data["target_p"])
        targets_c.append(data["target_c"])
        datasets.append(data["dataset"])
    
    return ids, vignettes, targets_p, targets_c, datasets

def main():
    extractor = MistralAttentionHeadExtractor()

    # Load training vignettes
    print("Loading vignettes...")
    ids, vignettes, targets_p, targets_c, datasets = load_vignettes("vignettes.jsonl")
    
    # Load existing CSV or create new DataFrame
    try:
        df = pd.read_csv("vignette_activations.csv")
        print(f"Loaded existing vignette_activations.csv with {len(df)} entries.")
    except FileNotFoundError:
        print("No existing vignette_activations.csv found, creating a new one.")

        feature_names = ['vignette_id', 'target_p', 'target_c', 'dataset']
        for layer_idx in range(extractor.get_num_layers()):
            for head_idx in range(extractor.get_num_heads()):
                for dim in range(extractor.get_head_dim()):
                    feature_names.append(f"layer_{layer_idx}_head_{head_idx}_dim_{dim}_attention")
        df = pd.DataFrame(columns=feature_names)
    
    # Extract activation features for all vignettes
    print("Extracting features ...")
    for id in range(len(df), len(vignettes)):
        print(f"Processing vignette {id + 1}/{len(vignettes)}...")
        features = extractor.extract_head_activations(vignettes[id])
        row = [ids[id], targets_p[id], targets_c[id], datasets[id]] + features
        df.loc[len(df)] = row
        if (id + 1) % 10 == 0 or (id + 1) == len(vignettes):
            print("Saving to CSV...")
            df.to_csv("vignette_activations.csv", index=False)

if __name__ == "__main__":
    main()