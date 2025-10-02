"""
This script forward-passes the vignettes from training_vignettes.jsonl and testing_vignettes.jsonl
through Mistral-7B-v0.3, captures activations, and saves them in an efficient format.
"""
import json
import numpy as np
import h5py
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple, Dict

MAX_LENGTH = 256
# MODEL_NAME = "mistralai/Mistral-7B-v0.3"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
BATCH_SIZE = 10  # Save every 10 processed vignettes
OUTPUT_FILE = "vignette_activations.h5"

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
        
        # Flatten activations into a single array
        flattened_activations = []
        for layer_idx in range(self.num_layers):
            layer_activations = head_activations[layer_idx]  # Shape: (num_heads, head_dim)
            for head_idx in range(self.num_heads):
                for dim_idx in range(self.head_dim):
                    flattened_activations.append(float(layer_activations[head_idx, dim_idx]))
        
        return np.array(flattened_activations, dtype=np.float32)

def load_vignettes(filename: str) -> Tuple[List[str], List[float], List[float], List[int], List[str]]:
    """Load vignettes from JSONL file."""
    ids, vignettes, targets_p, targets_c, template_ids, datasets = [], [], [], [], [], []
    
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    for line in lines:
        data = json.loads(line)
        ids.append(data["id"])
        vignettes.append(data["vignette"])
        targets_p.append(data["target_p"])
        targets_c.append(data["target_c"])
        template_ids.append(data["template_id"])
        datasets.append(data["dataset"])
    
    return ids, vignettes, targets_p, targets_c, template_ids, datasets

def get_last_processed_index(filename: str) -> int:
    """Get the highest index from existing HDF5 file."""
    if not os.path.exists(filename):
        return -1
    
    try:
        with h5py.File(filename, 'r') as f:
            if 'vignette_ids' in f:
                return len(f['vignette_ids']) - 1
            else:
                return -1
    except:
        return -1

def save_batch_to_hdf5(filename: str, batch_data: Dict, append: bool = True):
    """Save a batch of data to HDF5 file."""
    mode = 'a' if append and os.path.exists(filename) else 'w'
    
    with h5py.File(filename, mode) as f:
        if mode == 'w':
            # Create datasets for the first time
            batch_size = len(batch_data['vignette_ids'])
            activation_shape = batch_data['activations'][0].shape[0]
            
            # Convert IDs to strings and datasets to bytes for HDF5 compatibility
            vignette_ids_str = [str(vid) for vid in batch_data['vignette_ids']]
            datasets_str = [str(ds).encode('utf-8') for ds in batch_data['datasets']]
            
            f.create_dataset('vignette_ids', data=vignette_ids_str, 
                           maxshape=(None,), dtype=h5py.string_dtype())
            f.create_dataset('targets_p', data=batch_data['targets_p'], 
                           maxshape=(None,), dtype=np.float32)
            f.create_dataset('targets_c', data=batch_data['targets_c'], 
                           maxshape=(None,), dtype=np.float32)
            f.create_dataset('template_ids', data=batch_data['template_ids'], 
                           maxshape=(None,), dtype=np.int32)
            f.create_dataset('datasets', data=datasets_str, 
                           maxshape=(None,), dtype=h5py.string_dtype())
            f.create_dataset('activations', data=np.array(batch_data['activations']), 
                           maxshape=(None, activation_shape), dtype=np.float32)
        else:
            # Append to existing datasets
            current_size = len(f['vignette_ids'])
            new_size = current_size + len(batch_data['vignette_ids'])
            
            # Resize all datasets
            f['vignette_ids'].resize((new_size,))
            f['targets_p'].resize((new_size,))
            f['targets_c'].resize((new_size,))
            f['template_ids'].resize((new_size,))
            f['datasets'].resize((new_size,))
            f['activations'].resize((new_size, f['activations'].shape[1]))
            
            # Convert data to appropriate formats
            vignette_ids_str = [str(vid) for vid in batch_data['vignette_ids']]
            datasets_str = [str(ds).encode('utf-8') for ds in batch_data['datasets']]
            
            # Add new data
            f['vignette_ids'][current_size:] = vignette_ids_str
            f['targets_p'][current_size:] = batch_data['targets_p']
            f['targets_c'][current_size:] = batch_data['targets_c']
            f['template_ids'][current_size:] = batch_data['template_ids']
            f['datasets'][current_size:] = datasets_str
            f['activations'][current_size:] = np.array(batch_data['activations'])

def main():
    extractor = MistralAttentionHeadExtractor()

    # Load vignettes
    print("Loading vignettes...")
    ids, vignettes, targets_p, targets_c, template_ids, datasets = load_vignettes("vignettes.jsonl")
    
    # Find where to continue processing
    last_processed = get_last_processed_index(OUTPUT_FILE)
    start_idx = last_processed + 1
    
    if start_idx > 0:
        print(f"Continuing from vignette {start_idx} (found {last_processed + 1} existing entries)")
    else:
        print("Starting fresh processing")
    
    # Process vignettes in batches
    print("Extracting features...")
    batch_data = {
        'vignette_ids': [],
        'targets_p': [],
        'targets_c': [],
        'template_ids': [],
        'datasets': [],
        'activations': []
    }
    
    for i in range(start_idx, len(vignettes)):
        print(f"Processing vignette {i + 1}/{len(vignettes)} (ID: {ids[i]})")
        
        # Extract activations
        activations = extractor.extract_head_activations(vignettes[i])
        
        # Add to batch
        batch_data['vignette_ids'].append(ids[i])
        batch_data['targets_p'].append(targets_p[i])
        batch_data['targets_c'].append(targets_c[i])
        batch_data['template_ids'].append(template_ids[i])
        batch_data['datasets'].append(datasets[i])
        batch_data['activations'].append(activations)
        
        # Save batch when it reaches BATCH_SIZE or at the end
        if len(batch_data['vignette_ids']) >= BATCH_SIZE or i == len(vignettes) - 1:
            print(f"Saving batch of {len(batch_data['vignette_ids'])} vignettes...")
            append = (i != start_idx) or (start_idx > 0)  # Append unless this is the very first batch of a fresh start
            save_batch_to_hdf5(OUTPUT_FILE, batch_data, append=append)
            
            # Clear batch data to free memory
            batch_data = {
                'vignette_ids': [],
                'targets_p': [],
                'targets_c': [],
                'template_ids': [],
                'datasets': [],
                'activations': []
            }
            print("Batch saved and memory cleared.")
    
    print(f"Processing complete! Data saved to {OUTPUT_FILE}")
    
    # Print summary
    with h5py.File(OUTPUT_FILE, 'r') as f:
        total_entries = len(f['vignette_ids'])
        activation_dims = f['activations'].shape[1]
        print(f"Total entries: {total_entries}")
        print(f"Activation dimensions per entry: {activation_dims}")
        print(f"File size: {os.path.getsize(OUTPUT_FILE) / (1024*1024):.1f} MB")

if __name__ == "__main__":
    main()