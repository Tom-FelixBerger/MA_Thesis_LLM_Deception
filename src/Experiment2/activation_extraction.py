"""
This script forward-passes the vignettes from training_vignettes.jsonl and testing_vignettes.jsonl
through Mistral-7B-v0.3, captures activations, and saves them in an efficient format.
"""
import numpy as np
import h5py
import os
from typing import List, Tuple, Dict
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import utils


BATCH_SIZE = 10  # Save every 10 processed vignettes
OUTPUT_FILE = "..\\..\\data\\vignette_activations.h5"

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
    extractor = utils.MistralAttentionHeadExtractor()
    vignettes = utils.load_vignettes([utils.DATASET_NAMES['probe_train'], utils.DATASET_NAMES['probe_validate'], utils.DATASET_NAMES['probe_test']])
    
    # Find where to continue processing
    try:
        with h5py.File(OUTPUT_FILE, 'r') as f:
            start_idx = len(f['vignette_ids'])
    except FileNotFoundError:
        start_idx = -1
    
    if start_idx > 0:
        print(f"Continuing from vignette {start_idx}/{len(vignettes)}")
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
        print(f"Processing vignette {i + 1}/{len(vignettes)} (ID: {vignettes[i]['id']})")
        
        # Extract activations
        prompt = vignettes[i]['scenario'] + f"You respond: {vignettes[i]['response']}\n" + vignettes[i]['question']
        activations = extractor.extract_head_activations(prompt)
        
        # Add to batch
        batch_data['vignette_ids'].append(vignettes[i]['id'])
        batch_data['targets_p'].append(vignettes[i]['target_p'])
        batch_data['targets_c'].append(vignettes[i]['target_c'])
        batch_data['template_ids'].append(vignettes[i]['template_id'])
        batch_data['datasets'].append(vignettes[i]['dataset'])
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