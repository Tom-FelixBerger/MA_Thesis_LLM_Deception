"""
This script forward-passes the vignettes from training_vignettes.jsonl and testing_vignettes.jsonl
through Mistral-7B-v0.3, captures activations, and saves them in an efficient format.
"""
import h5py
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import utils


BATCH_SIZE = 10  # Save every 10 processed vignettes
OUTPUT_FILE = "..\\..\\data\\vignette_activations.h5"

def main():
    extractor = utils.MistralAttentionHeadExtractor()
    vignettes = utils.load_vignettes([utils.DATASET_NAMES['probe_train'], utils.DATASET_NAMES['probe_validate'], utils.DATASET_NAMES['probe_test']])
    
    # Find where to continue processing
    start_idx = utils.count_saved_activations(OUTPUT_FILE)

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
    
    metadata = utils.extractor_metadata(extractor)

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
            utils.save_activation_batch(OUTPUT_FILE, batch_data, metadata, append=append)
            
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