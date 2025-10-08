"""
This script loads vignettes, generates model responses, classifications,
and saves results incrementally to 'model_responses_baseline_assessment.jsonl'.
It uses substring matching for robust classification of the model's response.
"""
import json
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import utils

SAVE_INTERVAL = 10 
OUTPUT_FILENAME = '..\..\data\model_responses_baseline_assessment.jsonl'
VIGNETTE_PATH = '..\..\data\\vignettes.jsonl'

def filter_processed(vignettes):
    # Check for existing results to resume work
    processed_ids = set()
    if os.path.exists(OUTPUT_FILENAME):
        print(f"Resuming: Checking {OUTPUT_FILENAME} for already processed vignettes...")
        with open(OUTPUT_FILENAME, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                processed_ids.add(data.get('id'))
        print(f"Found {len(processed_ids)} already processed results.")

    # Filter out already processed vignettes
    vignettes_to_process = [item for item in vignettes if item['id'] not in processed_ids]
    total_to_process = len(vignettes_to_process)
    start_index = len(vignettes) - total_to_process

    return vignettes_to_process, total_to_process, start_index

def main():
    tokenizer = utils.load_tokenizer()
    model = utils.load_model()
    vignettes = utils.load_vignettes([utils.DATASET_NAMES[s] for s in ['excluded', 'assessment', 'additional']])
    vignettes_to_process, total_to_process, start_index = filter_processed(vignettes)

    if total_to_process == 0:
        print("All vignettes already processed. Exiting.")
        return

    print(f"Starting processing from vignette ID {start_index + 1} (Total new to process: {total_to_process})")
    # Process each vignette
    results_buffer = []
    
    for i, item in enumerate(vignettes_to_process):
        current_index = start_index + i + 1
        
        vignette_text = item['scenario'] + item["instruction"]
        response_a = item['response_a'] # Deceptive response (full string from JSON)
        response_b = item['response_b'] # Honest response (full string from JSON)

        # Print Prompt
        print("\n" + "="*80)
        print(f"Processing {current_index}/{len(vignettes)} (ID: {item['id']}):")
        print("PROMPT:")
        print(vignette_text.strip())
        print("-" * 80)
        
        # Generate model response
        full_response, only_new = utils.generate_text(model, tokenizer, vignette_text)
        
        # Classify response using substring matching
        classification = utils.classify_response(only_new, response_a, response_b)
        
        # Print Model Answer and Classification
        print(f"MODEL RESPONSE:\n'{only_new}'")
        print(f"CLASSIFICATION: {classification.upper()}")
        print("=" * 80)
        
        # Store result
        result = {
            'id': item['id'],
            'vignette': vignette_text,
            'template_id': item['template_id'],
            'model_response_raw': only_new, 
            'classification': classification
        }
        results_buffer.append(result)
        
        # Incremental Save
        if len(results_buffer) >= SAVE_INTERVAL or i == total_to_process - 1:
            print(f"Saving {len(results_buffer)} results to {OUTPUT_FILENAME}...")
            with open(OUTPUT_FILENAME, 'a', encoding='utf-8') as f:
                for buffered_result in results_buffer:
                    f.write(json.dumps(buffered_result, ensure_ascii=False) + '\n')
            results_buffer = []

if __name__ == '__main__':
    main()