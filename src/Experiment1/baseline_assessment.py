"""
This script loads vignettes, generates model responses, classifications,
and saves results incrementally to 'model_responses_baseline_assessment.jsonl'.
It uses substring matching for robust classification of the model's response.
"""
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import utils

SAVE_INTERVAL = 10
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / 'data'
OUTPUT_PATH = DATA_DIR / 'model_responses_baseline_assessment.jsonl'


def main():
    tokenizer = utils.load_tokenizer()
    model = utils.load_model()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    vignettes = utils.load_vignettes([utils.DATASET_NAMES[s] for s in ['excluded', 'assessment', 'additional']])
    vignettes_to_process, total_to_process, start_index = utils.filter_processed_vignettes(vignettes, OUTPUT_PATH)

    if total_to_process == 0:
        print("All vignettes already processed. Exiting.")
        return

    print(f"Starting processing from vignette ID {start_index + 1} (Total new to process: {total_to_process})")
    results_buffer = []

    for i, item in enumerate(vignettes_to_process):
        current_index = start_index + i + 1

        vignette_text = utils.build_vignette_prompt(item)
        response_a = item['response_a']
        response_b = item['response_b']

        print("\n" + "=" * 80)
        print(f"Processing {current_index}/{len(vignettes)} (ID: {item['id']}):")
        print("PROMPT:")
        print(vignette_text.strip())
        print("-" * 80)

        record = utils.generate_classification_record(model, tokenizer, item)
        only_new = record['model_response_raw']
        classification = record['classification']

        print(f"MODEL RESPONSE:\n'{only_new}'")
        print(f"CLASSIFICATION: {classification.upper()}")
        print("=" * 80)

        record['vignette'] = vignette_text
        results_buffer.append(record)

        if len(results_buffer) >= SAVE_INTERVAL or i == total_to_process - 1:
            print(f"Saving {len(results_buffer)} results to {OUTPUT_PATH}...")
            with OUTPUT_PATH.open('a', encoding='utf-8') as f:
                for buffered_result in results_buffer:
                    f.write(json.dumps(buffered_result, ensure_ascii=False) + '\n')
            results_buffer = []


if __name__ == '__main__':
    main()
