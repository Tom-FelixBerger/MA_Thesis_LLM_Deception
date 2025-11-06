"""Baseline deception assessment that supports multiple model providers."""

import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.append(str(Path(__file__).parent.parent))
import utils

SAVE_INTERVAL = 10
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
DEFAULT_CREDENTIALS_PATH = BASE_DIR / "credentials.txt"



def process_model(
    model_key: str,
    vignettes: List[dict],
    credentials: Dict[str, str],
) -> None:
    config = utils.MODEL_CONFIGS[model_key]
    record_generator = utils.create_classification_record_generator(model_key, credentials)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_DIR / f"{model_key}_responses_baseline_assessment.jsonl"

    print("#" * 80)
    print(f"Running baseline assessment with model '{model_key}' ({config['model_id']}).")
    vignettes_to_process, total_to_process, start_index = utils.filter_processed_vignettes(
        vignettes, output_path
    )

    if total_to_process == 0:
        print("All vignettes already processed for this model. Skipping.")
        return

    print(
        f"Starting processing from vignette ID {start_index + 1} (Total new to process: {total_to_process})"
    )
    results_buffer = []

    for i, item in enumerate(vignettes_to_process):
        current_index = start_index + i + 1

        vignette_text = utils.build_prompt(item)

        print("\n" + "=" * 80)
        print(f"Processing {current_index}/{len(vignettes)} (ID: {item['id']}):")
        print("PROMPT:")
        print(vignette_text.strip())
        print("-" * 80)

        record = record_generator(item)
        results_buffer.append(record)

        only_new = record["model_response_raw"]
        classification = record["classification"]
        print(f"MODEL RESPONSE:\n'{only_new}'")
        print(f"CLASSIFICATION: {classification.upper()}")
        print("=" * 80)

        if len(results_buffer) >= SAVE_INTERVAL or i == total_to_process - 1:
            print(f"Saving {len(results_buffer)} results to {output_path}...")
            with output_path.open("a", encoding="utf-8") as f:
                for buffered_result in results_buffer:
                    f.write(json.dumps(buffered_result, ensure_ascii=False) + "\n")
            results_buffer = []


def main():
    credentials = utils.load_credentials(DEFAULT_CREDENTIALS_PATH)
    vignettes = utils.load_vignettes([utils.DATASET_NAMES[s] for s in ["excluded", "baseline_assessment", "additional"]])

    for model_key in utils.MODEL_CONFIGS:
        process_model(model_key, vignettes, credentials)


if __name__ == "__main__":
    main()
