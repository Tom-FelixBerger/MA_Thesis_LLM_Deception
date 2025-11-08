"""Generate free-form deceptive (without lying) responses for assessment vignettes."""

import json
import sys
from pathlib import Path
from typing import Dict, List

import torch

sys.path.append(str(Path(__file__).parent.parent))
import utils

SAVE_INTERVAL = 10
MODEL_KEYS = ["mistral", "gemma-2-2b", "gemma-2-9b", "llama-3.1-8b-instruct"]
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
DEFAULT_CREDENTIALS_PATH = BASE_DIR / "credentials.txt"


def _ensure_instruction_available(vignette: dict) -> None:
    if "instruction_deception_no_lying" not in vignette:
        raise KeyError(
            "Vignette missing 'instruction_deception_no_lying'. "
            "Regenerate vignettes to include the new instruction field."
        )


def _process_model(
    model_key: str,
    vignettes: List[dict],
    credentials: Dict[str, str],
) -> None:
    config = utils.MODEL_CONFIGS[model_key]
    generator = utils.build_text_generation_backend(model_key, credentials)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_DIR / f"{model_key}_free_deception_without_lying.jsonl"

    print("#" * 80)
    print(f"Running free deception without lying with model '{model_key}' ({config['model_id']}).")

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

    for index, vignette in enumerate(vignettes_to_process):
        current_index = start_index + index + 1
        _ensure_instruction_available(vignette)
        prompt = utils.build_prompt(vignette, instruction_key="instruction_deception_no_lying")
        messages = utils.build_chat_messages(
            vignette,
            instruction_key="instruction_deception_no_lying",
        )

        print("\n" + "=" * 80)
        print(f"Processing {current_index}/{len(vignettes)} (ID: {vignette['id']}):")
        print("PROMPT:")
        print(prompt.strip())
        print("-" * 80)

        response_text = generator(messages)
        result = {
            "id": vignette["id"],
            "template_id": vignette.get("template_id"),
            "prompt": prompt,
            "model_response_raw": response_text,
        }
        results_buffer.append(result)

        print(f"MODEL RESPONSE:\n'{response_text}'")
        print("=" * 80)

        if len(results_buffer) >= SAVE_INTERVAL or index == total_to_process - 1:
            print(f"Saving {len(results_buffer)} results to {output_path}...")
            with output_path.open("a", encoding="utf-8") as handle:
                for buffered_result in results_buffer:
                    handle.write(json.dumps(buffered_result, ensure_ascii=False) + "\n")
            results_buffer = []

    if config["type"] == "huggingface":
        del generator  # release references to large models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    credentials = utils.load_credentials(DEFAULT_CREDENTIALS_PATH)
    vignettes = utils.load_vignettes([utils.DATASET_NAMES['baseline_assessment']])
    if not vignettes:
        raise RuntimeError("No assessment vignettes found. Run vignette_generation.py first.")

    for model_key in MODEL_KEYS:
        _process_model(model_key, vignettes, credentials)


if __name__ == "__main__":
    main()
