"""Baseline deception assessment that supports multiple model providers."""

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Dict, List

sys.path.append(str(Path(__file__).parent.parent))
import utils

SAVE_INTERVAL = 10
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
DEFAULT_CREDENTIALS_PATH = BASE_DIR / "credentials.txt"

MODEL_CONFIGS: Dict[str, Dict[str, object]] = {
    "mistral": {
        "type": "huggingface",
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "quantized": True,
        "device_map": "cuda",
        "attn_implementation": "eager",
        "required_credentials": ["HUGGINGFACE_TOKEN"],
    },
    "gemma": {
        "type": "huggingface",
        "model_id": "google/gemma-3-4b-it",
        "quantized": True,
        "device_map": "cuda",
        "attn_implementation": "eager",
        "required_credentials": ["HUGGINGFACE_TOKEN"],
    },
    "o3-mini": {
        "type": "openai",
        "model_id": "o3-mini",
        "temperature": 0.7,
        "max_output_tokens": 128,
        "required_credentials": ["OPENAI_API_KEY"],
    },
}


def load_credentials(credentials_path: Path) -> Dict[str, str]:
    if not credentials_path.exists():
        raise FileNotFoundError(
            f"Credentials file not found at {credentials_path}. "
            "Copy credentials.txt.stub to credentials.txt and fill in your tokens."
        )

    credentials: Dict[str, str] = {}
    with credentials_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            credentials[key.strip()] = value.strip()
    return credentials


def ensure_required_credentials(model_key: str, credentials: Dict[str, str]) -> None:
    config = MODEL_CONFIGS[model_key]
    required_keys = config.get("required_credentials", [])
    missing = [key for key in required_keys if not credentials.get(key)]
    if missing:
        missing_keys = ", ".join(missing)
        raise RuntimeError(
            f"Missing credentials for model '{model_key}'. Required keys: {missing_keys}. "
            "Add them to credentials.txt."
        )


def create_huggingface_record_generator(model_key: str, credentials: Dict[str, str]) -> Callable[[dict], dict]:
    config = MODEL_CONFIGS[model_key]
    token = credentials.get("HUGGINGFACE_TOKEN")
    model_id = config["model_id"]
    tokenizer = utils.load_tokenizer(path=model_id, token=token)
    model = utils.load_model(
        model_name=model_id,
        quantized=bool(config.get("quantized", True)),
        device_map=str(config.get("device_map", "cuda")),
        token=token,
        attn_implementation=str(config.get("attn_implementation", "eager")),
    )

    def generator(vignette: dict) -> dict:
        return utils.generate_classification_record(model, tokenizer, vignette)

    return generator


def create_openai_record_generator(model_key: str, credentials: Dict[str, str]) -> Callable[[dict], dict]:
    config = MODEL_CONFIGS[model_key]
    api_key = credentials.get("OPENAI_API_KEY")
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError("The openai package is required to use OpenAI models.") from exc

    client = OpenAI(api_key=api_key)
    model_id = str(config["model_id"])
    temperature = float(config.get("temperature", 0.7))
    max_output_tokens = int(config.get("max_output_tokens", 128))

    def generator(vignette: dict) -> dict:
        prompt = utils.build_vignette_prompt(vignette)
        response = client.responses.create(
            model=model_id,
            input=prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        output_text = getattr(response, "output_text", None)
        if output_text is None:
            chunks = []
            for item in getattr(response, "output", []) or []:
                for content in getattr(item, "content", []) or []:
                    text = getattr(content, "text", None)
                    if text:
                        chunks.append(text)
            output_text = "".join(chunks)
        only_new = (output_text or "").strip()
        classification = utils.classify_response(only_new, vignette["response_a"], vignette["response_b"])
        return {
            "id": vignette["id"],
            "template_id": vignette.get("template_id"),
            "prompt": prompt,
            "model_response_raw": only_new,
            "classification": classification,
        }

    return generator


def build_record_generator(model_key: str, credentials: Dict[str, str]) -> Callable[[dict], dict]:
    ensure_required_credentials(model_key, credentials)
    model_type = MODEL_CONFIGS[model_key]["type"]
    if model_type == "huggingface":
        return create_huggingface_record_generator(model_key, credentials)
    if model_type == "openai":
        return create_openai_record_generator(model_key, credentials)
    raise ValueError(f"Unsupported model type '{model_type}' for {model_key}")



def process_model(
    model_key: str,
    vignettes: List[dict],
    credentials: Dict[str, str],
) -> None:
    config = MODEL_CONFIGS[model_key]
    record_generator = build_record_generator(model_key, credentials)

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

        vignette_text = utils.build_vignette_prompt(item)

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
    credentials = load_credentials(DEFAULT_CREDENTIALS_PATH)
    vignettes = utils.load_vignettes([utils.DATASET_NAMES[s] for s in ["excluded", "assessment", "additional"]])

    for model_key in MODEL_CONFIGS:
        process_model(model_key, vignettes, credentials)


if __name__ == "__main__":
    main()
