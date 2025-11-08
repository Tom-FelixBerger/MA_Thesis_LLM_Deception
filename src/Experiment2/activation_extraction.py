"""Extract attention head activations for Experiment 2 probes."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import h5py
import torch

sys.path.append(str(Path(__file__).parent.parent))
import utils


BATCH_SIZE = 10
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
EXPERIMENT1_DIR = DATA_DIR / "experiment1"
EXPERIMENT2_DIR = DATA_DIR / "experiment2"
SIGNIFICANT_MODELS_PATH = EXPERIMENT1_DIR / "significant_models.json"
SIGNIFICANCE_REPORT_PATH = EXPERIMENT1_DIR / "significance_tests.txt"
CREDENTIALS_PATH = BASE_DIR / "credentials.txt"
TARGET_MODEL_KEYS: tuple[str, ...] = (
    "mistral",
    "gemma-2-9b",
    "llama-3.1-8b-instruct",
)
DATASET_LABELS: tuple[str, ...] = (
    utils.DATASET_NAMES["probe_train"],
    utils.DATASET_NAMES["probe_validate"],
    utils.DATASET_NAMES["probe_test"],
)


def _load_vignettes() -> list[dict]:
    print("Loading vignettes...")
    return utils.load_vignettes(list(DATASET_LABELS))


def _build_prompt(vignette: dict) -> str:
    return (
        vignette["scenario"]
        + f"You respond: {vignette['response']}\n"
        + vignette["question"]
    )


def _process_model(
    model_key: str,
    vignettes: Iterable[dict],
    credentials: dict,
) -> None:
    output_path = EXPERIMENT2_DIR / f"{model_key}_activations.h5"
    config = utils.MODEL_CONFIGS[model_key]

    extractor = utils.ModelAttentionHeadExtractor(model_key, credentials)
    metadata = utils.extractor_metadata(extractor)
    metadata["model_key"] = model_key
    metadata["model_id"] = str(config["model_id"])
    metadata["feature_names"] = utils.build_feature_names(
        metadata["num_layers"], metadata["num_heads"], metadata["head_dim"]
    )

    total_vignettes = len(vignettes)
    start_idx = utils.count_saved_activations(output_path)
    if start_idx >= total_vignettes:
        print(f"All {total_vignettes} vignettes already processed for {model_key}.")
        return

    print(
        f"Processing {total_vignettes - start_idx} remaining vignettes for model '{model_key}'."
    )

    batch_data = {
        "vignette_ids": [],
        "targets_p": [],
        "targets_c": [],
        "template_ids": [],
        "datasets": [],
        "activations": [],
    }

    append_existing = start_idx > 0

    for idx in range(start_idx, total_vignettes):
        vignette = vignettes[idx]
        prompt = _build_prompt(vignette)
        activations = extractor.extract_head_activations(prompt)

        batch_data["vignette_ids"].append(vignette["id"])
        batch_data["targets_p"].append(vignette["target_p"])
        batch_data["targets_c"].append(vignette["target_c"])
        batch_data["template_ids"].append(vignette["template_id"])
        batch_data["datasets"].append(vignette["dataset"])
        batch_data["activations"].append(activations)

        if len(batch_data["vignette_ids"]) >= BATCH_SIZE or idx == total_vignettes - 1:
            print(
                f"Saving batch of {len(batch_data['vignette_ids'])} vignettes "
                f"(up to index {idx + 1}/{total_vignettes})"
            )
            utils.save_activation_batch(
                output_path,
                batch_data,
                metadata,
                append=append_existing,
            )
            batch_data = {
                "vignette_ids": [],
                "targets_p": [],
                "targets_c": [],
                "template_ids": [],
                "datasets": [],
                "activations": [],
            }
            append_existing = True

    del extractor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    with h5py.File(output_path, "r") as handle:
        total_entries = len(handle["vignette_ids"])
        activation_dims = handle["activations"].shape[1]
        print(
            f"Model '{model_key}' extraction complete: {total_entries} entries, "
            f"{activation_dims} features."
        )
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"File size: {size_mb:.1f} MB -> {output_path}")


def main() -> None:
    EXPERIMENT2_DIR.mkdir(parents=True, exist_ok=True)

    credentials = utils.load_credentials(CREDENTIALS_PATH)
    significant_models = utils.load_significant_models(
        SIGNIFICANT_MODELS_PATH, SIGNIFICANCE_REPORT_PATH
    )
    eligible_models = [model for model in TARGET_MODEL_KEYS if model in significant_models]

    if not eligible_models:
        print("No models exceeded the deception significance threshold. Nothing to do.")
        return

    ignored = sorted(set(significant_models) - set(eligible_models))
    if ignored:
        print(
            "Skipping models without Experiment 2 support: "
            + ", ".join(ignored)
        )

    vignettes = _load_vignettes()
    if not vignettes:
        print("No vignettes found for the requested datasets. Aborting.")
        return

    for model_key in eligible_models:
        print("\n" + "=" * 80)
        print(f"Extracting activations for model: {model_key}")
        print("=" * 80)
        _process_model(model_key, vignettes, credentials)


if __name__ == "__main__":
    main()
