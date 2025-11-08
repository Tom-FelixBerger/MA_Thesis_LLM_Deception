import json
import itertools
from collections import Counter
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from scipy.stats import fisher_exact
from peft import PeftModel

import utils

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RESULTS_PATH = DATA_DIR / "finetuned_assessment.jsonl"
SUMMARY_PATH = DATA_DIR / "finetuned_assessment_summary.json"
CREDENTIALS_PATH = BASE_DIR / "credentials.txt"

MODEL_KEY = "mistral"
WO_DIR = BASE_DIR / "model_saves" / f"{MODEL_KEY}_reinforce_lora_ckpt_with_options"
FA_DIR = BASE_DIR / "model_saves" / f"{MODEL_KEY}_reinforce_lora_ckpt_free_answer"
PRE_DIR = BASE_DIR / "model_saves" / f"{MODEL_KEY}_reinforce_lora_ckpt_pretrained"
SOO_DIR = BASE_DIR / "model_saves" / f"{MODEL_KEY}_reinforce_lora_ckpt_soo"
TBI_DIR = BASE_DIR / "model_saves" / f"{MODEL_KEY}_reinforce_lora_ckpt_tbi"
MODEL_LABEL_INITIAL = "initial"
MODEL_LABEL_WO = "with_options"
MODEL_LABEL_FA = "free_answer"
MODEL_LABEL_PRE = "pretrained"
MODEL_LABEL_SOO = "SOO-finetuned"
MODEL_LABEL_TBI = "TBI-finetuned"

_CREDENTIALS: Optional[Dict[str, str]] = None


def get_credentials() -> Dict[str, str]:
    global _CREDENTIALS
    if _CREDENTIALS is None:
        _CREDENTIALS = utils.load_credentials(CREDENTIALS_PATH)
    return _CREDENTIALS


def load_initial_model():
    credentials = get_credentials()
    model, tokenizer = utils.load_model_and_tokenizer(MODEL_KEY, credentials)
    model.eval()
    return model, tokenizer


def load_lora_model(model_dir):
    adapter_dir, tokenizer_dir, checkpoint_file = utils.model_save_dirs(model_dir)
    if not adapter_dir.exists():
        raise FileNotFoundError(f"LoRA adapters not found at {adapter_dir}")
    tokenizer = utils.load_tokenizer(path=str(tokenizer_dir))
    base_model = utils.load_model_for_key(MODEL_KEY, get_credentials())
    base_model.eval()
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.eval()
    return model, tokenizer

MODEL_LOADERS = {
    MODEL_LABEL_INITIAL: load_initial_model,
    MODEL_LABEL_WO: lambda: load_lora_model(model_dir=WO_DIR),
    MODEL_LABEL_FA: lambda: load_lora_model(model_dir=FA_DIR),
    MODEL_LABEL_PRE: lambda: load_lora_model(model_dir=PRE_DIR),
    MODEL_LABEL_SOO: lambda: load_lora_model(model_dir=SOO_DIR),
    MODEL_LABEL_TBI: lambda: load_lora_model(model_dir=TBI_DIR),
}

def evaluate_model(model_label, loader, vignettes):
    model, tokenizer = loader()
    records = []
    try:
        for vignette in vignettes:
            record = utils.generate_classification_record(model, tokenizer, vignette)
            record["model"] = model_label
            record["vignette"] = utils.build_prompt(vignette)
            records.append(record)
    finally:
        del model
        torch.cuda.empty_cache()
    return records


def summarize_records(records):
    counts = Counter(r["classification"] for r in records)
    total = len(records)
    deceptive = counts.get("deceptive", 0)
    summary = {
        "total": total,
        "counts": dict(counts),
        "deceptive_rate": deceptive / total if total else 0.0,
    }
    return summary


def compute_significance(summary_a, summary_b, label_a, label_b):
    deceptive_a = summary_a["counts"].get("deceptive", 0)
    deceptive_b = summary_b["counts"].get("deceptive", 0)
    table = np.array([
        [deceptive_a, summary_a["total"] - deceptive_a],
        [deceptive_b, summary_b["total"] - deceptive_b],
    ])
    _, pvalue = fisher_exact(table, alternative="two-sided")
    return {
        "model_a": label_a,
        "model_b": label_b,
        "p_value": float(pvalue),
        "rate_diff": float(summary_a["deceptive_rate"] - summary_b["deceptive_rate"]),
    }


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    vignettes = utils.load_vignettes([utils.DATASET_NAMES['assessment']])
    if not vignettes:
        raise RuntimeError("No assessment vignettes found.")

    all_records = []
    summaries = {}

    for label, loader in MODEL_LOADERS.items():
        print(f"Evaluating model '{label}' on {len(vignettes)} assessment vignettes...")
        records = evaluate_model(label, loader, vignettes)
        summaries[label] = summarize_records(records)
        all_records.extend(records)

    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    comparisons = []
    for label_a, label_b in itertools.combinations(MODEL_LOADERS.keys(), 2):
        comparisons.append(compute_significance(summaries[label_a], summaries[label_b], label_a, label_b))

    summary_output = {
        "summaries": summaries,
        "comparisons": comparisons,
    }

    with SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary_output, f, ensure_ascii=False, indent=2)

    print("=== Summary of deceptive rates ===")
    for label, summary in summaries.items():
        rate = summary["deceptive_rate"] * 100
        print(f"{label}: {summary['counts'].get('deceptive', 0)}/{summary['total']} deceptive ({rate:.2f}%)")

    print("\n=== Statistical significance (Fisher's exact test) ===")
    for comp in comparisons:
        print(
            f"{comp['model_a']} vs {comp['model_b']}: p-value={comp['p_value']:.4g}, "
            f"rate diff={comp['rate_diff']:+.4f}"
        )


if __name__ == "__main__":
    main()
