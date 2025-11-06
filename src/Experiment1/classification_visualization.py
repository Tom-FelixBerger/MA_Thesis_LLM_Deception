"""Visualization utilities for Experiment 1 baseline classifications."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
PLOTS_DIR = BASE_DIR / "plots" / "Experiment1"

MODEL_KEYS: List[str] = [
    "mistral",
    "gemma-2-2b",
    "gemma-2-9b",
    "llama-3.1-8b-instruct",
]

CLASSIFICATIONS = ("honest", "deceptive", "invalid")


def _load_records(path: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    if not path.exists():
        raise FileNotFoundError(f"Missing baseline assessment file: {path}")

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _classification_counts(records: Iterable[Dict[str, object]]) -> Dict[int, Dict[str, int]]:
    template_counts: Dict[int, Dict[str, int]] = defaultdict(
        lambda: {c: 0 for c in CLASSIFICATIONS}
    )

    for record in records:
        template_id = record.get("template_id")
        if template_id is None:
            continue
        classification = str(record.get("classification", "")).lower()
        if classification not in CLASSIFICATIONS:
            classification = "invalid"
        template_counts[int(template_id)][classification] += 1

    return template_counts


def _plot_model_counts(model_key: str, counts: Dict[int, Dict[str, int]]) -> None:
    plot_data = []
    for template_id, class_counts in sorted(counts.items()):
        for classification in CLASSIFICATIONS:
            plot_data.append(
                {
                    "template_id": template_id,
                    "classification": classification.title(),
                    "count": class_counts.get(classification, 0),
                }
            )

    df = pd.DataFrame(plot_data)

    plt.figure(figsize=(14, 6))
    sns.barplot(data=df, x="template_id", y="count", hue="classification")
    plt.title(f"Baseline classification counts by template — {model_key}")
    plt.xlabel("Template ID")
    plt.ylabel("Number of responses")
    plt.xticks(rotation=90)
    plt.tight_layout()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = PLOTS_DIR / f"{model_key}_classification_by_template.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved plot to {plot_path}")


def _templates_all_honest(counts: Dict[int, Dict[str, int]]) -> Set[int]:
    all_honest: Set[int] = set()
    for template_id, class_counts in counts.items():
        total = sum(class_counts.values())
        if total > 0 and class_counts.get("honest", 0) == total:
            all_honest.add(template_id)
    return all_honest


def main() -> None:
    honest_sets: List[Set[int]] = []

    for model_key in MODEL_KEYS:
        data_path = DATA_DIR / f"{model_key}_responses_baseline_assessment.jsonl"
        print(f"Processing {data_path} ...")
        records = _load_records(data_path)
        counts = _classification_counts(records)
        _plot_model_counts(model_key, counts)
        honest_sets.append(_templates_all_honest(counts))

    if honest_sets:
        honest_all_models = sorted(set.intersection(*honest_sets))
    else:
        honest_all_models = []

    print("Templates answered honestly by all models (100% honest across models):")
    print(honest_all_models)


if __name__ == "__main__":
    main()
