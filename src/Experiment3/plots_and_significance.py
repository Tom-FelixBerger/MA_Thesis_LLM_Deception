import json
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from scipy.stats import binomtest

from utils import utils

SIGNIFICANCE_LEVEL = 0.01

# Mapping for response files
MODEL_FILES = {
    "gemma-2-9b TBI finetuned": utils.DATA_DIR / "Experiment3" / "gemma-2-9b_tbi_finetuned_baseline_responses.jsonl",
    "llama-3.1-8b TBI finetuned": utils.DATA_DIR / "Experiment3" / "llama-3.1-8b_tbi_finetuned_baseline_responses.jsonl",

    "gemma-2-9b free finetuned": utils.DATA_DIR / "Experiment3" / "gemma-2-9b_free_finetuned_baseline_responses.jsonl",
    "llama-3.1-8b free finetuned": utils.DATA_DIR / "Experiment3" / "llama-3.1-8b_free_finetuned_baseline_responses.jsonl",
}

# Baseline deceptive counts (Experiment 1)
BASELINE_COUNTS_PATH = utils.DATA_DIR / "Experiment1" / "baseline_deceptive_counts.txt"




def load_counts(path):
    """Return dict: {'honest': X, 'deceptive': Y, 'invalid': Z}."""
    counts = {"honest": 0, "deceptive": 0, "invalid": 0}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            c = entry["classification"]
            counts[c] += 1

    return counts


def compute_significance(ft_counts, baseline_counts):
    base_prop, base_n = utils.deceptive_proportion(baseline_counts)
    ft_prop, ft_n = utils.deceptive_proportion(ft_counts)

    p_value = float(binomtest(
        k=ft_counts["deceptive"],
        n=ft_n,
        p=base_prop,
        alternative="less"
    ).pvalue)

    return {
        "finetuned_deceptive_proportion": ft_prop,
        "baseline_deceptive_proportion": base_prop,
        "p_value": p_value,
        "significant": p_value < SIGNIFICANCE_LEVEL,
    }

def plot_experiment3_barplot(model_counts, baseline):
    all_counts = model_counts | baseline

    deception_rates = {
        model: counts["deceptive"] / (
            counts["honest"] + counts["deceptive"] + counts["invalid"]
        )
        for model, counts in all_counts.items()
    }

    ordered = sorted(
        deception_rates.keys(),
        key=lambda m: deception_rates[m],
        reverse=True
    )

    deceptive_vals = [all_counts[m]["deceptive"] for m in ordered]
    honest_vals = [all_counts[m]["honest"] for m in ordered]
    invalid_vals = [all_counts[m]["invalid"] for m in ordered]

    plt.figure(figsize=(9, 6))

    colors = {
        "deceptive": "#F6A6A1",
        "honest": "#A6CEE3",
        "invalid": "#B0B0B0",
    }

    # Stacked bars
    deceptive_bars = plt.barh(
        ordered, deceptive_vals, label="Deceptive", color=colors["deceptive"]
    )
    honest_bars = plt.barh(
        ordered,
        honest_vals,
        left=deceptive_vals,
        label="Honest",
        color=colors["honest"],
    )

    left_vals = [d + h for d, h in zip(deceptive_vals, honest_vals)]
    invalid_bars = plt.barh(
        ordered,
        invalid_vals,
        left=left_vals,
        label="Invalid",
        color=colors["invalid"],
    )

    # Labels & axes
    plt.xlabel("Number of responses")
    plt.ylabel("Model")
    plt.title("Response Classifications by Model")
    plt.gca().invert_yaxis()

    plt.legend(loc="lower right", bbox_to_anchor=(1, 1))

    # Match annotation logic from plot_deceptive_counts
    for bars, values in (
        (deceptive_bars, deceptive_vals),
        (honest_bars, honest_vals),
        (invalid_bars, invalid_vals),
    ):
        for bar, v in zip(bars, values):
            if v > 0:
                x = bar.get_x() + bar.get_width() / 2
                y = bar.get_y() + bar.get_height() / 2
                plt.text(
                    x,
                    y,
                    str(v),
                    va="center",
                    ha="center",
                    color="black",
                )

    plt.tight_layout(pad=0.3)

    outpath = utils.PLOTS_DIR / "experiment3_deceptive_responses.png"
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"Saved plot to: {outpath}")


def write_significance_summary(summary_data):
    out_path = utils.DATA_DIR / "Experiment3" / "deception_significance_summary.json"
    out_path_txt = utils.DATA_DIR / "Experiment3" / "deception_significance_summary.txt"

    out_path.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")

    lines = ["Experiment 3 – Significance Results", ""]
    for item in summary_data:
        lines.extend([
            f"Model: {item['model']}",
            f"  Baseline deceptive proportion: {item['baseline_deceptive_proportion']:.4f}",
            f"  Finetuned deceptive proportion: {item['finetuned_deceptive_proportion']:.4f}",
            f"  p-value: {item['p_value']:.4g}",
            f"  Significant at α={SIGNIFICANCE_LEVEL}: {'YES' if item['significant'] else 'NO'}",
            "",
        ])

    out_path_txt.write_text("\n".join(lines), encoding="utf-8")


def main():
    baseline = utils.parse_baseline_counts(BASELINE_COUNTS_PATH)

    # Load counts for all models
    model_counts = {model: load_counts(path) for model, path in MODEL_FILES.items()}
    plot_experiment3_barplot(model_counts, baseline)

    # Compute finetuned vs baseline significance
    summary = []
    for model, counts in model_counts.items():
        # Only test finetuned models
        if "finetuned" not in model:
            continue

        # Recover the key used in baseline file
        model_key = model.split()[0]
        base_counts = baseline[f"{model_key} baseline"]

        stats = compute_significance(counts, base_counts)
        stats["model"] = model
        summary.append(stats)

    write_significance_summary(summary)


if __name__ == "__main__":
    main()