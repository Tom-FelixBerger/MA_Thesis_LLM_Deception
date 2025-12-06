import json
import ast
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import binomtest

from utils import utils

OUTPUT_DIR = utils.DATA_DIR / "Experiment4"
BASELINE_COUNTS_PATH = utils.DATA_DIR / "Experiment1" / "baseline_deceptive_counts.txt"


# ---------------------------------------------------------
# Load JSONL classification counts
# ---------------------------------------------------------

def load_counts(jsonl_path):
    """Load deceptive classification counts from a saved jsonl responses file."""
    counts = {"honest": 0, "deceptive": 0, "invalid": 0}

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            counts[entry["classification"]] += 1

    return counts


# ---------------------------------------------------------
# Significance tests
# ---------------------------------------------------------

def test_a(baseline_counts, superdec_counts):
    base_prop, base_n = utils.deceptive_proportion(baseline_counts)
    pre_prop, pre_n = utils.deceptive_proportion(superdec_counts)

    p = binomtest(
        k=superdec_counts["deceptive"],
        n=pre_n,
        p=base_prop,
        alternative="greater"
    ).pvalue

    return {
        "test": "a_superdec_vs_baseline",
        "baseline_deceptive_proportion": base_prop,
        "superdec_deceptive_proportion": pre_prop,
        "p_value": p,
    }


def test_b(baseline_counts, standard_soo_counts):
    base_prop, base_n = utils.deceptive_proportion(baseline_counts)
    soo_prop, soo_n = utils.deceptive_proportion(standard_soo_counts)

    p = binomtest(
        k=standard_soo_counts["deceptive"],
        n=soo_n,
        p=base_prop,
        alternative="less"
    ).pvalue

    return {
        "test": "b_standard_soo_vs_baseline",
        "baseline_deceptive_proportion": base_prop,
        "standard_soo_deceptive_proportion": soo_prop,
        "p_value": p,
    }


def test_c(superdec_soo_counts, superdec_tbi_counts):
    soo_prop, soo_n = utils.deceptive_proportion(superdec_soo_counts)
    tbi_prop, tbi_n = utils.deceptive_proportion(superdec_tbi_counts)

    total_n = soo_n + tbi_n
    if total_n == 0:
        p_value = 1.0
    else:
        pooled = (
            superdec_soo_counts["deceptive"] +
            superdec_tbi_counts["deceptive"]
        ) / total_n

        p_value = binomtest(
            k=superdec_soo_counts["deceptive"],
            n=soo_n,
            p=pooled,
            alternative="two-sided"
        ).pvalue

    return {
        "test": "c_superdec_soo_vs_superdec_tbi",
        "superdec_soo_deceptive_proportion": soo_prop,
        "superdec_tbi_deceptive_proportion": tbi_prop,
        "p_value": p_value,
    }


# ---------------------------------------------------------
# Plotting (matching Experiment 3 style)
# ---------------------------------------------------------

def plot_experiment4_barplot(model_counts, baseline):
    all_counts = model_counts | {"standard": baseline}
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
    plt.ylabel("Finetuning Variant")
    plt.title("Response Classifications for Gemma-2-9b")
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

    outpath = utils.PLOTS_DIR / "experiment4_deceptive_responses.png"
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"Saved plot to: {outpath}")


# ---------------------------------------------------------
# Summary writers
# ---------------------------------------------------------

def write_summary(all_results):
    """Write JSON and TXT summaries."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    (OUTPUT_DIR / "experiment4_significance_summary.json").write_text(
        json.dumps(all_results, indent=2), encoding="utf-8"
    )

    lines = ["Experiment 4 – Significance Test Summary", ""]
    for r in all_results:
        lines.append(f"Test: {r['test']} | Model: {r['model_key']}")
        for k, v in r.items():
            if k not in ["test", "model_key"]:
                lines.append(f"  {k}: {v}")
        lines.append("")

    (OUTPUT_DIR / "experiment4_significance_summary.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )


# ---------------------------------------------------------
# Main analysis logic
# ---------------------------------------------------------

def main():
    baseline_all = utils.parse_baseline_counts(BASELINE_COUNTS_PATH)

    results = []

    print(f"Analyzing Experiment 4 significance for gemma-2-9b")
    baseline = baseline_all["gemma-2-9b baseline"]

    # Load variant counts
    superdeceiver = load_counts(OUTPUT_DIR / "gemma-2-9b_superdec_baseline.jsonl")
    standard_soo = load_counts(OUTPUT_DIR / "gemma-2-9b_standard_soo_baseline.jsonl")
    superdeceiver_tbi = load_counts(OUTPUT_DIR / "gemma-2-9b_superdec_tbi_baseline.jsonl")
    superdeceiver_soo = load_counts(OUTPUT_DIR / "gemma-2-9b_superdec_soo_baseline.jsonl")

    # Store for plotting
    model_counts = {
        "superdeceiver": superdeceiver,
        "standard_soo": standard_soo,
        "superdeceiver_tbi": superdeceiver_tbi,
        "superdeceiver_soo": superdeceiver_soo,
    }

    # Run statistical tests
    results.append({**test_a(baseline, superdeceiver), "model_key": "gemma-2-9b"})
    results.append({**test_b(baseline, standard_soo), "model_key": "gemma-2-9b"})
    results.append({**test_c(superdeceiver_soo, superdeceiver_tbi), "model_key": "gemma-2-9b"})

    # Write significance summary
    write_summary(results)

    # Create barplot
    plot_experiment4_barplot(model_counts, baseline)


if __name__ == "__main__":
    main()