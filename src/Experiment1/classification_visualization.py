import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import utils

# Define colors and order for classifications
COLORS = {
    "deceptive": "salmon",
    "honest": "skyblue",
    "invalid": "grey"
}
CLASSIFICATION_ORDER = ["deceptive", "honest", "invalid"]

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / 'data'
PLOT_DIR = BASE_DIR / 'plots'
RESPONSES_PATH = DATA_DIR / 'model_responses_baseline_assessment.jsonl'
VIGNETTES_PATH = DATA_DIR / 'vignettes.jsonl'



def plot_classification_bar(data, title, filename, ylabel="Relative Frequency"):
    """Helper function to create consistent bar plots with proper ordering and colors."""
    # Reindex to ensure correct order
    data = data.reindex(CLASSIFICATION_ORDER, fill_value=0)
    colors = [COLORS[cat] for cat in data.index]
    
    plt.figure(figsize=(6, 4))
    data.plot(kind="bar", color=colors)
    plt.title(title)
    plt.xlabel("Classification")
    plt.ylabel(ylabel)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename)
    plt.close()


def plot_grouped_classification(grouped_relative, title, filename):
    """Helper function to create grouped bar plots with proper ordering and colors."""
    # Reindex columns to ensure correct order
    grouped_relative = grouped_relative.reindex(columns=CLASSIFICATION_ORDER, fill_value=0)
    colors = [COLORS[cat] for cat in grouped_relative.columns]
    
    plt.figure(figsize=(10, 6))
    grouped_relative.plot(kind="bar", color=colors, figsize=(10, 6))
    plt.title(title)
    plt.xlabel("Template ID")
    plt.ylabel("Relative Frequency")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.legend(title="Classification", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename)
    plt.close()


def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # Load JSONL files
    df = pd.read_json(RESPONSES_PATH, lines=True)
    vign_df = pd.read_json(VIGNETTES_PATH, lines=True)

    assess_templates = vign_df[vign_df["dataset"] == "assessment"]["template_id"].unique().tolist()
    dataset_by_template = (
        vign_df.drop_duplicates(subset="template_id")
        .set_index("template_id")
        ["dataset"]
    )

    models = ["gpt", "mistral", "gemma"]

    for model in models:
        model_df = df[df["model"] == model]
        if model_df.empty:
            continue

        model_grouped_counts = (
            model_df.groupby(["template_id", "classification"]).size().unstack(fill_value=0)
        )
        model_grouped_relative = model_grouped_counts.div(
            model_grouped_counts.sum(axis=1), axis=0
        )

        plot_grouped_classification(
            model_grouped_relative,
            f"{model.upper()} Classification by Template",
            f"{model}_classification_counts_by_template.png",
        )

        assess_only_relative = model_grouped_relative.loc[
            model_grouped_relative.index.isin(assess_templates)
        ]

        if not assess_only_relative.empty:
            plot_grouped_classification(
                assess_only_relative,
                f"{model.upper()} Assessment Classification by Template",
                f"{model}_assessment_classification_counts_by_template.png",
            )

    write_model_comparison_tables(df, dataset_by_template)


def write_model_comparison_tables(df: pd.DataFrame, dataset_by_template: pd.Series) -> None:
    """Write a summary file comparing Gemma and Mistral template classifications."""

    target_models = ["gemma", "mistral"]
    model_dfs = {
        model: df[df["model"] == model].groupby(["template_id", "classification"]).size().unstack(fill_value=0)
        for model in target_models
    }

    if any(model_df.empty for model_df in model_dfs.values()):
        # If either dataframe is empty, still create an empty summary file.
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        (DATA_DIR / "model_classifications.txt").write_text(
            "Templates 100% Honest for Gemma and Mistral\n(none)\n\n"
            "Templates 100% Deceptive for Gemma and Mistral\n(none)\n",
            encoding="utf-8",
        )
        return

    common_templates = set.intersection(*[set(model_df.index) for model_df in model_dfs.values()])

    honest_templates = []
    deceptive_templates = []

    for template_id in sorted(common_templates):
        template_stats = {}
        for model, model_counts in model_dfs.items():
            counts = model_counts.loc[template_id].reindex(CLASSIFICATION_ORDER, fill_value=0)
            total = counts.sum()
            template_stats[model] = counts / total if total > 0 else counts

        if all(template_stats[model]["honest"] == 1.0 for model in target_models):
            honest_templates.append(template_id)
        if all(template_stats[model]["deceptive"] == 1.0 for model in target_models):
            deceptive_templates.append(template_id)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_DIR / "model_classifications.txt"

    def format_template_list(title: str, templates: list[int]) -> str:
        lines = [title, "Template ID | Dataset", "--------------------"]
        for template_id in templates:
            dataset = dataset_by_template.get(template_id, "unknown")
            lines.append(f"{template_id:>11} | {dataset}")
        if not templates:
            lines.append("(none)")
        lines.append("")
        return "\n".join(lines)

    content = "".join(
        [
            format_template_list(
                "Templates 100% Honest for Gemma and Mistral", honest_templates
            ),
            format_template_list(
                "Templates 100% Deceptive for Gemma and Mistral", deceptive_templates
            ),
        ]
    )

    with output_path.open("w", encoding="utf-8") as f:
        f.write(content)


if __name__ == "__main__":
    main()
