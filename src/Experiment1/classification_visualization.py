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

    # Calculate overall classification frequencies
    overall_counts = df["classification"].value_counts()
    overall_relative = overall_counts / overall_counts.sum()

    # Calculate assessment classification frequencies
    assess_templates = vign_df[vign_df['dataset'] == 'assessment']['template_id'].unique().tolist()
    assess_counts = df[df['template_id'].isin(assess_templates)]['classification'].value_counts()
    assess_relative = assess_counts / assess_counts.sum()

    # Calculate classification frequencies grouped by template_id
    grouped_counts = df.groupby(["template_id", "classification"]).size().unstack(fill_value=0)
    grouped_relative = grouped_counts.div(grouped_counts.sum(axis=1), axis=0)

    # Create plots
    plot_classification_bar(
        overall_relative,
        "Overall Classification Distribution",
        "overall_classification_counts.png"
    )

    plot_classification_bar(
        assess_relative,
        "Assessment Classification Distribution",
        "assessment_classification_counts.png"
    )

    plot_grouped_classification(
        grouped_relative,
        "Classification Distribution by Template ID",
        "classification_counts_by_template.png"
    )


if __name__ == "__main__":
    main()
