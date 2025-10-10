"""Execute all experiment pipelines sequentially with absolute script paths."""
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

SCRIPTS = [
    BASE_DIR / "vignette_generation.py",
    BASE_DIR / "Experiment1" / "baseline_assessment.py",
    BASE_DIR / "Experiment1" / "classification_visualization.py",
    BASE_DIR / "Experiment2" / "activation_extraction.py",
    BASE_DIR / "Experiment2" / "probe_training.py",
    BASE_DIR / "Experiment3" / "reinforce_finetuning_lora.py",
    BASE_DIR / "Experiment4" / "pretraining.py",
    BASE_DIR / "Experiment4" / "SOO_vs_TBI.py",
    BASE_DIR / "finetuned_assessment.py",
]


def main() -> None:
    for script in SCRIPTS:
        print(f"\n=== Running {script} ===")
        subprocess.run([sys.executable, str(script)], check=True)


if __name__ == "__main__":
    main()
