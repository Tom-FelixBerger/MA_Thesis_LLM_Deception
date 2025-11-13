import subprocess
import sys

PY = sys.executable

commands = [
    [PY, "-m", "Experiment2.attention_extraction"],
    [PY, "-m", "Experiment2.probe_training_and_evaluation"],
    [PY, "-m", "Experiment2.plot_layer_accuracies"],
]

for cmd in commands:
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)