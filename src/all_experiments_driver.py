import subprocess
import sys

PY = sys.executable

commands = [
    
    [PY, "-m", "Experiment1.forced_choice_task"],
    [PY, "-m", "Experiment1.classification_results"],
    
    [PY, "-m", "Experiment2.attention_extraction"],
    [PY, "-m", "Experiment2.probe_training_and_evaluation"],
    [PY, "-m", "Experiment2.plot_heatmaps"],
    [PY, "-m", "Experiment2.pkl_to_csv"],

    [PY, "-m", "Experiment3.tbi_and_free_finetuning"],
    [PY, "-m", "Experiment3.generate_finetuned_responses"],
    [PY, "-m", "Experiment3.plots_and_significance"],
    [PY, "-m", "Experiment3.example_free_responses"],

    [PY, "-m", "Experiment4.superdeceiver"],
    [PY, "-m", "Experiment4.soo_finetuning"],
    [PY, "-m", "Experiment4.tbi_on_superdeceiver"],
    [PY, "-m", "Experiment4.plots_and_significance"],
]

for cmd in commands:
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)