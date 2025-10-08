import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
import joblib
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import utils

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / 'data'
PLOTS_DIR = BASE_DIR / 'plots'
MODEL_SAVES_DIR = BASE_DIR / 'model_saves'
INPUT_PATH = DATA_DIR / 'vignette_activations.h5'
TARGETS = ['targets_p', 'targets_c']


class DataLoader:
    def __init__(self, filename):
        self.filename = Path(filename)
        self.metadata = utils.load_activation_metadata(str(self.filename))
        self._train_indices = None
        self._val_indices = None
        self._test_indices = None
        self._activations_flat = None

    def _ensure_indices_loaded(self):
        if self._train_indices is not None:
            return
        with h5py.File(str(self.filename), 'r') as f:
            datasets = f['datasets'][:]
            datasets_str = [ds.decode('utf-8') if isinstance(ds, bytes) else str(ds) for ds in datasets]
            datasets_arr = np.array(datasets_str)
            self._train_indices = np.where(datasets_arr == 'probe_train')[0]
            self._val_indices = np.where(datasets_arr == 'probe_validate')[0]
            self._test_indices = np.where(datasets_arr == 'probe_test')[0]

    def _ensure_activations_loaded(self):
        if self._activations_flat is None:
            self._activations_flat = utils.load_activation_batch(str(self.filename), return_flat=True)

    def get_split_info(self):
        self._ensure_indices_loaded()
        return len(self._train_indices), len(self._val_indices), len(self._test_indices)

    def load_targets(self, target_name):
        self._ensure_indices_loaded()
        with h5py.File(str(self.filename), 'r') as f:
            targets_all = f[target_name][:]
        return (
            targets_all[self._train_indices],
            targets_all[self._val_indices],
            targets_all[self._test_indices]
        )

    def load_head_train_val(self, layer_idx, head_idx):
        self._ensure_indices_loaded()
        self._ensure_activations_loaded()
        features_all = utils.select_activation_subset(
            self._activations_flat,
            [(layer_idx, head_idx)],
            self.metadata
        )
        return (
            features_all[self._train_indices],
            features_all[self._val_indices]
        )

    def load_multiple_head_features(self, head_list):
        self._ensure_indices_loaded()
        self._ensure_activations_loaded()
        features_all = utils.select_activation_subset(
            self._activations_flat,
            head_list,
            self.metadata
        )
        return (
            features_all[self._train_indices],
            features_all[self._val_indices],
            features_all[self._test_indices]
        )

def train_head_classifier(X_train, y_train, X_val, y_val):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    return accuracy_score(y_val, y_pred)

def plot_accuracy_heatmap(accuracies, target_name, num_layers, num_heads):
    plt.figure(figsize=(12, 10))
    sns.heatmap(accuracies, annot=False, cmap='viridis',
                xticklabels=range(num_heads), yticklabels=range(num_layers))
    plt.title(f'Attention Head Classification Accuracy - {target_name.upper()}')
    plt.xlabel('Attention Head')
    plt.ylabel('Layer')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{target_name}_accuracy_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

def get_top_heads(accuracies, num_heads, n_heads=10, layer_range=None):
    if layer_range is not None:
        accuracies = accuracies[layer_range, :]
        offset = layer_range.start
    else:
        offset = 0
    flat_accuracies = accuracies.flatten()
    top_indices = np.argsort(flat_accuracies)[-n_heads:]
    top_heads = []
    for idx in top_indices:
        layer_idx = idx // num_heads
        head_idx = idx % num_heads
        top_heads.append((layer_idx + offset, head_idx))
    return top_heads

def evaluate_classifier(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    return metrics

def main():
    print("Initializing data loader...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_SAVES_DIR.mkdir(parents=True, exist_ok=True)
    data_loader = DataLoader(INPUT_PATH)
    num_layers = data_loader.metadata['num_layers']
    num_heads = data_loader.metadata['num_heads']
    n_train, n_val, n_test = data_loader.get_split_info()
    print(f"Training samples: {n_train}\nValidation samples: {n_val}\nTest samples: {n_test}")

    for target in TARGETS:
        output_path = DATA_DIR / f"{target}_analysis_results.txt"
        with output_path.open('w') as outfile:
            header = f"\n{'='*50}\nAnalyzing target: {target}\n{'='*50}\n"
            print(header)
            outfile.write(header)

            y_train, y_val, y_test = data_loader.load_targets(target)

            print("Training individual attention head classifiers...")
            accuracies = np.zeros((num_layers, num_heads))
            for layer_idx in range(num_layers):
                print(f"Processing layer {layer_idx + 1}/{num_layers}")
                for head_idx in range(num_heads):
                    X_train_head, X_val_head = data_loader.load_head_train_val(layer_idx, head_idx)
                    accuracy = train_head_classifier(X_train_head, y_train, X_val_head, y_val)
                    accuracies[layer_idx, head_idx] = accuracy
                    del X_train_head, X_val_head

            top_heads_full = get_top_heads(accuracies, num_heads=num_heads, n_heads=10)
            top_heads_half = get_top_heads(
                accuracies,
                num_heads=num_heads,
                n_heads=10,
                layer_range=range(0, max(num_layers // 2, 1))
            )

            outfile.write("\n--- Top 10 Attention Heads (Full range) ---\n")
            for layer, head in top_heads_full:
                acc = accuracies[layer, head]
                outfile.write(f"Layer {layer:02d}, Head {head:02d}: Accuracy = {acc:.4f}\n")

            outfile.write("\n--- Top 10 Attention Heads (First half layers) ---\n")
            for layer, head in top_heads_half:
                acc = accuracies[layer, head]
                outfile.write(f"Layer {layer:02d}, Head {head:02d}: Accuracy = {acc:.4f}\n")

            X_train_f, X_val_f, X_test_f = data_loader.load_multiple_head_features(top_heads_full)
            X_train_val_f = np.concatenate((X_train_f, X_val_f), axis=0)
            y_train_val_f = np.concatenate((y_train, y_val), axis=0)
            clf_full = LogisticRegression(max_iter=1000)
            clf_full.fit(X_train_val_f, y_train_val_f)
            metrics_full = evaluate_classifier(clf_full, X_test_f, y_test)

            outfile.write("\n--- Test Set Evaluation (Full range top 10) ---\n")
            for k, v in metrics_full.items():
                outfile.write(f"{k:<15} {v:.4f}\n")

            X_train_h, X_val_h, X_test_h = data_loader.load_multiple_head_features(top_heads_half)
            X_train_val_h = np.concatenate((X_train_h, X_val_h), axis=0)
            y_train_val_h = np.concatenate((y_train, y_val), axis=0)
            clf_half = LogisticRegression(max_iter=1000)
            clf_half.fit(X_train_val_h, y_train_val_h)
            metrics_half = evaluate_classifier(clf_half, X_test_h, y_test)

            outfile.write("\n--- Test Set Evaluation (First half top 10) ---\n")
            for k, v in metrics_half.items():
                outfile.write(f"{k:<15} {v:.4f}\n")

            joblib.dump(clf_half, MODEL_SAVES_DIR / f"logreg_clf_{target}.pkl")

            # Save head and layer info for half classifier
            heads_info_path = DATA_DIR / f"top_heads_{target}.json"
            with heads_info_path.open('w') as f:
                json.dump([(int(layer), int(head)) for layer, head in top_heads_half], f)
            print(f"Head/Layer info saved: {heads_info_path}")

            print(f"Results saved to: {output_path}")
            print(f"Final classifier saved: {MODEL_SAVES_DIR / f'logreg_clf_{target}.pkl'}")

            plot_accuracy_heatmap(accuracies, target, num_layers, num_heads)

if __name__ == "__main__":
    main()
