import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
import joblib
import json

warnings.filterwarnings('ignore')

# Constants
NUM_LAYERS = 32
NUM_HEADS = 32
NUM_DIMENSIONS = 128
INPUT_FILE = '..\\..\\data\\vignette_activations.h5'
TARGETS = ['targets_p', 'targets_c']

class DataLoader:
    def __init__(self, filename):
        self.filename = filename
        self._train_indices = None
        self._val_indices = None
        self._test_indices = None
        self._total_samples = None

    def _load_dataset_splits(self):
        with h5py.File(self.filename, 'r') as f:
            datasets = f['datasets'][:]
            datasets_str = [ds.decode('utf-8') if isinstance(ds, bytes) else str(ds) for ds in datasets]
            self._train_indices = np.where(np.array(datasets_str) == 'probe_train')[0]
            self._val_indices = np.where(np.array(datasets_str) == 'probe_validate')[0]
            self._test_indices = np.where(np.array(datasets_str) == 'probe_test')[0]
            self._total_samples = len(datasets)

    def get_split_info(self):
        self._load_dataset_splits()
        return len(self._train_indices), len(self._val_indices), len(self._test_indices)

    def load_targets(self, target_name):
        self._load_dataset_splits()
        with h5py.File(self.filename, 'r') as f:
            targets_all = f[target_name][:]
            y_train = targets_all[self._train_indices]
            y_val = targets_all[self._val_indices]
            y_test = targets_all[self._test_indices]
        return y_train, y_val, y_test

    def load_head_train_val(self, layer_idx, head_idx):
        self._load_dataset_splits()
        start_dim = (layer_idx * NUM_HEADS + head_idx) * NUM_DIMENSIONS
        end_dim = start_dim + NUM_DIMENSIONS
        with h5py.File(self.filename, 'r') as f:
            features_all = f['activations'][:, start_dim:end_dim]
            X_train = features_all[self._train_indices]
            X_val = features_all[self._val_indices]
        return X_train, X_val

    def load_multiple_head_features(self, head_list):
        self._load_dataset_splits()
        all_dims = []
        for layer_idx, head_idx in head_list:
            start_dim = (layer_idx * NUM_HEADS + head_idx) * NUM_DIMENSIONS
            for dim in range(NUM_DIMENSIONS):
                all_dims.append(start_dim + dim)
        all_dims = sorted(all_dims)
        with h5py.File(self.filename, 'r') as f:
            features_all = f['activations'][:, all_dims]
            X_train = features_all[self._train_indices]
            X_val = features_all[self._val_indices]
            X_test = features_all[self._test_indices]
        return X_train, X_val, X_test

def train_head_classifier(X_train, y_train, X_val, y_val):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    return accuracy_score(y_val, y_pred)

def plot_accuracy_heatmap(accuracies, target_name):
    plt.figure(figsize=(12, 10))
    sns.heatmap(accuracies, annot=False, cmap='viridis',
                xticklabels=range(NUM_HEADS), yticklabels=range(NUM_LAYERS))
    plt.title(f'Attention Head Classification Accuracy - {target_name.upper()}')
    plt.xlabel('Attention Head')
    plt.ylabel('Layer')
    plt.tight_layout()
    plt.savefig(f'..\\..\\plots\\{target_name}_accuracy_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def get_top_heads(accuracies, n_heads=10, layer_range=None):
    if layer_range is not None:
        accuracies = accuracies[layer_range, :]
        offset = layer_range.start
    else:
        offset = 0
    flat_accuracies = accuracies.flatten()
    top_indices = np.argsort(flat_accuracies)[-n_heads:]
    top_heads = []
    for idx in top_indices:
        layer_idx = idx // NUM_HEADS
        head_idx = idx % NUM_HEADS
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
    data_loader = DataLoader(INPUT_FILE)
    n_train, n_val, n_test = data_loader.get_split_info()
    print(f"Training samples: {n_train}\nValidation samples: {n_val}\nTest samples: {n_test}")

    for target in TARGETS:
        output_filename = f'..\\..\\data\\{target}_analysis_results.txt'
        with open(output_filename, 'w') as outfile:
            header = f"\n{'='*50}\nAnalyzing target: {target}\n{'='*50}\n"
            print(header)
            outfile.write(header)

            y_train, y_val, y_test = data_loader.load_targets(target)

            print("Training individual attention head classifiers...")
            accuracies = np.zeros((NUM_LAYERS, NUM_HEADS))
            for layer_idx in range(NUM_LAYERS):
                print(f"Processing layer {layer_idx + 1}/{NUM_LAYERS}")
                for head_idx in range(NUM_HEADS):
                    X_train_head, X_val_head = data_loader.load_head_train_val(layer_idx, head_idx)
                    accuracy = train_head_classifier(X_train_head, y_train, X_val_head, y_val)
                    accuracies[layer_idx, head_idx] = accuracy
                    del X_train_head, X_val_head

            top_heads_full = get_top_heads(accuracies, n_heads=10)
            top_heads_half = get_top_heads(accuracies, n_heads=10, layer_range=range(0, 16))

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

            joblib.dump(clf_half, f"..\\..\\model_saves\\logreg_clf_{target}.pkl")

            # Save head and layer info for half classifier
            heads_info_file = f"..\\..\\data\\top_heads_{target}.json"
            with open(heads_info_file, 'w') as f:
                json.dump([(int(layer), int(head)) for layer, head in top_heads_half], f)
            print(f"Head/Layer info saved: {heads_info_file}")

            print(f"Results saved to: {output_filename}")
            print(f"Final classifier saved: logreg_clf_{target}.pkl")

            plot_accuracy_heatmap(accuracies, target)

if __name__ == "__main__":
    main()
