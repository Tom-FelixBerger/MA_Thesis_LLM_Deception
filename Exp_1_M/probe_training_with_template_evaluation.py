"""
This script performs the following steps:
1. Loads the vignette activations and target variables from vignette_activations.h5.
2. Splits the data into training, validation, and test sets based on the dataset column.
3. For each target variable (target_p and target_c):
    a. Trains a logistic regression classifier for each attention head in each layer on the training set.
    b. Evaluates and records the accuracy of each classifier on the validation set.
    c. Selects the top 10 most accurate (on the validation set) attention heads and prints their accuracies, differentiated by template, to the console and a text file.
    d. Trains (on the training + the validation set) a logistic regression classifier using the combined features of the top 10 attention heads.
    e. Evaluates the performance of this classifier on the test set using various metrics, prints this evaluation and additionally the accuracy differentiated by template to the console and the same text file.
    f. Generates and saves heatmap of attention head accuracies as a png.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Constants
NUM_LAYERS = 32
NUM_HEADS = 32
NUM_DIMENSIONS = 128
INPUT_FILE = 'vignette_activations_manipulated.h5'
TARGETS = ['targets_p', 'targets_c']

class MemoryEfficientDataLoader:
    """Memory-efficient data loader for HDF5 files"""
    
    def __init__(self, filename):
        self.filename = filename
        self._train_indices = None
        self._val_indices = None
        self._test_indices = None
        self._total_samples = None
        
    def _load_dataset_splits(self):
        """Load and cache train/validation/test split indices"""
            
        with h5py.File(self.filename, 'r') as f:
            datasets = f['datasets'][:]
            # Decode bytes to strings for comparison
            datasets_str = [ds.decode('utf-8') if isinstance(ds, bytes) else str(ds) for ds in datasets]
            
            self._train_indices = np.where(np.array(datasets_str) == 'train')[0]
            self._val_indices = np.where(np.array(datasets_str) == 'validate')[0]
            self._test_indices = np.where(np.array(datasets_str) == 'test')[0]
            self._total_samples = len(datasets)
    
    def get_split_info(self):
        """Get information about train/validation/test splits"""
        self._load_dataset_splits()
        return len(self._train_indices), len(self._val_indices), len(self._test_indices)
    
    def load_targets(self, target_name):
        """Load target variable for train, validation, test sets"""
        self._load_dataset_splits()
        
        with h5py.File(self.filename, 'r') as f:
            targets_all = f[target_name][:]
            y_train = targets_all[self._train_indices]
            y_val = targets_all[self._val_indices]
            y_test = targets_all[self._test_indices]
        
        return y_train, y_val, y_test
    
    def load_template_ids(self):
        """Load template IDs for train, validation, test sets"""
        self._load_dataset_splits()
        
        with h5py.File(self.filename, 'r') as f:
            template_ids_all = f['template_ids'][:]
            template_ids_train = template_ids_all[self._train_indices]
            template_ids_val = template_ids_all[self._val_indices]
            template_ids_test = template_ids_all[self._test_indices]
        
        return template_ids_train, template_ids_val, template_ids_test
    
    def load_head_train_val(self, layer_idx, head_idx):
        """Load features for a specific attention head"""
        self._load_dataset_splits()
        
        start_dim = (layer_idx * NUM_HEADS + head_idx) * NUM_DIMENSIONS
        end_dim = start_dim + NUM_DIMENSIONS
        
        with h5py.File(self.filename, 'r') as f:
            # Load only the required columns for this head
            features_all = f['activations'][:, start_dim:end_dim]
            X_train = features_all[self._train_indices]
            X_val = features_all[self._val_indices]
        
        return X_train, X_val
    
    def load_multiple_head_features(self, head_list):
        """Load features for multiple attention heads efficiently"""
        self._load_dataset_splits()
        
        # Calculate which dimensions we need
        all_dims = []
        for layer_idx, head_idx in head_list:
            start_dim = (layer_idx * NUM_HEADS + head_idx) * NUM_DIMENSIONS
            for dim in range(NUM_DIMENSIONS):
                all_dims.append(start_dim + dim)
        
        all_dims = sorted(all_dims)
        
        with h5py.File(self.filename, 'r') as f:
            # Load only the required dimensions
            features_all = f['activations'][:, all_dims]
            X_train = features_all[self._train_indices]
            X_val = features_all[self._val_indices]
            X_test = features_all[self._test_indices]
        
        return X_train, X_val, X_test

def train_head_classifier(X_train, y_train, X_val, y_val, template_ids_val=None):
    """Train logistic regression classifier for a single attention head"""
    
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_val)
    overall_accuracy = accuracy_score(y_val, y_pred)
    
    # Calculate per-template accuracies if template IDs provided
    template_accuracies = {}
    if template_ids_val is not None:
        unique_templates = np.unique(template_ids_val)
        for template_id in unique_templates:
            template_mask = template_ids_val == template_id
            if np.sum(template_mask) > 0:
                template_acc = accuracy_score(y_val[template_mask], y_pred[template_mask])
                template_accuracies[int(template_id)] = template_acc
    
    return overall_accuracy, template_accuracies, clf

def plot_accuracy_heatmap(accuracies, target_name):
    """Plot accuracy heatmap for all attention heads"""
    plt.figure(figsize=(12, 10))
    sns.heatmap(accuracies, annot=False, cmap='viridis', 
                xticklabels=range(NUM_HEADS), yticklabels=range(NUM_LAYERS))
    plt.title(f'Attention Head Classification Accuracy - {target_name.upper()}')
    plt.xlabel('Attention Head')
    plt.ylabel('Layer')
    plt.tight_layout()
    plt.savefig(f'{target_name}_accuracy_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def get_top_heads(accuracies, n_heads=10):
    """Get indices of top n most accurate attention heads"""
    # Flatten the accuracy matrix and get top indices
    flat_accuracies = accuracies.flatten()
    top_indices = np.argsort(flat_accuracies)[-n_heads:][::-1]  # Reverse to get descending order
    
    # Convert flat indices back to (layer, head) pairs
    top_heads = []
    for idx in top_indices:
        layer_idx = idx // NUM_HEADS
        head_idx = idx % NUM_HEADS
        top_heads.append((layer_idx, head_idx))
    
    return top_heads

def evaluate_classifier(clf, X_test, y_test, template_ids_test=None):
    """Evaluate classifier and return metrics"""
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    # Calculate per-template accuracies if template IDs provided
    template_accuracies = {}
    if template_ids_test is not None:
        unique_templates = np.unique(template_ids_test)
        for template_id in unique_templates:
            template_mask = template_ids_test == template_id
            if np.sum(template_mask) > 0:
                template_acc = accuracy_score(y_test[template_mask], y_pred[template_mask])
                template_accuracies[int(template_id)] = template_acc
    
    return metrics, template_accuracies

def main():
    # Initialize data loader
    print("Initializing data loader...")
    data_loader = MemoryEfficientDataLoader(INPUT_FILE)
    
    # Get dataset information
    n_train, n_val, n_test = data_loader.get_split_info()
    print(f"Training samples: {n_train}")
    print(f"Validation samples: {n_val}")
    print(f"Test samples: {n_test}")

    # Load template IDs
    template_ids_train, template_ids_val, template_ids_test = data_loader.load_template_ids()

    # Main analysis loop
    for target in TARGETS:
        output_filename = f'{target}_analysis_results.txt'
        
        # Open file to save results for this target
        with open(output_filename, 'w') as outfile:
            
            # Print to console and file
            header = f"\n{'='*50}\nAnalyzing target: {target}\n{'='*50}\n"
            print(header)
            outfile.write(header)
            
            # Load target variables
            y_train, y_val, y_test = data_loader.load_targets(target)
            
            # Step 3.a. and 3.b.: Train individual attention head classifiers and evaluate accuracy on validation set
            print("Training individual attention head classifiers...")
            accuracies = np.zeros((NUM_LAYERS, NUM_HEADS))
            all_template_accuracies = {}  # Store template accuracies for all heads
            
            for layer_idx in range(NUM_LAYERS):
                print(f"Processing layer {layer_idx + 1}/{NUM_LAYERS}")
                for head_idx in range(NUM_HEADS):
                    # Load features for this head only
                    X_train_head, X_val_head = data_loader.load_head_train_val(layer_idx, head_idx)
                    
                    # Train and evaluate classifier with template breakdown
                    accuracy, template_accs, _ = train_head_classifier(
                        X_train_head, y_train, X_val_head, y_val, template_ids_val
                    )
                    accuracies[layer_idx, head_idx] = accuracy
                    all_template_accuracies[(layer_idx, head_idx)] = template_accs
                    
                    # Clear memory
                    del X_train_head, X_val_head
            
            # Step 3.c.: Get top 10 attention heads
            top_heads = get_top_heads(accuracies, n_heads=10)
            
            top_heads_info = "\n--- Top 10 Attention Heads (Layer, Head) with Template-Specific Accuracies ---\n"
            print(top_heads_info)
            outfile.write(top_heads_info)
            
            # Get all unique templates
            unique_templates = sorted(np.unique(template_ids_val))
            
            for rank, (layer, head) in enumerate(top_heads, 1):
                acc = accuracies[layer, head]
                head_line = f"\nRank {rank}: Layer {layer:02d}, Head {head:02d}: Overall Accuracy = {acc:.4f}\n"
                print(head_line.strip())
                outfile.write(head_line)
                
                # Print template-specific accuracies for this head
                template_accs = all_template_accuracies[(layer, head)]
                for template_id in unique_templates:
                    template_id = int(template_id)
                    if template_id in template_accs:
                        template_count = np.sum(template_ids_val == template_id)
                        template_line = f"  Template {template_id:2d}: Accuracy = {template_accs[template_id]:.4f} (n={template_count})\n"
                        print(template_line.strip())
                        outfile.write(template_line)
                
            print("\nTraining classifier on combined top 10 attention heads features...")
            X_train, X_val, X_test = data_loader.load_multiple_head_features(top_heads)
            X_train_val = np.concatenate((X_train, X_val), axis=0)
            y_train_val = np.concatenate((y_train, y_val), axis=0)
            
            clf = LogisticRegression(max_iter=1000)
            clf.fit(X_train_val, y_train_val)
            
            # Step 5: Evaluate classifiers
            print("\nEvaluating classifier on Test Set...")
            metrics, test_template_accs = evaluate_classifier(clf, X_test, y_test, template_ids_test)
            
            evaluation_header = "\n--- Test Set Evaluation Metrics ---\n"
            print(evaluation_header.strip())
            outfile.write(evaluation_header)
            
            for metric_name in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
                eval_value = metrics[metric_name]
                metric_line = f"{metric_name:<15} {eval_value:<12.4f}\n"
                print(metric_line.strip())
                outfile.write(metric_line)
            
            # Write template-specific accuracies for 10-head classifier
            template_test_header = "\n--- 10-Head Classifier Accuracy by Template (Test Set) ---\n"
            print(template_test_header)
            outfile.write(template_test_header)
            
            sorted_test_templates = sorted(test_template_accs.items(), key=lambda x: x[1], reverse=True)
            for template_id, test_acc in sorted_test_templates:
                template_count = np.sum(template_ids_test == template_id)
                test_template_line = f"Template {template_id:2d}: Accuracy = {test_acc:.4f} (n={template_count})\n"
                print(test_template_line.strip())
                outfile.write(test_template_line)
        
        # Step 3.f.: Plot accuracy heatmap
        print("\nCreating accuracy heatmap...")
        plot_accuracy_heatmap(accuracies, target)
        
        print(f"Results saved to: {output_filename}")
        
        # Clean up memory
        del X_train, X_val, X_test
        del y_train, y_val, y_test, clf

if __name__ == "__main__": 
    main()