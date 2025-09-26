"""
This script performs the following steps:
1. Loads the vignette activations and target variables from vignette_activations.csv.
2. Splits the data into training and validation sets based on the dataset column.
3. For each target variable (target_p and target_c):
    a. Trains a logistic regression classifier for each attention head in each layer.
    b. Evaluates and records the accuracy of each classifier on the validation set.
    c. Identifies the top 10 most accurate attention heads.
    d. Trains two logistic regression classifiers using the top 10 heads:
        - One using heads from the first half of layers (0-15).
        - One using heads from all layers (0-31).
    e. Compares the performance of these two classifiers on the validation set using various metrics.
    f. Generates and saves visualizations:
        - Heatmap of attention head accuracies.
        - ROC curves for both classifiers.
        - PCA visualization with decision boundary for the first half classifier.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Load the data
print("Loading data...")
df = pd.read_csv('vignette_activations.csv')
print(f"Data shape: {df.shape}")

# Split data by dataset
train_data = df[df['dataset'] == 'train'].copy()
val_data = df[df['dataset'] == 'validate'].copy()

print(f"Training samples: {len(train_data)}")
print(f"Validation samples: {len(val_data)}")

# Constants
NUM_LAYERS = 32
NUM_HEADS = 32
NUM_DIMENSIONS = 128

# Target variables
targets = ['target_p', 'target_c']

def get_attention_columns(layer_idx, head_idx):
    """Get column names for a specific attention head"""
    return [f"layer_{layer_idx}_head_{head_idx}_dim_{dim}_attention" 
            for dim in range(NUM_DIMENSIONS)]

def train_head_classifier(X_train, y_train, X_val, y_val):
    """Train logistic regression classifier for a single attention head"""
    # Handle case where all targets are the same
    if len(np.unique(y_train)) < 2:
        return 0.5  # Return chance accuracy
    
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    
    if len(np.unique(y_val)) < 2:
        return 0.5
    
    y_pred = clf.predict(X_val)
    return accuracy_score(y_val, y_pred)

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
    top_indices = np.argsort(flat_accuracies)[-n_heads:]
    
    # Convert flat indices back to (layer, head) pairs
    top_heads = []
    for idx in top_indices:
        layer_idx = idx // NUM_HEADS
        head_idx = idx % NUM_HEADS
        top_heads.append((layer_idx, head_idx))
    
    return top_heads

def extract_features_for_heads(data, head_list):
    """Extract features for specified attention heads"""
    all_features = []
    for layer_idx, head_idx in head_list:
        attention_cols = get_attention_columns(layer_idx, head_idx)
        head_features = data[attention_cols].values
        all_features.append(head_features)
    
    return np.concatenate(all_features, axis=1)

def evaluate_classifier(clf, X_val, y_val):
    """Evaluate classifier and return metrics"""
    y_pred = clf.predict(X_val)
    y_pred_proba = clf.predict_proba(X_val)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred, zero_division=0),
        'recall': recall_score(y_val, y_pred, zero_division=0),
        'f1': f1_score(y_val, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_val, y_pred_proba) if len(np.unique(y_val)) > 1 else 0.5
    }
    
    return metrics, y_pred_proba

def plot_roc_curves(y_val, y_pred_proba_first, y_pred_proba_full, target_name):
    """Plot ROC curves for both classifiers"""
    plt.figure(figsize=(10, 8))
    
    # First half classifier ROC
    fpr_first, tpr_first, _ = roc_curve(y_val, y_pred_proba_first)
    roc_auc_first = roc_auc_score(y_val, y_pred_proba_first)
    
    # Full range classifier ROC
    fpr_full, tpr_full, _ = roc_curve(y_val, y_pred_proba_full)
    roc_auc_full = roc_auc_score(y_val, y_pred_proba_full)
    
    plt.plot(fpr_first, tpr_first, label=f'First Half Layers (AUC = {roc_auc_first:.3f})', 
             linewidth=2, color='blue')
    plt.plot(fpr_full, tpr_full, label=f'Full Range Layers (AUC = {roc_auc_full:.3f})', 
             linewidth=2, color='red')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves - {target_name.upper()}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{target_name}_roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_pca_visualization(X, y, clf, target_name):
    """Plot PCA visualization with decision boundary"""
    # Standardize features before PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Train classifier on PCA-transformed data for decision boundary
    clf_pca = LogisticRegression()
    clf_pca.fit(X_pca, y)
    
    # Create a mesh for decision boundary
    h = 0.02
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Plot
    plt.figure(figsize=(12, 10))
    
    # Decision boundary
    Z = clf_pca.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, levels=50, alpha=0.6, cmap='RdYlBu')
    plt.colorbar(label='Prediction Probability')
    
    # Data points
    colors = ['red', 'blue']
    labels = ['False', 'True']
    for i, (color, label) in enumerate(zip(colors, labels)):
        mask = (y == i) if i == 0 else (y == 1)
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   c=color, alpha=0.7, s=30, label=f'{target_name.upper()} = {label}')
    
    plt.xlabel(f'First Principal Component (explained variance: {pca.explained_variance_ratio_[0]:.3f})')
    plt.ylabel(f'Second Principal Component (explained variance: {pca.explained_variance_ratio_[1]:.3f})')
    plt.title(f'PCA Visualization with Decision Boundary - {target_name.upper()}\n'
              f'Total explained variance: {sum(pca.explained_variance_ratio_):.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{target_name}_pca_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()

# Main analysis loop
for target in targets:
    print(f"\n{'='*50}")
    print(f"Analyzing target: {target}")
    print(f"{'='*50}")
    
    # Extract target variables
    y_train = train_data[target].values
    y_val = val_data[target].values
    
    print(f"Training target distribution: {np.bincount(y_train)}")
    print(f"Validation target distribution: {np.bincount(y_val)}")
    
    # Step 1: Train individual attention head classifiers
    print("Training individual attention head classifiers...")
    accuracies = np.zeros((NUM_LAYERS, NUM_HEADS))
    
    for layer_idx in range(NUM_LAYERS):
        print(f"Processing layer {layer_idx + 1}/{NUM_LAYERS}")
        for head_idx in range(NUM_HEADS):
            # Get attention columns for this head
            attention_cols = get_attention_columns(layer_idx, head_idx)
            
            # Extract features
            X_train_head = train_data[attention_cols].values
            X_val_head = val_data[attention_cols].values
            
            # Train and evaluate classifier
            accuracy = train_head_classifier(X_train_head, y_train, X_val_head, y_val)
            accuracies[layer_idx, head_idx] = accuracy
    
    # Step 2: Plot accuracy heatmap
    print("Creating accuracy heatmap...")
    plot_accuracy_heatmap(accuracies, target)
    
    # Step 3: Get top 10 attention heads
    top_heads = get_top_heads(accuracies, n_heads=10)
    print(f"Top 10 attention heads: {top_heads}")
    print(f"Their accuracies: {[accuracies[layer, head] for layer, head in top_heads]}")
    
    # Step 4: Train classifiers on top heads
    print("Training classifiers on top 10 attention heads...")
    
    # First half of layers (0-15)
    first_half_heads = [(layer, head) for layer, head in top_heads if layer < 16]
    if len(first_half_heads) < 10:
        # If we don't have 10 heads in first half, take top heads from first half only
        first_half_accuracies = accuracies[:16, :].copy()
        first_half_heads = get_top_heads(first_half_accuracies, n_heads=min(10, len(first_half_heads)))
    
    print(f"First half heads: {first_half_heads}")
    
    # Extract features for first half classifier
    X_train_first = extract_features_for_heads(train_data, first_half_heads)
    X_val_first = extract_features_for_heads(val_data, first_half_heads)
    
    # Extract features for full range classifier  
    X_train_full = extract_features_for_heads(train_data, top_heads)
    X_val_full = extract_features_for_heads(val_data, top_heads)
    
    # Train classifiers
    clf_first = LogisticRegression()
    clf_full = LogisticRegression()
    
    clf_first.fit(X_train_first, y_train)
    clf_full.fit(X_train_full, y_train)
    
    # Step 5: Evaluate classifiers
    print("Evaluating classifiers...")
    
    metrics_first, y_pred_proba_first = evaluate_classifier(clf_first, X_val_first, y_val)
    metrics_full, y_pred_proba_full = evaluate_classifier(clf_full, X_val_full, y_val)
    
    # Print comparison
    print(f"\nClassifier Comparison for {target}:")
    print("-" * 60)
    print(f"{'Metric':<15} {'First Half':<12} {'Full Range':<12} {'Difference':<12}")
    print("-" * 60)
    
    for metric_name in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        first_val = metrics_first[metric_name]
        full_val = metrics_full[metric_name]
        diff = full_val - first_val
        print(f"{metric_name:<15} {first_val:<12.4f} {full_val:<12.4f} {diff:<+12.4f}")
    
    # Step 6: Plot ROC curves
    print("Creating ROC curves...")
    plot_roc_curves(y_val, y_pred_proba_first, y_pred_proba_full, target)
    
    # Step 7: PCA visualization for first half classifier
    print("Creating PCA visualization...")
    plot_pca_visualization(X_val_first, y_val, clf_first, target)
    
print(f"\n{'='*50}")
print("Analysis complete! Generated files:")
for target in targets:
    print(f"- {target}_accuracy_heatmap.png")
    print(f"- {target}_roc_curves.png") 
    print(f"- {target}_pca_visualization.png")
print(f"{'='*50}")