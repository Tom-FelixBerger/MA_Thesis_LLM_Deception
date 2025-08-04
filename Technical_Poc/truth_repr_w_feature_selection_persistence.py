import os
import torch
import numpy as np
import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from typing import List, Dict, Tuple
import random
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

class MistralAttentionExtractor:
    def __init__(self, model_name="mistralai/Mistral-7B-v0.3"):
        """Initialize the Mistral model and tokenizer for attention extraction.
        
        Args:
            model_name: HuggingFace model identifier
        """

        print("Loading tokenizer and model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.getenv("HUGGINGFACE_TOKEN"))

        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with device mapping for efficiency
        # Force eager attention to support output_attentions
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            token=os.getenv("HUGGINGFACE_TOKEN"),
            attn_implementation="eager"  # Force eager attention
        )

        self.model.eval()  # Set model to evaluation mode
        self.num_layers = self.model.config.num_hidden_layers
        print(f"Model has {self.num_layers} layers")

    def extract_attention_features_with_layer_info(self, texts: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        Extract attention features with detailed layer information for feature importance analysis.
        
        Returns:
            features: Feature array
            feature_names: List of feature names with layer information
        """
        all_features = []
        feature_names = []
        
        print(f"Processing {len(texts)} texts...")
        
        # Build feature names first
        for layer_idx in range(self.num_layers):
            num_heads = self.model.config.num_attention_heads
            
            # Feature types per head
            for head_idx in range(num_heads):
                feature_names.append(f"layer_{layer_idx}_head_{head_idx}_max_attention")
                feature_names.append(f"layer_{layer_idx}_head_{head_idx}_attention_entropy") 
                feature_names.append(f"layer_{layer_idx}_head_{head_idx}_self_attention")
                feature_names.append(f"layer_{layer_idx}_head_{head_idx}_early_attention")
        
        for i, text in enumerate(texts):
            if i % 10 == 0:
                print(f"Processing text {i+1}/{len(texts)}")
            
            # Tokenize input
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            )
            
            # Move to the same device as model
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Forward pass with attention outputs
            with torch.no_grad():
                outputs = self.model(**inputs, output_attentions=True)
            
            attention_weights = outputs.attentions
            
            # Extract features
            features = []
            final_token_idx = inputs['attention_mask'].sum(dim=1) - 1
            
            for layer_idx, layer_attention in enumerate(attention_weights):
                batch_size, num_heads, seq_len, _ = layer_attention.shape
                
                # Get final token attention for this layer
                final_token_attention = layer_attention[0, :, final_token_idx[0], :]
                
                for head_idx in range(num_heads):
                    head_attention = final_token_attention[head_idx]
                    
                    # 1. Max attention weight
                    max_attention = head_attention.max().cpu().numpy()
                    features.append(max_attention)
                    
                    # 2. Attention entropy
                    head_attention_prob = head_attention + 1e-8
                    entropy = -(head_attention_prob * torch.log(head_attention_prob)).sum().cpu().numpy()
                    features.append(entropy)
                    
                    # 3. Self-attention weight
                    self_attention = head_attention[final_token_idx[0]].cpu().numpy()
                    features.append(self_attention)
                    
                    # 4. Early token attention (first 3 tokens)
                    early_attention = head_attention[:3].mean().cpu().numpy()
                    features.append(early_attention)
            
            all_features.append(features)
        
        return np.array(all_features), feature_names

def analyze_feature_importance(features: np.ndarray, labels: List[int], feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Analyze feature importance and select the most important early-layer features.
    
    Args:
        features: Full feature array
        labels: Class labels
        feature_names: Names of all features
        
    Returns:
        indices of selected features, names of selected features
    """
    print("Analyzing feature importance...")
    
    # Train a classifier on all features to get importance
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Train Random Forest to get feature importance
    rf_full = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_full.fit(X_train, y_train)
    
    # Get feature importances
    importances = rf_full.feature_importances_
    
    # Create importance dataframe for analysis
    import pandas as pd
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    
    # Add layer information
    importance_df['layer'] = importance_df['feature'].str.extract(r'layer_(\d+)').astype(int)
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    print("\nTop 20 most important features:")
    print(importance_df.head(20))
    
    # Focus on early layers (first half of the model)
    num_layers = max(importance_df['layer']) + 1
    early_layer_threshold = num_layers // 2
    print(f"\nFocusing on early layers (0 to {early_layer_threshold-1})")
    
    # Filter for early layers and select top features
    early_layer_features = importance_df[importance_df['layer'] < early_layer_threshold]
    
    # Select top features from early layers (e.g., top 50% of early layer features)
    num_selected = max(20, len(early_layer_features) // 4)  # At least 20 features
    selected_features = early_layer_features.head(num_selected)
    
    print(f"\nSelected {len(selected_features)} features from early layers:")
    print(selected_features)
    
    # Get indices of selected features
    selected_indices = []
    selected_names = []
    for _, row in selected_features.iterrows():
        idx = feature_names.index(row['feature'])
        selected_indices.append(idx)
        selected_names.append(row['feature'])
    
    # Find the last layer used
    last_layer_used = selected_features['layer'].max()
    print(f"\nLast layer used by classifier: {last_layer_used}")
    print(f"Finetuning should affect layers {last_layer_used + 1} onwards")
    
    return np.array(selected_indices), selected_names, last_layer_used

def train_subset_classifier(features: np.ndarray, labels: List[int], selected_indices: np.ndarray, selected_names: List[str]) -> RandomForestClassifier:
    """
    Train a classifier on the selected subset of features.
    """
    print(f"\nTraining classifier on {len(selected_indices)} selected features...")
    
    # Select subset of features
    subset_features = features[:, selected_indices]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        subset_features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Train classifier
    classifier = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10)
    classifier.fit(X_train, y_train)
    
    # Evaluate
    y_pred = classifier.predict(X_test)
    y_prob = classifier.predict_proba(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Subset Classifier Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['False', 'True']))
    
    # Show feature importance for the subset
    subset_importance = classifier.feature_importances_
    importance_subset_df = pd.DataFrame({
        'feature': selected_names,
        'importance': subset_importance
    }).sort_values('importance', ascending=False)
    
    print("\nFeature importance in subset classifier:")
    print(importance_subset_df.head(10))
    
    return classifier

def save_classifier_and_metadata(classifier, selected_indices, selected_names, last_layer_used, save_dir="saved_models"):
    """
    Save the trained classifier and all necessary metadata.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save classifier
    classifier_path = os.path.join(save_dir, "attention_classifier.pkl")
    with open(classifier_path, 'wb') as f:
        pickle.dump(classifier, f)
    
    # Save metadata
    metadata = {
        'selected_indices': selected_indices,
        'selected_names': selected_names,
        'last_layer_used': last_layer_used,
        'num_features': len(selected_indices)
    }
    
    metadata_path = os.path.join(save_dir, "classifier_metadata.pkl")
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"\nSaved classifier to: {classifier_path}")
    print(f"Saved metadata to: {metadata_path}")
    print(f"Last layer used by classifier: {last_layer_used}")
    print(f"Layers to finetune: {last_layer_used + 1} onwards")

def main():
    # Define number words
    numbers = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]

    # True statements (a + b = result)
    true_statements = []
    true_labels = []
    for a in range(len(numbers)):             # 0 to 10
        for b in range(len(numbers)-a):       # sum may not exceed 10
            result = a + b
            sentence = f"{numbers[a].capitalize()} plus {numbers[b]} is {numbers[result]}."
            true_statements.append(sentence)
            true_labels.append(1)

    # False statements (a + b != result)
    false_statements = []
    false_labels = []
    offsets = [-4, -3, -2, -1, 1, 2, 3, 4]  # plausible wrong results

    for a in range(len(numbers)):
        for b in range(len(numbers)-a):
            correct = a + b
            offset = random.choice(offsets)  # Randomly choose an offset
            wrong = correct + offset
            if wrong < 0:
                wrong = wrong + 5 # will still be wrong
            elif wrong > len(numbers) - 1:
                wrong = wrong - 5 # will still be wrong
            sentence = f"{numbers[a].capitalize()} plus {numbers[b]} is {numbers[wrong]}."
            false_statements.append(sentence)
            false_labels.append(0)

    # Combine and shuffle
    sample_texts = true_statements + false_statements
    sample_labels = true_labels + false_labels

    combined = list(zip(sample_texts, sample_labels))
    random.shuffle(combined)
    sample_texts, sample_labels = zip(*combined)
    sample_texts = list(sample_texts)
    sample_labels = list(sample_labels)

    print(f"Total samples: {len(sample_texts)}")
    print(f"Class distribution: {[sample_labels.count(i) for i in range(2)]}")

    # Initialize extractor
    extractor = MistralAttentionExtractor()
    
    # Extract attention features with layer information
    features, feature_names = extractor.extract_attention_features_with_layer_info(sample_texts)
    print(f"Extracted features shape: {features.shape}")
    
    # Analyze feature importance and select subset
    selected_indices, selected_names, last_layer_used = analyze_feature_importance(features, sample_labels, feature_names)
    
    # Train classifier on selected features
    classifier = train_subset_classifier(features, sample_labels, selected_indices, selected_names)
    
    # Save classifier and metadata
    save_classifier_and_metadata(classifier, selected_indices, selected_names, last_layer_used)

if __name__ == "__main__":
    main()