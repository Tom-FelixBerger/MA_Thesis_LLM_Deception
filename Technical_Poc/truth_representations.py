import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from typing import List, Dict, Tuple
import random
import warnings
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

        # Storage for attention weights
        self.attention_weights = []

    def extract_attention_features_improved(self, texts: List[str]) -> np.ndarray:
        """
        Extract more sophisticated attention features.
        """
        all_features = []
        
        print(f"Processing {len(texts)} texts...")
        
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
            
            # Extract multiple types of features
            features = []
            final_token_idx = inputs['attention_mask'].sum(dim=1) - 1
            
            for layer_idx, layer_attention in enumerate(attention_weights):
                batch_size, num_heads, seq_len, _ = layer_attention.shape
                
                # 1. Max attention weight per head (what the final token attends to most)
                final_token_attention = layer_attention[0, :, final_token_idx[0], :]
                max_attention = final_token_attention.max(dim=1)[0]  # Shape: (num_heads,)
                features.extend(max_attention.cpu().numpy())
                
                # 2. Attention entropy per head (how focused/distributed the attention is)
                # Add small epsilon to avoid log(0)
                final_token_attention_prob = final_token_attention + 1e-8
                entropy = -(final_token_attention_prob * torch.log(final_token_attention_prob)).sum(dim=1)
                features.extend(entropy.cpu().numpy())
                
                # 3. Self-attention weight (how much final token attends to itself)
                self_attention = final_token_attention[:, final_token_idx[0]]
                features.extend(self_attention.cpu().numpy())
                
                # 4. Attention to specific token types (numbers, operators)
                # This requires more sophisticated tokenization analysis
                # For now, we'll use attention to early tokens (likely contain the numbers)
                early_attention = final_token_attention[:, :3].mean(dim=1)  # First 3 tokens
                features.extend(early_attention.cpu().numpy())
            
            all_features.append(features)
        
        return np.array(all_features)

def train_classifier(features: np.ndarray, labels: List[int]) -> RandomForestClassifier:
    """
    Train a classifier on the extracted attention features.
    
    Args:
        features: Attention features array
        labels: Class labels (0, 1, 2, 3 for 4 classes)
        
    Returns:
        Trained classifier
    """
    print(f"Training classifier on {features.shape[0]} samples with {features.shape[1]} features")
    
    # Check if we have features
    if features.shape[1] == 0:
        raise ValueError("No features extracted. Check the attention extraction process.")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Train classifier
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train, y_train)
    
    # Evaluate
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return classifier

def train_and_predict(sample_texts: List[str], sample_labels: List[int], test_texts: List[str]) -> None:
    print(f"Total samples: {len(sample_texts)}")
    print(f"Total labels: {len(sample_labels)}")
    print(f"Class distribution: {[sample_labels.count(i) for i in range(2)]}")

    # Initialize extractor
    extractor = MistralAttentionExtractor()
    
    # Extract attention features
    features = extractor.extract_attention_features_improved(sample_texts)
    
    print(f"Extracted features shape: {features.shape}")
    
    # Train classifier
    classifier = train_classifier(features, sample_labels)
    
    # Example prediction on new text
    new_features = extractor.extract_attention_features_improved(test_texts)
    prediction = classifier.predict(new_features)
        
    for i in range(len(test_texts)):
        print(f"Prediction for \"{test_texts[i]}\":\n{prediction[i]}")

# Example usage
if __name__ == "__main__":
    
    # Define number words
    numbers = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",]

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

    for a in range(len(numbers)):         # 0 to 5
        for b in range(len(numbers)-a):     # 0 to 5
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
    sample_texts_A = true_statements + false_statements
    sample_labels_A = true_labels + false_labels

    combined = list(zip(sample_texts_A, sample_labels_A))
    random.shuffle(combined)
    sample_texts_A, sample_labels_A = zip(*combined)
    sample_texts_A = list(sample_texts_A)
    sample_labels_A = list(sample_labels_A)
    # print(f"All samples: \n{combined}") # sanity check was successful, no need to print anymore

    train_and_predict(sample_texts_A[10:], sample_labels_A[10:], sample_texts_A[:10])  # Use first 10 for testing
