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
    
    def extract_attention_features(self, texts: List[str]) -> np.ndarray:
        """
        Extract attention head activations for the final token position.
        
        Args:
            texts: List of input texts
            
        Returns:
            numpy array of shape (n_samples, n_features)
            where n_features = n_layers * n_heads
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
            
            # Get attention weights from the model outputs
            attention_weights = outputs.attentions  # Tuple of attention weights for each layer
            
            # Extract features from final token position
            features = []
            final_token_idx = inputs['attention_mask'].sum(dim=1) - 1  # Last non-padding token
            
            for layer_idx, layer_attention in enumerate(attention_weights):
                # layer_attention shape: (batch_size, num_heads, seq_len, seq_len)
                batch_size, num_heads, seq_len, _ = layer_attention.shape
                
                # Get attention weights FROM the final token (what the final token attends to)
                final_token_attention = layer_attention[0, :, final_token_idx[0], :]
                
                # Average over the sequence dimension to get a single value per head
                head_activations = final_token_attention.mean(dim=1)  # Shape: (num_heads,)
                
                features.extend(head_activations.cpu().numpy())
            
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
    features = extractor.extract_attention_features(sample_texts)
    
    print(f"Extracted features shape: {features.shape}")
    
    # Train classifier
    classifier = train_classifier(features, sample_labels)
    
    # Example prediction on new text
    new_features = extractor.extract_attention_features(test_texts)
    prediction = classifier.predict(new_features)
        
    for i in range(len(test_texts)):
        print(f"Prediction for \"{test_texts[i]}\":\n{prediction[i]}")
    
    # # Optional: Show feature importance
    # feature_importance = classifier.feature_importances_
    # print(f"\nTop 10 most important features:")
    # top_features = np.argsort(feature_importance)[-10:][::-1]
    # for i, feat_idx in enumerate(top_features):
    #     layer_idx = feat_idx // 32  # Assuming 32 heads per layer
    #     head_idx = feat_idx % 32
    #     print(f"{i+1}. Layer {layer_idx}, Head {head_idx}: {feature_importance[feat_idx]:.4f}")

# Example usage
if __name__ == "__main__":
    
    # Define number words
    numbers = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
               "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
               "twenty-one", "twenty-two", "twenty-three", "twenty-four", "twenty-five", "twenty-six", "twenty-seven", "twenty-eight", "twenty-nine", "thirty"]

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
    print(f"All samples: \n{combined}")

    train_and_predict(sample_texts_A[10:], sample_labels_A[10:], sample_texts_A[:10])  # Use first 10 for testing

    # sample_texts_B = [
    #     "Zero plus five is",
    #     "One plus four is",
    #     "Two plus three is",
    #     "Three plus two is",
    #     "Four plus one is",
    #     "Six plus eleven is",
    #     "Seven plus ten is",
    #     "Eight plus nine is",
    #     "Nine plus eight is",
    #     "Ten plus seven is",
    # ]
    # sample_labels_B = [1]*5 + [0]*5  # 1 for answer "five", 0 for answer "seventeen"
    # test_texts_B = [
    #     "Five plus zero is",
    #     "Eleven plus six is",
    # ]
    # train_and_predict(sample_texts_B, sample_labels_B, test_texts_B)
