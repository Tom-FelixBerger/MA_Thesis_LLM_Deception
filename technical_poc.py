import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from typing import List, Dict, Tuple
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
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            token=os.getenv("HUGGINGFACE_TOKEN")
        )

        self.model.eval()  # Set model to evaluation mode

        # Storage for attention weights
        self.attention_weights = []
    
    def _attention_hook(self, module, input, output):
        """Hook function to capture attention weights from each layer."""

        # output[1] contains the attention weights
### COPIED FROM HERE ###
        # Shape: (batch_size, num_heads, seq_len, seq_len)
        attention_weights = output[1]
        if attention_weights is not None:
            self.attention_weights.append(attention_weights.detach())
    
    def register_hooks(self):
        """
        Register hooks to capture attention weights from all layers.
        """
        self.hooks = []
        for layer in self.model.model.layers:
            hook = layer.self_attn.register_forward_hook(self._attention_hook)
            self.hooks.append(hook)
    
    def remove_hooks(self):
        """
        Remove all registered hooks.
        """
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
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
            
            # Clear previous attention weights
            self.attention_weights = []
            
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
            
            # Register hooks
            self.register_hooks()
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs, output_attentions=True)
            
            # Remove hooks
            self.remove_hooks()
            
            # Extract features from final token position
            features = []
            final_token_idx = inputs['attention_mask'].sum(dim=1) - 1  # Last non-padding token
            
            for layer_attention in self.attention_weights:
                # layer_attention shape: (batch_size, num_heads, seq_len, seq_len)
                batch_size, num_heads, seq_len, _ = layer_attention.shape
                
                # Get attention weights for the final token
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

# Example usage
if __name__ == "__main__":
    # Example data - replace with your actual data
    sample_texts = [
        "This is a positive example of class 0.",
        "This represents class 1 with different characteristics.",
        "Class 2 examples have this particular pattern.",
        "Finally, class 3 shows these features.",
        "Another class 0 example with similar properties.",
        "More class 1 data for training.",
        "Additional class 2 sample text.",
        "Class 3 example with distinct attributes.",
        "This is an example of class 0.",
        "This represents class 1 yet again.",
        "Now, class 2 examples have this pattern.",
        "While class 3 shows these features.",
        "Look at this class 0 example.",
        "Aha, another class 1 data point.",
        "And here we have class 2 again.",
        "One final example of class 3 ."
    ]
    
    sample_labels = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]  # Corresponding class labels
    
    # Initialize extractor
    extractor = MistralAttentionExtractor()
    
    # Extract attention features
    features = extractor.extract_attention_features(sample_texts)
    
    print(f"Extracted features shape: {features.shape}")
    
    # Train classifier
    classifier = train_classifier(features, sample_labels)
    
    # Example prediction on new text
    new_text = ["This is a test example of class 3 for prediction."]
    new_features = extractor.extract_attention_features(new_text)
    prediction = classifier.predict(new_features)
    
    print(f"Prediction for new text: Class {prediction[0]}")