"""
Enhanced model for decision making with explanation generation.

This module implements a model that can simultaneously make binary decisions
and generate natural language explanations for those decisions.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class DecisionWithExplanationModel(nn.Module):
    """
    A model that makes binary decisions and generates explanations.
    
    This model uses a shared encoder (BERT-like) and two separate heads:
    1. Classification head for binary decisions (Yes/No)
    2. Generation head for explanations
    """
    
    def __init__(self, model_name: str = "bert-base-uncased", max_explanation_length: int = 128):
        super().__init__()
        
        self.model_name = model_name
        self.max_explanation_length = max_explanation_length
        
        # Load pre-trained model and config
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add special tokens if not present
        special_tokens = {'pad_token': '[PAD]', 'sep_token': '[SEP]', 'cls_token': '[CLS]'}
        num_added_tokens = self.tokenizer.add_special_tokens(special_tokens)
        if num_added_tokens > 0:
            self.encoder.resize_token_embeddings(len(self.tokenizer))
        
        # Classification head for binary decisions
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.config.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2)  # 2 classes: Yes/No
        )
        
        # Explanation generation head
        self.explanation_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size, len(self.tokenizer))  # Vocabulary size
        )
        
        # Loss functions
        self.classification_loss = nn.CrossEntropyLoss()
        self.generation_loss = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decision_labels: Optional[torch.Tensor] = None,
        explanation_input_ids: Optional[torch.Tensor] = None,
        explanation_attention_mask: Optional[torch.Tensor] = None,
        explanation_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for both classification and generation.
        
        Args:
            input_ids: Input token IDs for the question
            attention_mask: Attention mask for the question
            decision_labels: Target labels for classification (optional, for training)
            explanation_input_ids: Input IDs for explanation generation (optional)
            explanation_attention_mask: Attention mask for explanation (optional)
            explanation_labels: Target explanation tokens (optional, for training)
            
        Returns:
            Dictionary containing logits and losses
        """
        # Encode the input question
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get [CLS] token representation for classification
        cls_representation = encoder_outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        
        # Classification head
        classification_logits = self.classifier(cls_representation)  # [batch_size, 2]
        
        outputs = {
            'classification_logits': classification_logits,
            'encoder_hidden_states': encoder_outputs.last_hidden_state
        }
        
        # Calculate classification loss if labels provided
        if decision_labels is not None:
            classification_loss = self.classification_loss(classification_logits, decision_labels)
            outputs['classification_loss'] = classification_loss
        
        # Explanation generation (simplified - using teacher forcing during training)
        if explanation_input_ids is not None:
            # Use the encoder representation to guide explanation generation
            batch_size, seq_len = explanation_input_ids.shape
            
            # Expand cls representation to match explanation sequence length
            expanded_cls = cls_representation.unsqueeze(1).expand(batch_size, seq_len, -1)
            
            # Simple approach: project expanded cls to vocabulary
            explanation_logits = self.explanation_head(expanded_cls)  # [batch_size, seq_len, vocab_size]
            
            outputs['explanation_logits'] = explanation_logits
            
            # Calculate generation loss if labels provided
            if explanation_labels is not None:
                generation_loss = self.generation_loss(
                    explanation_logits.view(-1, explanation_logits.size(-1)),
                    explanation_labels.view(-1)
                )
                outputs['generation_loss'] = generation_loss
        
        return outputs
    
    def generate_explanation(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
        max_length: Optional[int] = None
    ) -> List[str]:
        """
        Generate explanations for given inputs.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            max_length: Maximum explanation length
            
        Returns:
            List of generated explanations
        """
        if max_length is None:
            max_length = self.max_explanation_length
            
        self.eval()
        with torch.no_grad():
            # Get encoder representation
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            
            cls_representation = encoder_outputs.last_hidden_state[:, 0, :]
            batch_size = cls_representation.size(0)
            
            # Simple greedy generation
            generated_ids = []
            current_token = torch.full((batch_size, 1), self.tokenizer.cls_token_id, device=input_ids.device)
            
            for _ in range(max_length):
                # Expand cls representation
                expanded_cls = cls_representation.unsqueeze(1)
                
                # Get next token logits
                token_logits = self.explanation_head(expanded_cls).squeeze(1)  # [batch_size, vocab_size]
                
                # Get next token (greedy)
                next_token = torch.argmax(token_logits, dim=-1, keepdim=True)  # [batch_size, 1]
                
                generated_ids.append(next_token)
                current_token = next_token
                
                # Stop if all sequences generated EOS token
                if (next_token == self.tokenizer.sep_token_id).all():
                    break
            
            # Concatenate generated tokens
            if generated_ids:
                generated_sequences = torch.cat(generated_ids, dim=1)  # [batch_size, generated_length]
                
                # Decode to text
                explanations = []
                for sequence in generated_sequences:
                    # Remove special tokens and decode
                    tokens = sequence.cpu().tolist()
                    explanation = self.tokenizer.decode(tokens, skip_special_tokens=True)
                    explanations.append(explanation)
                
                return explanations
            else:
                return ["" for _ in range(batch_size)]
    
    def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, Any]:
        """
        Make predictions for both classification and explanation.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            Dictionary with predictions
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask)
            
            # Get classification prediction
            classification_logits = outputs['classification_logits']
            classification_probs = torch.softmax(classification_logits, dim=-1)
            predicted_class = torch.argmax(classification_logits, dim=-1)
            confidence = torch.max(classification_probs, dim=-1)[0]
            
            # Generate explanations
            explanations = self.generate_explanation(input_ids, attention_mask)
            
            return {
                'predictions': predicted_class.cpu().tolist(),
                'confidences': confidence.cpu().tolist(),
                'explanations': explanations
            }
    
    def save_pretrained(self, save_directory: str) -> None:
        """Save the model to a directory."""
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        # Save model state
        torch.save(self.state_dict(), os.path.join(save_directory, 'model.pt'))
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_directory)
        
        # Save config
        config_dict = {
            'model_name': self.model_name,
            'max_explanation_length': self.max_explanation_length,
            'vocab_size': len(self.tokenizer)
        }
        
        import json
        with open(os.path.join(save_directory, 'model_config.json'), 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load_pretrained(cls, load_directory: str) -> 'DecisionWithExplanationModel':
        """Load the model from a directory."""
        import os
        import json
        
        # Load config
        with open(os.path.join(load_directory, 'model_config.json'), 'r') as f:
            config_dict = json.load(f)
        
        # Create model
        model = cls(
            model_name=config_dict['model_name'],
            max_explanation_length=config_dict['max_explanation_length']
        )
        
        # Load state dict
        state_dict = torch.load(os.path.join(load_directory, 'model.pt'), map_location='cpu')
        model.load_state_dict(state_dict)
        
        return model