"""
Training script for the decision model with explanation generation.

This script trains a model that can make binary decisions and generate
natural language explanations for those decisions.
"""

import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

from core.explanation_model import DecisionWithExplanationModel
from utils.config_loader import config_loader
from utils.logger import setup_logger

logger = logging.getLogger(__name__)


class DecisionExplanationDataset(Dataset):
    """Dataset for training decision-making with explanations."""
    
    def __init__(self, data_path: str, tokenizer, max_input_length: int = 128, max_explanation_length: int = 128):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_explanation_length = max_explanation_length
        
        # Load data
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Create label mapping
        self.label_to_id = {"No": 0, "Yes": 1}
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}
        
        logger.info(f"Loaded {len(self.data)} examples from {data_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize input
        input_encoding = self.tokenizer(
            item['input'],
            truncation=True,
            padding='max_length',
            max_length=self.max_input_length,
            return_tensors='pt'
        )
        
        # Tokenize explanation
        explanation_encoding = self.tokenizer(
            item['explanation'],
            truncation=True,
            padding='max_length',
            max_length=self.max_explanation_length,
            return_tensors='pt'
        )
        
        # Convert decision to label
        decision_label = self.label_to_id[item['decision']]
        
        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'decision_label': torch.tensor(decision_label, dtype=torch.long),
            'explanation_input_ids': explanation_encoding['input_ids'].squeeze(),
            'explanation_attention_mask': explanation_encoding['attention_mask'].squeeze(),
            'explanation_labels': explanation_encoding['input_ids'].squeeze()  # Same as input for teacher forcing
        }


def train_model(
    model: DecisionWithExplanationModel,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    device: str = 'cpu',
    save_dir: str = './models/explanation_model'
):
    """Train the decision-explanation model."""
    
    model.to(device)
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Scheduler
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
        
        # Training
        model.train()
        total_train_loss = 0
        total_classification_loss = 0
        total_generation_loss = 0
        
        for batch_idx, batch in enumerate(train_dataloader):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            decision_labels = batch['decision_label'].to(device)
            explanation_input_ids = batch['explanation_input_ids'].to(device)
            explanation_attention_mask = batch['explanation_attention_mask'].to(device)
            explanation_labels = batch['explanation_labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decision_labels=decision_labels,
                explanation_input_ids=explanation_input_ids,
                explanation_attention_mask=explanation_attention_mask,
                explanation_labels=explanation_labels
            )
            
            # Calculate combined loss
            classification_loss = outputs['classification_loss']
            generation_loss = outputs['generation_loss']
            
            # Weight the losses (classification is more important)
            total_loss = 0.7 * classification_loss + 0.3 * generation_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            # Track losses
            total_train_loss += total_loss.item()
            total_classification_loss += classification_loss.item()
            total_generation_loss += generation_loss.item()
            
            if batch_idx % 10 == 0:
                logger.info(
                    f"Epoch {epoch + 1}, Batch {batch_idx}, "
                    f"Total Loss: {total_loss.item():.4f}, "
                    f"Classification: {classification_loss.item():.4f}, "
                    f"Generation: {generation_loss.item():.4f}"
                )
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_classification_loss = total_classification_loss / len(train_dataloader)
        avg_generation_loss = total_generation_loss / len(train_dataloader)
        
        logger.info(
            f"Epoch {epoch + 1} Training - "
            f"Avg Loss: {avg_train_loss:.4f}, "
            f"Classification: {avg_classification_loss:.4f}, "
            f"Generation: {avg_generation_loss:.4f}"
        )
        
        # Validation
        model.eval()
        total_val_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                decision_labels = batch['decision_label'].to(device)
                explanation_input_ids = batch['explanation_input_ids'].to(device)
                explanation_attention_mask = batch['explanation_attention_mask'].to(device)
                explanation_labels = batch['explanation_labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decision_labels=decision_labels,
                    explanation_input_ids=explanation_input_ids,
                    explanation_attention_mask=explanation_attention_mask,
                    explanation_labels=explanation_labels
                )
                
                classification_loss = outputs['classification_loss']
                generation_loss = outputs['generation_loss']
                total_loss = 0.7 * classification_loss + 0.3 * generation_loss
                
                total_val_loss += total_loss.item()
                
                # Calculate accuracy
                predictions = torch.argmax(outputs['classification_logits'], dim=-1)
                correct_predictions += (predictions == decision_labels).sum().item()
                total_predictions += decision_labels.size(0)
        
        avg_val_loss = total_val_loss / len(val_dataloader)
        accuracy = correct_predictions / total_predictions
        
        logger.info(
            f"Epoch {epoch + 1} Validation - "
            f"Loss: {avg_val_loss:.4f}, "
            f"Accuracy: {accuracy:.4f}"
        )
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            logger.info(f"New best model! Saving to {save_dir}")
            model.save_pretrained(save_dir)
    
    logger.info("Training completed!")


def main():
    """Main training function."""
    setup_logger("train_explanation_model", "INFO")
    
    # Configuration
    config = {
        'model_name': 'bert-base-uncased',
        'max_input_length': 128,
        'max_explanation_length': 128,
        'batch_size': 8,
        'learning_rate': 2e-5,
        'num_epochs': 5,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    logger.info(f"Training configuration: {config}")
    logger.info(f"Using device: {config['device']}")
    
    # Create model
    model = DecisionWithExplanationModel(
        model_name=config['model_name'],
        max_explanation_length=config['max_explanation_length']
    )
    
    # Create datasets
    train_dataset = DecisionExplanationDataset(
        data_path='./data/training/train_with_explanations.json',
        tokenizer=model.tokenizer,
        max_input_length=config['max_input_length'],
        max_explanation_length=config['max_explanation_length']
    )
    
    val_dataset = DecisionExplanationDataset(
        data_path='./data/training/validation_with_explanations.json',
        tokenizer=model.tokenizer,
        max_input_length=config['max_input_length'],
        max_explanation_length=config['max_explanation_length']
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    # Train model
    train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_epochs=config['num_epochs'],
        learning_rate=config['learning_rate'],
        device=config['device']
    )


if __name__ == "__main__":
    main()