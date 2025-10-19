#!/usr/bin/env python3
"""Train distilled LLaVA-3D using pre-generated teacher responses - Fixed version."""

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import argparse
from typing import Dict, List, Any
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.distillation.student_model import DistilledLLaVA3D, DistilledLLaVA3DConfig
from scripts.distillation.distillation_loss import KnowledgeDistillationLoss
from PIL import Image
import torchvision.transforms as transforms

class TeacherResponseDataset(Dataset):
    """Dataset for teacher response distillation."""
    
    def __init__(self, manifest_path: str, data_root: str = None):
        with open(manifest_path, 'r') as f:
            self.samples = json.load(f)
        
        self.data_root = Path(data_root) if data_root else Path(".")
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load and process image
        image_path = sample["image_path"]
        image_path_str = str(image_path)  # Convert to string for string operations
        
        # Handle URL images (use a placeholder for now)
        if image_path_str.startswith("http"):
            # Create a random image for URL-based samples
            image = Image.new('RGB', (224, 224), color=(100, 150, 200))
        else:
            if not Path(image_path).is_absolute():
                image_path = self.data_root / image_path
            image = Image.open(image_path).convert('RGB')
        
        pixel_values = self.transform(image)
        
        # Tokenize question and answer
        question = sample["question"]
        teacher_answer = sample["answer"]
        
        # Simple tokenization (in practice, you'd use a proper tokenizer)
        question_tokens = self.simple_tokenize(question)
        answer_tokens = self.simple_tokenize(teacher_answer)
        
        return {
            'pixel_values': pixel_values,
            'question': question,
            'question_tokens': question_tokens,
            'teacher_answer': teacher_answer,
            'answer_tokens': answer_tokens
        }
    
    def simple_tokenize(self, text: str) -> torch.Tensor:
        """Simple tokenization for demonstration."""
        # Convert to lowercase and split
        words = text.lower().split()
        # Create a simple vocabulary mapping
        vocab = {word: i + 1 for i, word in enumerate(set(words))}
        vocab['<pad>'] = 0
        vocab['<unk>'] = len(vocab)
        
        # Convert words to token IDs
        token_ids = [vocab.get(word, vocab['<unk>']) for word in words]
        return torch.tensor(token_ids, dtype=torch.long)

def collate_fn(batch):
    """Custom collate function to handle variable length sequences."""
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    questions = [item['question'] for item in batch]
    teacher_answers = [item['teacher_answer'] for item in batch]
    
    # Handle variable length sequences
    question_tokens = [item['question_tokens'] for item in batch]
    answer_tokens = [item['answer_tokens'] for item in batch]
    
    # Pad sequences to the same length
    max_question_len = max(len(tokens) for tokens in question_tokens)
    max_answer_len = max(len(tokens) for tokens in answer_tokens)
    
    # Pad question tokens
    padded_question_tokens = []
    for tokens in question_tokens:
        if len(tokens) < max_question_len:
            padding = torch.zeros(max_question_len - len(tokens), dtype=tokens.dtype)
            padded_tokens = torch.cat([tokens, padding])
        else:
            padded_tokens = tokens
        padded_question_tokens.append(padded_tokens)
    
    # Pad answer tokens
    padded_answer_tokens = []
    for tokens in answer_tokens:
        if len(tokens) < max_answer_len:
            padding = torch.zeros(max_answer_len - len(tokens), dtype=tokens.dtype)
            padded_tokens = torch.cat([tokens, padding])
        else:
            padded_tokens = tokens
        padded_answer_tokens.append(padded_tokens)
    
    return {
        'pixel_values': pixel_values,
        'question': questions,
        'question_tokens': torch.stack(padded_question_tokens),
        'teacher_answer': teacher_answers,
        'answer_tokens': torch.stack(padded_answer_tokens)
    }

class DistillationTrainer:
    """Trainer for distilled LLaVA-3D using teacher responses."""
    
    def __init__(self, config: DistilledLLaVA3DConfig, device: str = "cuda"):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Initialize models
        self.student_model = DistilledLLaVA3D(config).to(self.device)
        
        # Loss function
        self.distillation_loss = KnowledgeDistillationLoss(
            temperature=3.0,
            alpha=0.7
        )
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.student_model.parameters(),
            lr=1e-4,
            weight_decay=0.01
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=10
        )
    
    def train_epoch(self, dataloader: DataLoader, epoch: int):
        """Train for one epoch."""
        self.student_model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            self.optimizer.zero_grad()
            
            # Move data to device
            pixel_values = batch['pixel_values'].to(self.device)
            question_tokens = batch['question_tokens'].to(self.device)
            answer_tokens = batch['answer_tokens'].to(self.device)
            
            # Create attention masks
            question_mask = (question_tokens != 0).long()
            answer_mask = (answer_tokens != 0).long()
            
            # Forward pass
            try:
                # Get student predictions
                student_outputs = self.student_model(
                    input_ids=question_tokens,
                    attention_mask=question_mask,
                    pixel_values=pixel_values.unsqueeze(0)  # Add batch dimension
                )
                
                # Calculate loss (simplified - in practice you'd use proper language modeling loss)
                student_logits = student_outputs.logits
                
                # Simple cross-entropy loss for demonstration
                target_tokens = answer_tokens[:, :student_logits.size(1)]
                if target_tokens.size(1) < student_logits.size(1):
                    # Pad target tokens
                    padding = torch.zeros(
                        target_tokens.size(0), 
                        student_logits.size(1) - target_tokens.size(1),
                        dtype=target_tokens.dtype,
                        device=target_tokens.device
                    )
                    target_tokens = torch.cat([target_tokens, padding], dim=1)
                
                # Reshape for cross-entropy
                student_logits_flat = student_logits.view(-1, student_logits.size(-1))
                target_tokens_flat = target_tokens.view(-1)
                
                # Mask out padding tokens
                mask = (target_tokens_flat != 0)
                if mask.sum() > 0:
                    loss = nn.CrossEntropyLoss()(
                        student_logits_flat[mask], 
                        target_tokens_flat[mask]
                    )
                else:
                    loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), 1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / max(num_batches, 1)
        print(f"Epoch {epoch} Average Loss: {avg_loss:.4f}")
        return avg_loss
    
    def train(self, dataloader: DataLoader, num_epochs: int = 5):
        """Train the model."""
        print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            avg_loss = self.train_epoch(dataloader, epoch)
            self.scheduler.step()
            
            # Save checkpoint
            if (epoch + 1) % 2 == 0:
                checkpoint_path = f"models/checkpoints/distilled_llava3d_teacher_responses_epoch_{epoch+1}.pt"
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                torch.save(self.student_model.state_dict(), checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path}")
        
        print("Training completed!")
    
    def evaluate(self, dataloader: DataLoader):
        """Evaluate the model."""
        self.student_model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Similar to training but without gradients
                pixel_values = batch['pixel_values'].to(self.device)
                question_tokens = batch['question_tokens'].to(self.device)
                answer_tokens = batch['answer_tokens'].to(self.device)
                
                # Process and calculate loss (simplified)
                try:
                    question_mask = (question_tokens != 0).long()
                    
                    student_outputs = self.student_model(
                        input_ids=question_tokens,
                        attention_mask=question_mask,
                        pixel_values=pixel_values.unsqueeze(0)
                    )
                    
                    # Calculate loss
                    student_logits = student_outputs.logits
                    target_tokens = answer_tokens[:, :student_logits.size(1)]
                    
                    if target_tokens.size(1) < student_logits.size(1):
                        padding = torch.zeros(
                            target_tokens.size(0), 
                            student_logits.size(1) - target_tokens.size(1),
                            dtype=target_tokens.dtype,
                            device=target_tokens.device
                        )
                        target_tokens = torch.cat([target_tokens, padding], dim=1)
                    
                    student_logits_flat = student_logits.view(-1, student_logits.size(-1))
                    target_tokens_flat = target_tokens.view(-1)
                    
                    mask = (target_tokens_flat != 0)
                    if mask.sum() > 0:
                        loss = nn.CrossEntropyLoss()(
                            student_logits_flat[mask], 
                            target_tokens_flat[mask]
                        )
                        total_loss += loss.item()
                        num_batches += 1
                
                except Exception as e:
                    print(f"Error in evaluation batch: {e}")
                    continue
        
        avg_loss = total_loss / max(num_batches, 1)
        print(f"Evaluation Loss: {avg_loss:.4f}")
        return avg_loss

def main():
    parser = argparse.ArgumentParser(description="Train distilled LLaVA-3D with teacher responses")
    parser.add_argument("--teacher-responses", type=str, default="teacher_responses.json",
                       help="Path to teacher responses JSON file")
    parser.add_argument("--data-root", type=str, default=".",
                       help="Root directory for data paths")
    parser.add_argument("--batch-size", type=int, default=2,
                       help="Batch size for training")
    parser.add_argument("--num-epochs", type=int, default=5,
                       help="Number of training epochs")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device for training")
    
    args = parser.parse_args()
    
    # Create dataset and dataloader with custom collate function
    dataset = TeacherResponseDataset(args.teacher_responses, args.data_root)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    
    print(f"Loaded {len(dataset)} samples")
    
    # Create model and trainer
    config = DistilledLLaVA3DConfig()
    trainer = DistillationTrainer(config, args.device)
    
    # Train the model
    trainer.train(dataloader, args.num_epochs)
    
    # Evaluate
    print("\nEvaluating...")
    trainer.evaluate(dataloader)
    
    print("\nTraining completed! You can now test the model with:")
    print("python run_distilled_llava3d.py --model-path models/checkpoints --image-file <image> --query <question>")

if __name__ == "__main__":
    main()
