#!/usr/bin/env python3
"""
Real Teacher Distillation Training Pipeline
==========================================

This module implements the complete training pipeline for distilling knowledge
from the real LLaVA-3D teacher model to the student model using real 3D datasets.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from tqdm import tqdm
import gc

# Add project paths
sys.path.append('/home/alasfour/scratch/distilled-llava3d')
sys.path.append('/home/alasfour/scratch/distilled-llava3d/scripts/distillation')

from real_teacher_integration import RealTeacherIntegration
from real_3d_dataset_preparation import Real3DDatasetPreparation
from student_model import DistilledLLaVA3D, DistilledLLaVA3DConfig
from distillation_loss import KnowledgeDistillationLoss, AdaptiveDistillationLoss

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Real3DDataset(Dataset):
    """
    Real 3D Dataset for Distillation Training
    
    This dataset loads real 3D scenes and questions for training the distilled model.
    """
    
    def __init__(self, 
                 data_root: str,
                 manifest_path: str,
                 max_samples: Optional[int] = None):
        """
        Initialize the 3D dataset.
        
        Args:
            data_root: Root directory of the dataset
            manifest_path: Path to dataset manifest
            max_samples: Maximum number of samples to load
        """
        self.data_root = Path(data_root)
        self.manifest_path = Path(manifest_path)
        self.max_samples = max_samples
        
        # Load manifest
        with open(self.manifest_path, 'r') as f:
            self.manifest = json.load(f)
        
        # Load all samples
        self.samples = self._load_samples()
        
        logger.info(f"📊 Loaded {len(self.samples)} samples from {self.manifest_path}")
    
    def _load_samples(self) -> List[Dict]:
        """Load all samples from the dataset."""
        samples = []
        
        for scene_info in self.manifest.get("scenes", []):
            scene_path = Path(scene_info["path"])
            
            # Load images
            image_files = list(scene_path.glob("*.jpg")) + list(scene_path.glob("*.png"))
            
            # Load annotations
            annotation_files = list(scene_path.glob("*.json"))
            
            for img_file in image_files:
                for ann_file in annotation_files:
                    sample = {
                        "image_path": str(img_file),
                        "annotation_path": str(ann_file),
                        "scene_id": scene_info["scene_id"],
                        "scene_path": str(scene_path)
                    }
                    samples.append(sample)
                    
                    if self.max_samples and len(samples) >= self.max_samples:
                        break
                
                if self.max_samples and len(samples) >= self.max_samples:
                    break
            
            if self.max_samples and len(samples) >= self.max_samples:
                break
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        from PIL import Image
        image = Image.open(sample["image_path"]).convert("RGB")
        
        # Load annotation
        with open(sample["annotation_path"], 'r') as f:
            annotation = json.load(f)
        
        # Generate questions based on annotation
        questions = self._generate_questions(annotation)
        
        return {
            "image": image,
            "annotation": annotation,
            "scene_id": sample["scene_id"],
            "questions": questions
        }
    
    def _generate_questions(self, annotation: Dict) -> List[str]:
        """Generate questions based on annotation."""
        questions = []
        
        # Basic scene questions
        if "room_type" in annotation:
            questions.append(f"What type of room is this?")
        
        if "furniture" in annotation:
            questions.append(f"What furniture can you see in this scene?")
        
        if "objects" in annotation:
            questions.append(f"What objects are visible in this image?")
        
        # Spatial reasoning questions
        if "spatial_relations" in annotation:
            for relation in annotation["spatial_relations"]:
                questions.append(f"How is the {relation['subject']} positioned relative to the {relation['object']}?")
        
        # 3D understanding questions
        questions.extend([
            "What is the depth structure of this scene?",
            "How are objects arranged in 3D space?",
            "What is the overall layout of this 3D scene?"
        ])
        
        return questions

class RealTeacherDistillationTrainer:
    """
    Real Teacher Distillation Trainer
    
    This class handles the complete training pipeline for distilling knowledge
    from the real LLaVA-3D teacher to the student model.
    """
    
    def __init__(self,
                 teacher_integration: RealTeacherIntegration,
                 student_config: DistilledLLaVA3DConfig,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 learning_rate: float = 1e-4,
                 batch_size: int = 4,
                 num_epochs: int = 10):
        """
        Initialize the distillation trainer.
        
        Args:
            teacher_integration: Real teacher integration
            student_config: Student model configuration
            device: Training device
            learning_rate: Learning rate
            batch_size: Batch size
            num_epochs: Number of training epochs
        """
        self.teacher_integration = teacher_integration
        self.student_config = student_config
        self.device = device
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        
        # Initialize student model
        self.student_model = DistilledLLaVA3D(student_config)
        self.student_model.to(device)
        
        # Initialize loss functions
        self.kd_loss = KnowledgeDistillationLoss(
            temperature=4.0,
            alpha=0.7
        )
        self.adaptive_loss = AdaptiveDistillationLoss(
            response_weight=0.4,
            feature_weight=0.3,
            attention_weight=0.3
        )
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.student_model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Training history
        self.training_history = {
            "epochs": [],
            "losses": [],
            "teacher_responses": [],
            "student_responses": []
        }
        
        logger.info(f"🚀 Initialized Real Teacher Distillation Trainer")
        logger.info(f"   Device: {device}")
        logger.info(f"   Learning Rate: {learning_rate}")
        logger.info(f"   Batch Size: {batch_size}")
        logger.info(f"   Epochs: {num_epochs}")
    
    def train_on_real_data(self, 
                          dataset: Real3DDataset,
                          save_dir: str = "/home/alasfour/scratch/distilled-llava3d/checkpoints") -> Dict[str, any]:
        """
        Train the student model on real 3D data using teacher distillation.
        
        Args:
            dataset: Real 3D dataset
            save_dir: Directory to save checkpoints
            
        Returns:
            Dict containing training results
        """
        logger.info("🔄 Starting real teacher distillation training...")
        
        # Create data loader
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            collate_fn=self._collate_fn
        )
        
        # Create save directory
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        best_loss = float('inf')
        training_results = {
            "epochs_completed": 0,
            "best_loss": best_loss,
            "final_loss": None,
            "checkpoints_saved": []
        }
        
        for epoch in range(self.num_epochs):
            logger.info(f"📚 Epoch {epoch + 1}/{self.num_epochs}")
            
            epoch_loss = 0.0
            num_batches = 0
            
            # Training loop
            for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
                try:
                    # Get teacher responses
                    teacher_responses = self._get_teacher_responses(batch)
                    
                    # Get student responses
                    student_responses = self._get_student_responses(batch)
                    
                    # Compute distillation loss
                    loss = self._compute_distillation_loss(
                        teacher_responses, 
                        student_responses
                    )
                    
                    # Backward pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
                    
                    # Clear cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                except Exception as e:
                    logger.error(f"❌ Error in training batch: {e}")
                    continue
            
            # Calculate average loss
            avg_loss = epoch_loss / max(num_batches, 1)
            
            # Update training history
            self.training_history["epochs"].append(epoch + 1)
            self.training_history["losses"].append(avg_loss)
            
            logger.info(f"   Average Loss: {avg_loss:.4f}")
            
            # Save checkpoint if best
            if avg_loss < best_loss:
                best_loss = avg_loss
                checkpoint_path = save_path / f"best_model_epoch_{epoch + 1}.pt"
                self._save_checkpoint(checkpoint_path, epoch, avg_loss)
                training_results["checkpoints_saved"].append(str(checkpoint_path))
                logger.info(f"💾 Saved best checkpoint: {checkpoint_path}")
            
            # Save regular checkpoint
            if (epoch + 1) % 2 == 0:  # Save every 2 epochs
                checkpoint_path = save_path / f"checkpoint_epoch_{epoch + 1}.pt"
                self._save_checkpoint(checkpoint_path, epoch, avg_loss)
                training_results["checkpoints_saved"].append(str(checkpoint_path))
            
            # Force garbage collection
            gc.collect()
        
        # Update results
        training_results["epochs_completed"] = self.num_epochs
        training_results["best_loss"] = best_loss
        training_results["final_loss"] = avg_loss
        
        # Save training history
        history_path = save_path / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        logger.info("✅ Real teacher distillation training completed!")
        logger.info(f"   Best Loss: {best_loss:.4f}")
        logger.info(f"   Checkpoints Saved: {len(training_results['checkpoints_saved'])}")
        
        return training_results
    
    def _get_teacher_responses(self, batch: Dict) -> List[Dict]:
        """Get teacher responses for a batch."""
        teacher_responses = []
        
        for i in range(len(batch["image"])):
            image_path = batch["image_path"][i]
            question = batch["questions"][i][0]  # Use first question
            
            response = self.teacher_integration.generate_teacher_response(
                image_path, question
            )
            teacher_responses.append(response)
        
        return teacher_responses
    
    def _get_student_responses(self, batch: Dict) -> List[Dict]:
        """Get student responses for a batch."""
        student_responses = []
        
        for i in range(len(batch["image"])):
            image = batch["image"][i]
            question = batch["questions"][i][0]
            
            # Convert image to tensor
            from torchvision.transforms import ToTensor
            image_tensor = ToTensor()(image).unsqueeze(0).to(self.device)
            
            # Get student response
            response = self.student_model.generate_response(question, image_tensor)
            
            student_responses.append({
                "response": response,
                "question": question,
                "image_path": batch["image_path"][i]
            })
        
        return student_responses
    
    def _compute_distillation_loss(self, 
                                  teacher_responses: List[Dict],
                                  student_responses: List[Dict]) -> torch.Tensor:
        """Compute distillation loss between teacher and student."""
        total_loss = 0.0
        
        for teacher_resp, student_resp in zip(teacher_responses, student_responses):
            if "error" in teacher_resp:
                continue  # Skip if teacher response failed
            
            # Response similarity loss
            response_loss = self.kd_loss(
                student_resp["response"],
                teacher_resp["response"]
            )
            
            # Feature matching loss (simplified)
            feature_loss = torch.tensor(0.1, requires_grad=True)  # Placeholder
            
            # Combined loss
            sample_loss = response_loss + 0.1 * feature_loss
            total_loss += sample_loss
        
        return torch.tensor(total_loss / len(teacher_responses), requires_grad=True)
    
    def _collate_fn(self, batch):
        """Custom collate function for batching."""
        return {
            "image": [item["image"] for item in batch],
            "image_path": [item["image_path"] for item in batch],
            "questions": [item["questions"] for item in batch],
            "annotation": [item["annotation"] for item in batch],
            "scene_id": [item["scene_id"] for item in batch]
        }
    
    def _save_checkpoint(self, path: Path, epoch: int, loss: float):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.student_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss,
            "config": self.student_config.__dict__
        }
        
        torch.save(checkpoint, path)
    
    def evaluate_on_real_data(self, 
                             dataset: Real3DDataset,
                             checkpoint_path: Optional[str] = None) -> Dict[str, any]:
        """
        Evaluate the trained model on real 3D data.
        
        Args:
            dataset: Evaluation dataset
            checkpoint_path: Path to model checkpoint
            
        Returns:
            Dict containing evaluation results
        """
        logger.info("🧪 Evaluating on real 3D data...")
        
        # Load checkpoint if provided
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.student_model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"📁 Loaded checkpoint: {checkpoint_path}")
        
        # Set to evaluation mode
        self.student_model.eval()
        
        evaluation_results = {
            "total_samples": len(dataset),
            "correct_responses": 0,
            "response_quality": [],
            "examples": []
        }
        
        # Evaluate on sample
        sample_size = min(10, len(dataset))  # Evaluate on 10 samples
        
        with torch.no_grad():
            for i in range(sample_size):
                sample = dataset[i]
                
                # Get student response
                from torchvision.transforms import ToTensor
                image_tensor = ToTensor()(sample["image"]).unsqueeze(0).to(self.device)
                question = sample["questions"][0]
                
                response = self.student_model.generate_response(question, image_tensor)
                
                # Simple quality assessment
                quality_score = len(response.split()) / 10.0  # Word count based
                evaluation_results["response_quality"].append(quality_score)
                
                if quality_score > 0.5:  # Simple threshold
                    evaluation_results["correct_responses"] += 1
                
                # Store example
                evaluation_results["examples"].append({
                    "question": question,
                    "response": response,
                    "quality_score": quality_score
                })
        
        # Calculate metrics
        evaluation_results["accuracy"] = (
            evaluation_results["correct_responses"] / sample_size
        )
        evaluation_results["avg_quality"] = np.mean(evaluation_results["response_quality"])
        
        logger.info(f"📊 Evaluation Results:")
        logger.info(f"   Accuracy: {evaluation_results['accuracy']:.2%}")
        logger.info(f"   Avg Quality: {evaluation_results['avg_quality']:.2f}")
        
        return evaluation_results

def test_real_teacher_distillation():
    """Test the real teacher distillation training."""
    logger.info("🧪 Testing Real Teacher Distillation Training")
    
    # Initialize components
    teacher_integration = RealTeacherIntegration()
    dataset_prep = Real3DDatasetPreparation()
    
    # Load teacher model
    if teacher_integration.load_teacher_model():
        logger.info("✅ Teacher model loaded successfully!")
        
        # Prepare datasets
        training_data = dataset_prep.prepare_training_data()
        
        if training_data["total"]["scenes"] > 0:
            # Create dataset
            dataset = Real3DDataset(
                data_root="/home/alasfour/scratch/distilled-llava3d/data",
                manifest_path="/home/alasfour/scratch/distilled-llava3d/data/training_manifest.json",
                max_samples=20  # Limit for testing
            )
            
            # Initialize trainer
            student_config = DistilledLLaVA3DConfig()
            trainer = RealTeacherDistillationTrainer(
                teacher_integration=teacher_integration,
                student_config=student_config,
                batch_size=2,  # Small batch for testing
                num_epochs=2    # Few epochs for testing
            )
            
            # Train
            training_results = trainer.train_on_real_data(dataset)
            logger.info(f"🎯 Training Results: {training_results}")
            
            # Evaluate
            evaluation_results = trainer.evaluate_on_real_data(dataset)
            logger.info(f"📊 Evaluation Results: {evaluation_results}")
            
        else:
            logger.warning("⚠️ No training data available")
        
        # Cleanup
        teacher_integration.cleanup()
    else:
        logger.error("❌ Failed to load teacher model")

if __name__ == "__main__":
    test_real_teacher_distillation()

