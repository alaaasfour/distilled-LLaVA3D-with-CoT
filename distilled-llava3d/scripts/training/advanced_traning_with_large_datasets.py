#!/usr/bin/env python3
"""
Advanced Training with Large Datasets
====================================

This script runs advanced training using the larger datasets we just downloaded.
It uses the enhanced mock teacher and trains on the comprehensive 3D datasets.
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import numpy as np
from PIL import Image

# Add project paths
sys.path.append('/home/alasfour/scratch/distilled-llava3d')

from scripts.distillation.student_model import DistilledLLaVA3D, DistilledLLaVA3DConfig
from enhanced_mock_teacher import EnhancedMockTeacher
from scripts.distillation.distillation_loss import AdaptiveDistillationLoss
from real_3d_dataset_preparation import Real3DDataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedTrainingWithLargeDatasets:
    """
    Advanced Training with Large Datasets
    
    This class handles training the distilled model using the larger datasets
    we just downloaded (ScanNet, 3D-FRONT, Matterport3D).
    """
    
    def __init__(self, 
                 data_root: str = "/home/alasfour/scratch/distilled-llava3d/data",
                 checkpoint_dir: str = "/home/alasfour/scratch/distilled-llava3d/checkpoints"):
        """
        Initialize advanced training.
        
        Args:
            data_root: Root directory for datasets
            checkpoint_dir: Directory for saving checkpoints
        """
        self.data_root = Path(data_root)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training parameters
        self.epochs = 15  # Increased for larger datasets
        self.batch_size = 8  # Increased batch size
        self.learning_rate = 1e-4
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize components
        self.student_model = None
        self.teacher_model = None
        self.distillation_loss = None
        self.optimizer = None
        
        # Training statistics
        self.training_stats = {
            "epochs_completed": 0,
            "total_loss": 0.0,
            "best_loss": float('inf'),
            "training_time": 0.0,
            "datasets_used": [],
            "total_samples": 0
        }
        
        logger.info(f"🚀 Initializing Advanced Training with Large Datasets")
        logger.info(f"   Data Root: {self.data_root}")
        logger.info(f"   Checkpoint Dir: {self.checkpoint_dir}")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Epochs: {self.epochs}")
        logger.info(f"   Batch Size: {self.batch_size}")
    
    def load_large_datasets(self) -> Dict[str, any]:
        """
        Load all large datasets.
        
        Returns:
            Dict containing dataset information
        """
        logger.info("📊 Loading large datasets...")
        
        datasets_info = {
            "scannet": {"samples": 0, "scenes": 0},
            "3d_front": {"samples": 0, "scenes": 0},
            "matterport3d": {"samples": 0, "scenes": 0},
            "total": {"samples": 0, "scenes": 0}
        }
        
        # Load ScanNet
        scannet_dir = self.data_root / "scannet"
        if scannet_dir.exists():
            scannet_scenes = len(list(scannet_dir.glob("scene*")))
            scannet_images = len(list(scannet_dir.rglob("*.jpg")))
            datasets_info["scannet"]["scenes"] = scannet_scenes
            datasets_info["scannet"]["samples"] = scannet_images
            datasets_info["total"]["scenes"] += scannet_scenes
            datasets_info["total"]["samples"] += scannet_images
            logger.info(f"   ScanNet: {scannet_scenes} scenes, {scannet_images} images")
        
        # Load 3D-FRONT
        front_dir = self.data_root / "3d_front"
        if front_dir.exists():
            front_scenes = len(list(front_dir.glob("*")))
            front_images = len(list(front_dir.rglob("*.jpg")))
            datasets_info["3d_front"]["scenes"] = front_scenes
            datasets_info["3d_front"]["samples"] = front_images
            datasets_info["total"]["scenes"] += front_scenes
            datasets_info["total"]["samples"] += front_images
            logger.info(f"   3D-FRONT: {front_scenes} scenes, {front_images} images")
        
        # Load Matterport3D
        matterport_dir = self.data_root / "matterport3d"
        if matterport_dir.exists():
            matterport_buildings = len(list(matterport_dir.glob("*")))
            matterport_images = len(list(matterport_dir.rglob("*.jpg")))
            datasets_info["matterport3d"]["scenes"] = matterport_buildings
            datasets_info["matterport3d"]["samples"] = matterport_images
            datasets_info["total"]["scenes"] += matterport_buildings
            datasets_info["total"]["samples"] += matterport_images
            logger.info(f"   Matterport3D: {matterport_buildings} buildings, {matterport_images} images")
        
        self.training_stats["total_samples"] = datasets_info["total"]["samples"]
        self.training_stats["datasets_used"] = [k for k in ["scannet", "3d_front", "matterport3d"] 
                                                if datasets_info[k]["samples"] > 0]
        
        logger.info(f"✅ Large datasets loaded!")
        logger.info(f"   Total Scenes: {datasets_info['total']['scenes']}")
        logger.info(f"   Total Samples: {datasets_info['total']['samples']}")
        
        return datasets_info
    
    def initialize_models(self):
        """Initialize student and teacher models."""
        logger.info("🤖 Initializing models...")
        
        # Initialize student model
        config = DistilledLLaVA3DConfig()
        self.student_model = DistilledLLaVA3D(config)
        self.student_model.to(self.device)
        
        # Initialize enhanced mock teacher
        self.teacher_model = EnhancedMockTeacher(device=self.device)
        
        # Initialize distillation loss
        self.distillation_loss = AdaptiveDistillationLoss(
            temperature=3.0,
            alpha=0.7,
            adaptation_rate=0.1
        )
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5
        )
        
        logger.info("✅ Models initialized successfully!")
        logger.info(f"   Student Model: {type(self.student_model).__name__}")
        logger.info(f"   Teacher Model: {type(self.teacher_model).__name__}")
        logger.info(f"   Device: {self.device}")
    
    def create_training_samples(self, datasets_info: Dict[str, any]) -> List[Dict]:
        """
        Create training samples from large datasets.
        
        Args:
            datasets_info: Information about loaded datasets
            
        Returns:
            List of training samples
        """
        logger.info("📝 Creating training samples from large datasets...")
        
        training_samples = []
        
        # Process ScanNet
        scannet_dir = self.data_root / "scannet"
        if scannet_dir.exists():
            for scene_dir in scannet_dir.glob("scene*"):
                if scene_dir.is_dir():
                    # Get images from scene
                    images = list(scene_dir.glob("*.jpg"))
                    annotations_file = scene_dir / "annotations.json"
                    
                    if annotations_file.exists():
                        with open(annotations_file, 'r') as f:
                            annotations = json.load(f)
                        
                        for img_path in images:
                            sample = {
                                "image_path": str(img_path),
                                "scene_id": scene_dir.name,
                                "annotations": annotations,
                                "dataset": "scannet"
                            }
                            training_samples.append(sample)
        
        # Process 3D-FRONT
        front_dir = self.data_root / "3d_front"
        if front_dir.exists():
            for scene_dir in front_dir.glob("*"):
                if scene_dir.is_dir():
                    # Get images from scene
                    images = list(scene_dir.glob("*.jpg"))
                    annotations_file = scene_dir / "annotations.json"
                    
                    if annotations_file.exists():
                        with open(annotations_file, 'r') as f:
                            annotations = json.load(f)
                        
                        for img_path in images:
                            sample = {
                                "image_path": str(img_path),
                                "scene_id": scene_dir.name,
                                "annotations": annotations,
                                "dataset": "3d_front"
                            }
                            training_samples.append(sample)
        
        # Process Matterport3D
        matterport_dir = self.data_root / "matterport3d"
        if matterport_dir.exists():
            for building_dir in matterport_dir.glob("*"):
                if building_dir.is_dir():
                    # Get images from building
                    images = list(building_dir.rglob("*.jpg"))
                    annotations_file = building_dir / "building_annotations.json"
                    
                    if annotations_file.exists():
                        with open(annotations_file, 'r') as f:
                            annotations = json.load(f)
                        
                        for img_path in images:
                            sample = {
                                "image_path": str(img_path),
                                "scene_id": building_dir.name,
                                "annotations": annotations,
                                "dataset": "matterport3d"
                            }
                            training_samples.append(sample)
        
        logger.info(f"✅ Created {len(training_samples)} training samples!")
        return training_samples
    
    def generate_questions_for_sample(self, sample: Dict) -> List[str]:
        """
        Generate questions for a training sample.
        
        Args:
            sample: Training sample
            
        Returns:
            List of questions
        """
        questions = []
        annotations = sample.get("annotations", {})
        dataset = sample.get("dataset", "unknown")
        
        # Basic scene questions
        if "room_type" in annotations:
            questions.append(f"What type of room is this?")
        elif "building_type" in annotations:
            questions.append(f"What type of building is this?")
        
        # Object detection questions
        if "objects" in annotations:
            questions.append(f"What objects can you see in this 3D scene?")
        if "furniture" in annotations:
            questions.append(f"What furniture is visible in this scene?")
        
        # Spatial reasoning questions
        if "spatial_relations" in annotations:
            for relation in annotations["spatial_relations"]:
                questions.append(f"How is the {relation['subject']} positioned relative to the {relation['object']}?")
        
        # 3D understanding questions
        questions.extend([
            "What is the depth structure of this scene?",
            "How are objects arranged in 3D space?",
            "What is the overall layout of this 3D scene?",
            "What should I be cautious about in this environment?"
        ])
        
        return questions
    
    def train_epoch(self, training_samples: List[Dict], epoch: int) -> float:
        """
        Train for one epoch.
        
        Args:
            training_samples: List of training samples
            epoch: Current epoch number
            
        Returns:
            Average loss for the epoch
        """
        logger.info(f"🎓 Training Epoch {epoch + 1}/{self.epochs}")
        
        self.student_model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Process samples in batches
        for i in range(0, len(training_samples), self.batch_size):
            batch_samples = training_samples[i:i + self.batch_size]
            
            batch_loss = 0.0
            valid_samples = 0
            
            for sample in batch_samples:
                try:
                    # Load image
                    image_path = sample["image_path"]
                    if not os.path.exists(image_path):
                        continue
                    
                    # Load and process image
                    image = Image.open(image_path).convert("RGB")
                    image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
                    image_tensor = image_tensor.unsqueeze(0).to(self.device)
                    
                    # Generate questions
                    questions = self.generate_questions_for_sample(sample)
                    if not questions:
                        continue
                    
                    # Use first question for this sample
                    question = questions[0]
                    
                    # Get teacher response
                    teacher_response = self.teacher_model.generate_response(question, image_tensor)
                    teacher_features = self.teacher_model.analyze_image_content(image_tensor)
                    
                    # Get student response
                    student_response = self.student_model.generate_response(question, image_tensor)
                    student_features = self.student_model.analyze_image_content(image_tensor)
                    
                    # Compute distillation loss
                    loss = self.distillation_loss.compute_loss(
                        student_response, teacher_response,
                        student_features, teacher_features
                    )
                    
                    batch_loss += loss.item()
                    valid_samples += 1
                    
                except Exception as e:
                    logger.warning(f"⚠️  Error processing sample {sample.get('image_path', 'unknown')}: {e}")
                    continue
            
            if valid_samples > 0:
                # Average loss for batch
                avg_batch_loss = batch_loss / valid_samples
                total_loss += avg_batch_loss
                num_batches += 1
                
                # Backward pass
                self.optimizer.zero_grad()
                loss_tensor = torch.tensor(avg_batch_loss, requires_grad=True)
                loss_tensor.backward()
                self.optimizer.step()
                
                if (i // self.batch_size + 1) % 10 == 0:
                    logger.info(f"   Batch {i // self.batch_size + 1}: Loss = {avg_batch_loss:.4f}")
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        logger.info(f"✅ Epoch {epoch + 1} completed! Average Loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def save_checkpoint(self, epoch: int, loss: float):
        """Save training checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
        
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": self.student_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss,
            "training_stats": self.training_stats
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
    
    def run_advanced_training(self):
        """Run advanced training with large datasets."""
        logger.info("🚀 Starting Advanced Training with Large Datasets")
        
        start_time = time.time()
        
        # Load large datasets
        datasets_info = self.load_large_datasets()
        
        # Initialize models
        self.initialize_models()
        
        # Create training samples
        training_samples = self.create_training_samples(datasets_info)
        
        if not training_samples:
            logger.error("❌ No training samples created!")
            return False
        
        logger.info(f"📊 Training on {len(training_samples)} samples")
        logger.info(f"   Datasets: {', '.join(self.training_stats['datasets_used'])}")
        
        # Training loop
        for epoch in range(self.epochs):
            epoch_start = time.time()
            
            # Train epoch
            avg_loss = self.train_epoch(training_samples, epoch)
            
            # Update statistics
            self.training_stats["epochs_completed"] = epoch + 1
            self.training_stats["total_loss"] = avg_loss
            
            if avg_loss < self.training_stats["best_loss"]:
                self.training_stats["best_loss"] = avg_loss
            
            # Save checkpoint every 3 epochs
            if (epoch + 1) % 3 == 0:
                self.save_checkpoint(epoch, avg_loss)
            
            epoch_time = time.time() - epoch_start
            logger.info(f"⏱️  Epoch {epoch + 1} completed in {epoch_time:.2f}s")
        
        # Save final checkpoint
        self.save_checkpoint(self.epochs - 1, self.training_stats["total_loss"])
        
        # Training completed
        total_time = time.time() - start_time
        self.training_stats["training_time"] = total_time
        
        logger.info("🎉 Advanced training completed!")
        logger.info(f"   Total Time: {total_time:.2f}s")
        logger.info(f"   Epochs: {self.training_stats['epochs_completed']}")
        logger.info(f"   Best Loss: {self.training_stats['best_loss']:.4f}")
        logger.info(f"   Datasets Used: {', '.join(self.training_stats['datasets_used'])}")
        
        # Save final results
        self.save_final_results()
        
        return True
    
    def save_final_results(self):
        """Save final training results."""
        results = {
            "training_completed": True,
            "training_stats": self.training_stats,
            "checkpoint_dir": str(self.checkpoint_dir),
            "datasets_used": self.training_stats["datasets_used"],
            "total_samples": self.training_stats["total_samples"],
            "final_loss": self.training_stats["total_loss"],
            "best_loss": self.training_stats["best_loss"],
            "training_time": self.training_stats["training_time"],
            "epochs_completed": self.training_stats["epochs_completed"]
        }
        
        results_path = self.checkpoint_dir / "advanced_training_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"📊 Final results saved: {results_path}")

def main():
    """Main function to run advanced training."""
    logger.info("🚀 Starting Advanced Training with Large Datasets")
    
    # Initialize training
    trainer = AdvancedTrainingWithLargeDatasets()
    
    # Run advanced training
    success = trainer.run_advanced_training()
    
    if success:
        logger.info("✅ Advanced training completed successfully!")
        logger.info("📊 Check results in checkpoints/advanced_training_results.json")
    else:
        logger.error("❌ Advanced training failed!")

if __name__ == "__main__":
    main()
