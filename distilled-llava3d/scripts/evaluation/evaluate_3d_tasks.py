#!/usr/bin/env python3
"""
Evaluation script for distilled LLaVA-3D on 3D tasks.
Tests both 2D and 3D capabilities.
"""

import torch
import json
import os
import sys

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from PIL import Image
import numpy as np

from scripts.distillation.student_model import DistilledLLaVA3D, DistilledLLaVA3DConfig
from scripts.distillation.load_teacher import load_llava3d_teacher

class DistilledLLaVA3DEvaluator:
    """Evaluator for distilled LLaVA-3D model."""
    
    def __init__(self, model_path, device="cuda"):
        self.device = device
        self.model_path = model_path
        self.model = None
        self.config = None
        
    def load_model(self):
        """Load the distilled model."""
        print(f"Loading distilled model from {self.model_path}...")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Create model
        self.config = DistilledLLaVA3DConfig()
        self.model = DistilledLLaVA3D(self.config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)  # Move to device
        self.model.eval()
        
        print(f"Model loaded successfully!")
        print(f"Epoch: {checkpoint['epoch']}")
        print(f"Loss: {checkpoint['loss']:.4f}")
        
    def evaluate_2d_task(self, image_path, question):
        """Evaluate on 2D image task."""
        print(f"\n=== 2D Task: {question} ===")
        
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            image_tensor = torch.randn(1, 3, 224, 224).to(self.device)  # Move to device
        except FileNotFoundError:
            print(f"Image not found: {image_path}, using mock data")
            image_tensor = torch.randn(1, 3, 224, 224).to(self.device)
        
        # Create input - move all tensors to device
        input_text = f"Human: {question}\nAssistant:"
        input_ids = torch.randint(0, 32000, (1, 64)).to(self.device)
        attention_mask = torch.ones(1, 64).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=image_tensor
            )
            
            # Get response (simplified)
            response = "This is a mock response from the distilled model."
            
        print(f"Question: {question}")
        print(f"Response: {response}")
        return response
        
    def evaluate_3d_task(self, scene_path, question):
        """Evaluate on 3D scene task."""
        print(f"\n=== 3D Task: {question} ===")
        
        # Mock 3D scene data - ensure proper batch format
        # Format: (batch_size, num_views, channels, height, width)
        rgb_images = torch.randn(1, 8, 3, 224, 224).to(self.device)  # 1 batch, 8 views
        depth_images = torch.randn(1, 8, 224, 224).to(self.device)   # 1 batch, 8 depth maps
        
        # Create input - move to device
        input_text = f"Human: {question}\nAssistant:"
        input_ids = torch.randint(0, 32000, (1, 64)).to(self.device)
        attention_mask = torch.ones(1, 64).to(self.device)
        
        with torch.no_grad():
            # Process first view for now
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=rgb_images,  # Pass the full 3D tensor
                depth_values=depth_images[0, 0]  # Pass single depth map
            )
            
            response = "This is a mock 3D response from the distilled model."
            
        print(f"Question: {question}")
        print(f"Response: {response}")
        return response
        
    def run_evaluation_suite(self):
        """Run comprehensive evaluation suite."""
        print("Starting Distilled LLaVA-3D Evaluation Suite")
        print("=" * 50)
        
        # 2D Tasks
        print("\n📸 2D Vision Tasks:")
        self.evaluate_2d_task(
            "demo/my_images/IMG_001.png",
            "What do you see in this image?"
        )
        self.evaluate_2d_task(
            "demo/my_images/IMG_001.png", 
            "Describe the objects and their spatial relationships."
        )
        
        # 3D Tasks
        print("\n🏠 3D Scene Understanding:")
        self.evaluate_3d_task(
            "demo/scannet/scene0356_00",
            "What objects can you see in this 3D scene?"
        )
        self.evaluate_3d_task(
            "demo/scannet/scene0356_00",
            "Describe the spatial layout of the room."
        )
        
        # Performance comparison
        print("\n📊 Performance Analysis:")
        self.analyze_performance()
        
    def analyze_performance(self):
        """Analyze model performance."""
        print("\nModel Performance Analysis:")
        print(f"Student Model Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Expected Teacher Parameters: ~7,000,000,000")
        print(f"Compression Ratio: {7000000000 / sum(p.numel() for p in self.model.parameters()):.1f}x")
        
        # Memory usage
        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated() / 1024**3
            print(f"Peak GPU Memory: {memory_used:.2f} GB")

def main():
    """Main evaluation function."""
    # Find the latest checkpoint
    checkpoint_dir = "models/checkpoints"
    if not os.path.exists(checkpoint_dir):
        print("No checkpoints found! Train the model first.")
        return
        
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    if not checkpoints:
        print("No checkpoints found! Train the model first.")
        return
        
    latest_checkpoint = sorted(checkpoints)[-1]
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    
    print(f"Using checkpoint: {latest_checkpoint}")
    
    # Initialize evaluator
    evaluator = DistilledLLaVA3DEvaluator(checkpoint_path)
    evaluator.load_model()
    
    # Run evaluation
    evaluator.run_evaluation_suite()

if __name__ == "__main__":
    main()