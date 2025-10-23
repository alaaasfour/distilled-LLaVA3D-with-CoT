#!/usr/bin/env python3
"""Debug script to check structure detection."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchvision.transforms as transforms
from PIL import Image

def debug_structure_detection(image_path):
    """Debug structure detection step by step."""
    print(f"Debugging structure detection for: {image_path}")
    print("=" * 60)
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    pixel_values = transform(image).unsqueeze(0)  # Add batch dimension
    raw_pixels = pixel_values.squeeze(0)  # (3, 224, 224)
    
    # Edge and structure analysis
    edge_detection = torch.std(raw_pixels, dim=0)
    structure_score = torch.mean(edge_detection).item()
    
    print(f"Structure score: {structure_score:.3f}")
    print(f"Structure score > 0.15: {structure_score > 0.15}")
    print(f"Structure score > 0.25: {structure_score > 0.25}")
    
    # Let's also check the actual edge detection values
    print(f"Edge detection shape: {edge_detection.shape}")
    print(f"Edge detection min: {torch.min(edge_detection).item():.3f}")
    print(f"Edge detection max: {torch.max(edge_detection).item():.3f}")
    print(f"Edge detection mean: {torch.mean(edge_detection).item():.3f}")
    print(f"Edge detection std: {torch.std(edge_detection).item():.3f}")

if __name__ == "__main__":
    image_path = "/scratch/alasfour/llava-3d/LLaVA-3D/demo/my_images/IMG_001.png"
    debug_structure_detection(image_path)



