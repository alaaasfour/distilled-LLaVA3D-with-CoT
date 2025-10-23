#!/usr/bin/env python3
"""Debug script to check all detection conditions."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchvision.transforms as transforms
from PIL import Image

def debug_all_conditions(image_path):
    """Debug all detection conditions."""
    print(f"Debugging all conditions for: {image_path}")
    print("=" * 60)
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    pixel_values = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Import the improved model
    from scripts.distillation.improved_student_model import DistilledLLaVA3D, DistilledLLaVA3DConfig
    
    config = DistilledLLaVA3DConfig()
    model = DistilledLLaVA3D(config)
    model.eval()
    
    with torch.no_grad():
        features = model.analyze_image_content(pixel_values)
    
    print("Detection results:")
    print(f"has_person: {features['has_person']}")
    print(f"is_outdoor: {features['is_outdoor']}")
    print(f"has_buildings: {features['has_buildings']}")
    print(f"has_sky: {features['has_sky']}")
    print(f"has_natural_elements: {features['has_natural_elements']}")
    print(f"outdoor_score: {features['outdoor_score']}")
    print(f"indoor_score: {features['indoor_score']}")
    
    print("\nSafety response conditions:")
    print(f"is_outdoor and has_buildings and has_person: {features['is_outdoor'] and features['has_buildings'] and features['has_person']}")
    print(f"is_outdoor and has_buildings: {features['is_outdoor'] and features['has_buildings']}")
    print(f"has_person: {features['has_person']}")
    
    print("\nSpatial response conditions:")
    print(f"has_foreground: {features['has_foreground']}")
    print(f"has_background: {features['has_background']}")
    print(f"has_foreground and has_background: {features['has_foreground'] and features['has_background']}")
    print(f"has_person and is_outdoor: {features['has_person'] and features['is_outdoor']}")

if __name__ == "__main__":
    image_path = "/scratch/alasfour/llava-3d/LLaVA-3D/demo/my_images/IMG_001.png"
    debug_all_conditions(image_path)



