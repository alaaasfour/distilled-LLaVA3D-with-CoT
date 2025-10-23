#!/usr/bin/env python3
"""Test script to compare old vs improved indoor/outdoor detection."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchvision.transforms as transforms
from PIL import Image
import argparse

# Import both models
from scripts.distillation.simple_student_model import DistilledLLaVA3D as OldModel, DistilledLLaVA3DConfig as OldConfig
from scripts.distillation.improved_student_model import DistilledLLaVA3D as NewModel, DistilledLLaVA3DConfig as NewConfig

def load_and_preprocess_image(image_path):
    """Load and preprocess image for testing."""
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Preprocess
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    pixel_values = transform(image).unsqueeze(0)  # Add batch dimension
    return pixel_values

def test_detection_methods(image_path, question):
    """Test both old and new detection methods."""
    print(f"Testing image: {image_path}")
    print(f"Question: {question}")
    print("=" * 60)
    
    # Load and preprocess image
    pixel_values = load_and_preprocess_image(image_path)
    
    # Test old model
    print("🔍 OLD DETECTION METHOD:")
    print("-" * 30)
    old_config = OldConfig()
    old_model = OldModel(old_config)
    old_model.eval()
    
    with torch.no_grad():
        old_features = old_model.analyze_image_content(pixel_values)
        old_response = old_model.generate_response(question, pixel_values)
    
    print(f"Indoor: {old_features['is_indoor']}")
    print(f"Outdoor: {old_features['is_outdoor']}")
    print(f"Has Sky: {old_features['has_sky']}")
    print(f"Brightness: {old_features['brightness']:.3f}")
    print(f"Contrast: {old_features['complexity']:.3f}")
    print(f"Response: {old_response}")
    print()
    
    # Test new model
    print("🚀 NEW IMPROVED DETECTION METHOD:")
    print("-" * 30)
    new_config = NewConfig()
    new_model = NewModel(new_config)
    new_model.eval()
    
    with torch.no_grad():
        new_features = new_model.analyze_image_content(pixel_values)
        new_response = new_model.generate_response(question, pixel_values)
    
    print(f"Indoor: {new_features['is_indoor']}")
    print(f"Outdoor: {new_features['is_outdoor']}")
    print(f"Has Sky: {new_features['has_sky']}")
    print(f"Has Horizon: {new_features['has_horizon']}")
    print(f"Has Natural Elements: {new_features['has_natural_elements']}")
    print(f"Brightness: {new_features['brightness']:.3f}")
    print(f"Contrast: {new_features['complexity']:.3f}")
    print(f"Outdoor Score: {new_features['outdoor_score']}")
    print(f"Indoor Score: {new_features['indoor_score']}")
    print(f"Confidence: {new_features['outdoor_confidence']:.3f}")
    print(f"Sky Brightness: {new_features['sky_brightness']:.3f}")
    print(f"Green Dominance: {new_features['green_dominance']:.3f}")
    print(f"Warm Lighting: {new_features['warm_lighting']:.3f}")
    print(f"Response: {new_response}")
    print()
    
    # Compare results
    print("📊 COMPARISON:")
    print("-" * 30)
    indoor_correct = "✅" if new_features['is_outdoor'] else "❌"
    outdoor_correct = "✅" if new_features['is_outdoor'] else "❌"
    
    print(f"Indoor/Outdoor Classification: {indoor_correct}")
    print(f"Old: {'Indoor' if old_features['is_indoor'] else 'Outdoor'}")
    print(f"New: {'Indoor' if new_features['is_indoor'] else 'Outdoor'}")
    print(f"Sky Detection: {'✅' if new_features['has_sky'] else '❌'}")
    print(f"Response Quality: {'✅' if len(new_response) > len(old_response) else '❌'}")
    print()

def main():
    parser = argparse.ArgumentParser(description="Test improved indoor/outdoor detection")
    parser.add_argument("--image", type=str, required=True, help="Path to test image")
    parser.add_argument("--question", type=str, default="What can you see in this image?", help="Question to ask")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: Image file {args.image} not found!")
        return
    
    test_detection_methods(args.image, args.question)

if __name__ == "__main__":
    main()



