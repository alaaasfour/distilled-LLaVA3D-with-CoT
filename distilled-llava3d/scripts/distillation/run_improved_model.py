#!/usr/bin/env python3
"""Simple CLI for the improved distilled LLaVA-3D model."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchvision.transforms as transforms
from PIL import Image
import argparse
from pathlib import Path

# Import the improved model
from scripts.distillation.student_model import DistilledLLaVA3D, DistilledLLaVA3DConfig

class ImprovedModelCLI:
    """Simple CLI for the improved distilled LLaVA-3D model."""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.config = DistilledLLaVA3DConfig()
        self.model = DistilledLLaVA3D(self.config)
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        print("✅ Improved Distilled LLaVA-3D model loaded successfully!")
        print(f"🔧 Device: {self.device}")
        print(f"📊 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def load_image(self, image_path):
        """Load and preprocess image."""
        try:
            if image_path.startswith("http"):
                print(f"⚠️  Web image {image_path} - using placeholder")
                # Create a random image as placeholder
                pixel_values = torch.randn(1, 3, 224, 224).to(self.device)
                return pixel_values, True
            else:
                # Load local image
                image = Image.open(image_path).convert('RGB')
                pixel_values = self.transform(image).unsqueeze(0).to(self.device)
                return pixel_values, False
        except Exception as e:
            print(f"❌ Error loading image {image_path}: {e}")
            return None, False
    
    def generate_response(self, question, image_path):
        """Generate response for a question and image."""
        print(f"\n🖼️  Image: {image_path}")
        print(f"❓ Question: {question}")
        print("-" * 60)
        
        # Load image
        pixel_values, is_placeholder = self.load_image(image_path)
        if pixel_values is None:
            return "Error: Could not load image"
        
        try:
            # Generate response
            response = self.model.generate_response(question, pixel_values)
            
            # Get detailed analysis
            with torch.no_grad():
                features = self.model.analyze_image_content(pixel_values)
            
            print(f"🤖 Response: {response}")
            print(f"\n🔍 Analysis:")
            print(f"   - Indoor: {features['is_indoor']}")
            print(f"   - Outdoor: {features['is_outdoor']}")
            print(f"   - Has Sky: {features['has_sky']}")
            print(f"   - Has Person: {features['has_person']}")
            print(f"   - Has Buildings: {features['has_buildings']}")
            print(f"   - Has Natural Elements: {features['has_natural_elements']}")
            print(f"   - Outdoor Score: {features['outdoor_score']}")
            print(f"   - Indoor Score: {features['indoor_score']}")
            print(f"   - Confidence: {features['outdoor_confidence']:.3f}")
            
            return response
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="Improved Distilled LLaVA-3D CLI")
    parser.add_argument("--image-file", type=str, required=True, help="Path to image file")
    parser.add_argument("--query", type=str, required=True, help="Question to ask about the image")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Check if image file exists
    if not args.image_file.startswith("http") and not os.path.exists(args.image_file):
        print(f"❌ Error: Image file {args.image_file} not found!")
        return
    
    # Initialize CLI
    cli = ImprovedModelCLI(device=args.device)
    
    # Generate response
    response = cli.generate_response(args.query, args.image_file)
    
    print(f"\n✅ Response generated successfully!")

if __name__ == "__main__":
    main()

