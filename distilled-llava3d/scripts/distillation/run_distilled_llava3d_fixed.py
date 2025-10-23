#!/usr/bin/env python3
"""Command-line interface for distilled LLaVA-3D model."""

import argparse
import torch
from PIL import Image
import torchvision.transforms as transforms
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.distillation.simple_student_model import DistilledLLaVA3D, DistilledLLaVA3DConfig

class DistilledLLaVA3DCLI:
    """Command-line interface for distilled LLaVA-3D."""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.model = None
        self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load the distilled model."""
        print(f"📚 Loading distilled model from {model_path}...")
        
        # Create model configuration
        config = DistilledLLaVA3DConfig()
        self.model = DistilledLLaVA3D(config).to(self.device)
        
        # Load checkpoint
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint, strict=False)
            print("📊 Model loaded from checkpoint")
        except Exception as e:
            print(f"⚠️  Error loading checkpoint: {e}")
            print("Using randomly initialized model")
        
        self.model.eval()
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"🔢 Parameters: {total_params:,}")
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess image for model input."""
        # Load image
        if image_path.startswith("http"):
            # For URL images, create a placeholder
            image = Image.new('RGB', (224, 224), color=(100, 150, 200))
        else:
            image = Image.open(image_path).convert('RGB')
        
        # Preprocess
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        pixel_values = transform(image).unsqueeze(0)  # Add batch dimension
        return pixel_values.to(self.device)
    
    def generate_response(self, question: str, image_path: str) -> str:
        """Generate response for given question and image."""
        # Preprocess image
        pixel_values = self.preprocess_image(image_path)
        
        # Use the model's generate_response method
        response = self.model.generate_response(question, pixel_values)
        return response
    
    def run_interactive(self):
        """Run interactive mode."""
        print("🤖 Distilled LLaVA-3D Interactive Mode")
        print("Type 'quit' to exit")
        print("-" * 50)
        
        while True:
            try:
                image_path = input("🖼️  Image path: ").strip()
                if image_path.lower() == 'quit':
                    break
                
                question = input("❓ Question: ").strip()
                if question.lower() == 'quit':
                    break
                
                response = self.generate_response(question, image_path)
                print(f"🤖 Response: {response}")
                print("-" * 50)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def run_single(self, image_path: str, question: str):
        """Run single inference."""
        print(f"🖼️  Image: {image_path}")
        print(f"❓ Query: {question}")
        print("-" * 50)
        
        response = self.generate_response(question, image_path)
        print(f"🤖 Distilled LLaVA-3D Response:")
        print(f"   {response}")

def main():
    parser = argparse.ArgumentParser(description="Distilled LLaVA-3D CLI")
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--image-file", type=str,
                       help="Path to image file")
    parser.add_argument("--query", type=str,
                       help="Question to ask about the image")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive mode")
    
    args = parser.parse_args()
    
    # Create CLI
    cli = DistilledLLaVA3DCLI(args.model_path, args.device)
    
    if args.interactive:
        cli.run_interactive()
    elif args.image_file and args.query:
        cli.run_single(args.image_file, args.query)
    else:
        print("❌ Please provide both --image-file and --query, or use --interactive")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
