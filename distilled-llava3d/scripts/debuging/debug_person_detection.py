#!/usr/bin/env python3
"""Debug script to understand person detection."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchvision.transforms as transforms
from PIL import Image

def debug_person_detection(image_path):
    """Debug person detection step by step."""
    print(f"Debugging person detection for: {image_path}")
    print("=" * 60)
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    pixel_values = transform(image).unsqueeze(0)  # Add batch dimension
    raw_pixels = pixel_values.squeeze(0)  # (3, 224, 224)
    
    # Calculate basic image statistics
    mean_rgb = torch.mean(raw_pixels, dim=(1, 2))  # (3,)
    std_rgb = torch.std(raw_pixels, dim=(1, 2))    # (3,)
    
    # Get max and min values per channel
    max_rgb = torch.tensor([torch.max(raw_pixels[0]).item(), torch.max(raw_pixels[1]).item(), torch.max(raw_pixels[2]).item()])
    min_rgb = torch.tensor([torch.min(raw_pixels[0]).item(), torch.min(raw_pixels[1]).item(), torch.min(raw_pixels[2]).item()])
    
    # Basic image properties
    brightness = torch.mean(mean_rgb).item()
    contrast = torch.mean(std_rgb).item()
    color_variance = torch.var(mean_rgb).item()
    
    # Edge and structure analysis
    edge_detection = torch.std(raw_pixels, dim=0)
    structure_score = torch.mean(edge_detection).item()
    
    print(f"RGB means: R={mean_rgb[0]:.3f}, G={mean_rgb[1]:.3f}, B={mean_rgb[2]:.3f}")
    print(f"Brightness: {brightness:.3f}")
    print(f"Contrast: {contrast:.3f}")
    print(f"Structure score: {structure_score:.3f}")
    
    # Detect person (skin tones and human-like shapes)
    skin_tone_range = (mean_rgb[0] > 0.4) & (mean_rgb[0] < 0.8) & \
                    (mean_rgb[1] > 0.3) & (mean_rgb[1] < 0.7) & \
                    (mean_rgb[2] > 0.2) & (mean_rgb[2] < 0.6)
    
    print(f"\nSkin tone detection:")
    print(f"R > 0.4 and R < 0.8: {mean_rgb[0] > 0.4 and mean_rgb[0] < 0.8} ({mean_rgb[0]:.3f})")
    print(f"G > 0.3 and G < 0.7: {mean_rgb[1] > 0.3 and mean_rgb[1] < 0.7} ({mean_rgb[1]:.3f})")
    print(f"B > 0.2 and B < 0.6: {mean_rgb[2] > 0.2 and mean_rgb[2] < 0.6} ({mean_rgb[2]:.3f})")
    print(f"Skin tone range: {skin_tone_range}")
    print(f"Structure score > 0.2: {structure_score > 0.2}")
    
    has_person = skin_tone_range.item() and structure_score > 0.2
    print(f"Final person detection: {has_person}")

if __name__ == "__main__":
    image_path = "/scratch/alasfour/llava-3d/LLaVA-3D/demo/my_images/IMG_001.png"
    debug_person_detection(image_path)



