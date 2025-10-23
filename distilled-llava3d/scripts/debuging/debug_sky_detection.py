#!/usr/bin/env python3
"""Debug script to understand why sky detection is failing."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

def debug_sky_detection(image_path):
    """Debug sky detection step by step."""
    print(f"Debugging sky detection for: {image_path}")
    print("=" * 60)
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    pixel_values = transform(image).unsqueeze(0)  # Add batch dimension
    raw_pixels = pixel_values.squeeze(0)  # (3, 224, 224)
    
    h, w = raw_pixels.shape[1], raw_pixels.shape[2]
    print(f"Image shape: {raw_pixels.shape}")
    print(f"Height: {h}, Width: {w}")
    
    # Analyze top portion for sky (more aggressive - top half)
    top_portion = raw_pixels[:, :h//2, :]  # Top half
    print(f"Top portion shape: {top_portion.shape}")
    
    # Sky characteristics: bright, blue-tinted, low contrast
    brightness = torch.mean(top_portion).item()
    blue_dominance = (torch.mean(top_portion[2]) - torch.mean(top_portion[0]) - torch.mean(top_portion[1])).item()
    contrast = torch.std(top_portion).item()
    
    print(f"Top portion brightness: {brightness:.3f}")
    print(f"Blue dominance: {blue_dominance:.3f}")
    print(f"Contrast: {contrast:.3f}")
    
    # Additional sky features
    center_region = top_portion[:, h//4:3*h//8, w//4:3*w//4]  # Center of top portion
    center_brightness = torch.mean(center_region).item()
    print(f"Center brightness: {center_brightness:.3f}")
    
    # Sky often has gradient (brighter at top, darker at bottom)
    top_row = torch.mean(top_portion[:, :h//8, :]).item()  # Very top
    bottom_row = torch.mean(top_portion[:, 3*h//8:h//2, :]).item()  # Bottom of top portion
    gradient = top_row - bottom_row
    print(f"Top row brightness: {top_row:.3f}")
    print(f"Bottom row brightness: {bottom_row:.3f}")
    print(f"Gradient: {gradient:.3f}")
    
    # Check each condition
    print("\nSky detection conditions:")
    print(f"1. brightness > 0.3: {brightness > 0.3} ({brightness:.3f})")
    print(f"2. blue_dominance > -0.2: {blue_dominance > -0.2} ({blue_dominance:.3f})")
    print(f"3. contrast < 0.4: {contrast < 0.4} ({contrast:.3f})")
    print(f"4. center_brightness > brightness * 0.9: {center_brightness > brightness * 0.9} ({center_brightness:.3f} > {brightness * 0.9:.3f})")
    print(f"5. gradient > -0.1: {gradient > -0.1} ({gradient:.3f})")
    
    # Final result
    has_sky = (
        brightness > 0.3 and  # Reasonably bright
        blue_dominance > -0.2 and  # Not too red/green (more permissive)
        contrast < 0.4 and  # Low contrast (more permissive)
        center_brightness > brightness * 0.9 and  # Center is bright
        gradient > -0.1  # Not too much reverse gradient
    )
    
    print(f"\nFinal sky detection: {has_sky}")
    
    # Let's also check the RGB values in the top portion
    print(f"\nRGB values in top portion:")
    print(f"Red mean: {torch.mean(top_portion[0]).item():.3f}")
    print(f"Green mean: {torch.mean(top_portion[1]).item():.3f}")
    print(f"Blue mean: {torch.mean(top_portion[2]).item():.3f}")
    
    # Check if there's actually sky-like content
    print(f"\nSky-like content analysis:")
    print(f"Blue > Red: {torch.mean(top_portion[2]) > torch.mean(top_portion[0])}")
    print(f"Blue > Green: {torch.mean(top_portion[2]) > torch.mean(top_portion[1])}")
    print(f"Brightness > 0.5: {brightness > 0.5}")

if __name__ == "__main__":
    image_path = "/scratch/alasfour/llava-3d/LLaVA-3D/demo/my_images/IMG_001.png"
    debug_sky_detection(image_path)



