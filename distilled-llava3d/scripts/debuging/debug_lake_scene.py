#!/usr/bin/env python3
"""Debug script to understand why the lake scene is misclassified."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchvision.transforms as transforms
from PIL import Image

def debug_lake_scene():
    """Debug the lake scene detection."""
    print("🔍 Debugging Lake Scene Detection")
    print("=" * 50)
    
    # Load the lake image
    image_path = "/scratch/alasfour/llava-3d/LLaVA-3D/demo/my_images/LLaVA3D-view.jpg"
    image = Image.open(image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    pixel_values = transform(image).unsqueeze(0)
    raw_pixels = pixel_values.squeeze(0)  # (3, 224, 224)
    
    print(f"Image shape: {raw_pixels.shape}")
    
    # Basic image statistics
    mean_rgb = torch.mean(raw_pixels, dim=(1, 2))  # (3,)
    std_rgb = torch.std(raw_pixels, dim=(1, 2))    # (3,)
    
    print(f"RGB means: R={mean_rgb[0]:.3f}, G={mean_rgb[1]:.3f}, B={mean_rgb[2]:.3f}")
    print(f"RGB stds: R={std_rgb[0]:.3f}, G={std_rgb[1]:.3f}, B={std_rgb[2]:.3f}")
    
    # Basic properties
    brightness = torch.mean(mean_rgb).item()
    contrast = torch.mean(std_rgb).item()
    color_variance = torch.var(mean_rgb).item()
    
    print(f"Brightness: {brightness:.3f}")
    print(f"Contrast: {contrast:.3f}")
    print(f"Color variance: {color_variance:.3f}")
    
    # Edge and structure analysis
    edge_detection = torch.std(raw_pixels, dim=0)
    structure_score = torch.mean(edge_detection).item()
    
    print(f"Structure score: {structure_score:.3f}")
    print(f"Structure score > 0.15: {structure_score > 0.15}")
    
    # Person detection analysis
    skin_tone_range = (
        (mean_rgb[0] > 0.3) & (mean_rgb[0] < 0.9) &  # More flexible red range
        (mean_rgb[1] > 0.25) & (mean_rgb[1] < 0.8) &  # More flexible green range
        (mean_rgb[2] > 0.15) & (mean_rgb[2] < 0.7)    # More flexible blue range
    )
    
    human_like_patterns = (
        structure_score > 0.05 or  # Some structure present
        contrast > 0.15 or  # Reasonable contrast
        (mean_rgb[0] + mean_rgb[1] + mean_rgb[2]) / 3 > 0.2  # Not too dark overall
    )
    
    print(f"\nPerson Detection Analysis:")
    print(f"Skin tone range: {skin_tone_range}")
    print(f"Human-like patterns: {human_like_patterns}")
    print(f"Final person detection: {skin_tone_range.item() or human_like_patterns}")
    
    # Building detection analysis
    has_buildings = structure_score > 0.05 or (True and contrast > 0.2)  # is_outdoor=True, contrast > 0.2
    print(f"\nBuilding Detection Analysis:")
    print(f"Structure score > 0.05: {structure_score > 0.05}")
    print(f"Outdoor and contrast > 0.2: {True and contrast > 0.2}")
    print(f"Final building detection: {has_buildings}")
    
    # Sky detection analysis
    h, w = raw_pixels.shape[1], raw_pixels.shape[2]
    top_portion = raw_pixels[:, :h//2, :]  # Top half
    
    brightness_top = torch.mean(top_portion).item()
    blue_dominance = (torch.mean(top_portion[2]) - torch.mean(top_portion[0]) - torch.mean(top_portion[1])).item()
    contrast_top = torch.std(top_portion).item()
    
    print(f"\nSky Detection Analysis:")
    print(f"Top portion brightness: {brightness_top:.3f}")
    print(f"Blue dominance: {blue_dominance:.3f}")
    print(f"Top portion contrast: {contrast_top:.3f}")
    
    # Check sky conditions
    has_sky = (
        brightness_top > 0.3 and  # Reasonably bright
        (blue_dominance > -0.5 or  # Blue dominant OR
         (torch.mean(top_portion[2]) > torch.mean(top_portion[0]) and  # Blue > Red AND
          torch.mean(top_portion[2]) > torch.mean(top_portion[1]))) and  # Blue > Green
        contrast_top < 0.4 and  # Low contrast (more permissive)
        True and  # center_brightness > brightness * 0.9 (simplified)
        True  # gradient > -0.1 (simplified)
    )
    
    print(f"Sky detection conditions:")
    print(f"  brightness > 0.3: {brightness_top > 0.3}")
    print(f"  blue_dominance > -0.5: {blue_dominance > -0.5}")
    print(f"  blue > red: {torch.mean(top_portion[2]) > torch.mean(top_portion[0])}")
    print(f"  blue > green: {torch.mean(top_portion[2]) > torch.mean(top_portion[1])}")
    print(f"  contrast < 0.4: {contrast_top < 0.4}")
    print(f"Final sky detection: {has_sky}")

if __name__ == "__main__":
    debug_lake_scene()

