#!/usr/bin/env python3
"""Generate mock teacher responses for RGB-D question prompts.

This creates realistic teacher responses without loading the full LLaVA-3D model.
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List

def generate_mock_teacher_response(question: str, image_path: str) -> str:
    """Generate a mock teacher response based on the question and image path."""
    
    question_lower = question.lower()
    
    # Mock responses based on question type and image content
    if "what can you see" in question_lower or "what do you see" in question_lower:
        if "IMG_001" in image_path:
            return "I can see a person standing in what appears to be an outdoor environment. There are buildings or structures in the background, and the scene has good lighting with visible sky. The person seems to be positioned in the foreground with various objects and elements around them."
        else:
            return "I can see a beautiful landscape with mountains, trees, and a clear sky. There appears to be a path or road leading through the scene, with natural elements like rocks and vegetation. The lighting suggests it's either early morning or late afternoon, creating a peaceful and scenic view."
    
    elif "cautious" in question_lower or "danger" in question_lower or "safety" in question_lower:
        if "IMG_001" in image_path:
            return "You should be cautious about uneven surfaces, potential obstacles on the ground, and any structural elements that might not be stable. Watch out for changes in elevation, loose materials, and ensure you have proper footing when moving around this area."
        else:
            return "Be cautious about the terrain conditions, especially if there are steep slopes or uneven surfaces. Watch for loose rocks, slippery areas, and changes in weather that could affect visibility and safety. Stay on designated paths if available."
    
    elif "spatial" in question_lower or "relationship" in question_lower:
        if "IMG_001" in image_path:
            return "The spatial relationships show a clear foreground-background structure. The person is positioned in the foreground, with buildings and structures creating depth in the background. The scene has good depth perception with objects at various distances from the viewer."
        else:
            return "The spatial layout shows a natural landscape with elements arranged at different depths. There's a clear sense of perspective with foreground elements, middle ground features, and distant background elements creating a three-dimensional composition."
    
    elif "objects" in question_lower or "what objects" in question_lower:
        if "IMG_001" in image_path:
            return "I can identify a person, buildings or architectural structures, various objects in the environment, and natural elements like sky and possibly vegetation. The scene contains both human-made and natural objects arranged in a spatial layout."
        else:
            return "The image contains natural objects like mountains, trees, rocks, and vegetation. There may also be man-made elements such as paths, structures, or other features integrated into the landscape."
    
    elif "describe" in question_lower:
        if "IMG_001" in image_path:
            return "This image shows a person in an outdoor setting with architectural elements in the background. The composition has good balance with the subject in the foreground and supporting elements creating depth and context. The lighting appears natural and the scene has a realistic, everyday quality."
        else:
            return "This is a scenic landscape image featuring natural elements like mountains, trees, and sky. The composition follows natural perspective with foreground, middle ground, and background elements. The lighting and colors suggest a peaceful, natural environment."
    
    else:
        # Default response
        if "IMG_001" in image_path:
            return "This image shows a person in an outdoor environment with various architectural and natural elements. The scene has good visual composition with clear spatial relationships between different objects and features."
        else:
            return "This is a landscape image with natural elements arranged in a three-dimensional space. The scene shows good depth and perspective with various objects and features visible at different distances."

def main():
    # Load input manifest
    with open("input_manifest.json", "r") as f:
        input_manifest = json.load(f)
    
    # Generate teacher responses
    augmented_samples = []
    for sample in input_manifest:
        augmented = dict(sample)
        augmented["answer"] = generate_mock_teacher_response(
            sample["question"], 
            sample["image_path"]
        )
        augmented_samples.append(augmented)
    
    # Save output manifest
    with open("teacher_responses.json", "w") as f:
        json.dump(augmented_samples, f, indent=2)
    
    print(f"Generated {len(augmented_samples)} teacher responses")
    print("Sample responses:")
    for i, sample in enumerate(augmented_samples[:2]):
        print(f"\nSample {i+1}:")
        print(f"Question: {sample['question']}")
        print(f"Answer: {sample['answer']}")

if __name__ == "__main__":
    main()
