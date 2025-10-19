#!/usr/bin/env python3
"""
Evaluation script for distilled LLaVA-3D model.
"""

import torch
import json
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.distillation.student_model import DistilledLLaVA3D, DistilledLLaVA3DConfig

def evaluate_model(model_path, test_data):
    """Evaluate the distilled model."""
    # Load model
    config = DistilledLLaVA3DConfig()
    model = DistilledLLaVA3D(config)
    
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Run evaluation
    with torch.no_grad():
        for sample in test_data:
            # Process sample
            input_ids = sample['input_ids'].unsqueeze(0)
            attention_mask = sample['attention_mask'].unsqueeze(0)
            pixel_values = sample['pixel_values'].unsqueeze(0)
            
            # Get model output
            outputs = model(input_ids, attention_mask, pixel_values)
            
            # Process results
            print(f"Input: {sample['conversation']}")
            print(f"Output: {outputs.logits.shape}")
            print("-" * 50)

if __name__ == "__main__":
    print("Evaluation script ready!")