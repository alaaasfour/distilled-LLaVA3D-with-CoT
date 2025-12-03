#!/usr/bin/env python3
"""
Benchmark Fixed Model (Simplified)
==================================

Simplified benchmark to avoid memory issues.
"""

import os
import sys
import json
import torch
from pathlib import Path

# Add project paths
sys.path.append('/home/alasfour/scratch/distilled-llava3d')

from scripts.distillation.student_model import DistilledLLaVA3D, DistilledLLaVA3DConfig
from benchmark_framework import BenchmarkFramework

def load_fixed_model():
    """Load the fixed model from checkpoint."""
    checkpoint_dir = Path("/home/alasfour/scratch/distilled-llava3d/checkpoints")
    checkpoint_path = checkpoint_dir / "fixed_model_best.pt"
    
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    
    # Initialize model
    config = DistilledLLaVA3DConfig()
    model = DistilledLLaVA3D(config)
    
    # Load checkpoint
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"   Val Loss: {checkpoint.get('training_stats', {}).get('best_val_loss', 'unknown')}")
    else:
        print(f"⚠️  Checkpoint not found: {checkpoint_path}")
    
    model.eval()
    return model

def run_simple_benchmark():
    """Run simplified benchmark."""
    print("🚀 BENCHMARKING FIXED MODEL")
    print("=" * 60)
    
    # Load model
    model = load_fixed_model()
    
    # Use CPU to avoid memory issues
    device = "cpu"
    model = model.to(device)
    print(f"   Device: {device}")
    
    # Run basic benchmark only
    print("\n📋 Running Basic Benchmark...")
    print("-" * 40)
    
    try:
        benchmark = BenchmarkFramework(model, device=device)
        results = benchmark.run_comprehensive_benchmark()
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 BENCHMARK SUMMARY")
        print("=" * 60)
        
        overall_correct = 0
        overall_total = 0
        
        for task_name, task_result in results.items():
            if isinstance(task_result, dict) and 'accuracy' in task_result:
                acc = task_result['accuracy']
                total = task_result.get('total_questions', task_result.get('total_tasks', task_result.get('total_scenes', 0)))
                correct = task_result.get('correct_answers', task_result.get('correct_tasks', task_result.get('correct_classifications', 0)))
                
                print(f"   {task_name}: {acc:.2%} ({correct}/{total})")
                overall_correct += correct
                overall_total += total
        
        overall_acc = overall_correct / overall_total if overall_total > 0 else 0.0
        print(f"\n🎯 Overall Accuracy: {overall_acc:.2%} ({overall_correct}/{overall_total})")
        
        # Save results
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"fixed_model_benchmark_{timestamp}.json"
        
        save_data = {
            'timestamp': timestamp,
            'checkpoint': 'fixed_model_best.pt',
            'overall_accuracy': overall_acc,
            'results': results
        }
        
        with open(results_file, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {results_file}")
        
        return results
        
    except Exception as e:
        print(f"❌ Benchmark failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    run_simple_benchmark()