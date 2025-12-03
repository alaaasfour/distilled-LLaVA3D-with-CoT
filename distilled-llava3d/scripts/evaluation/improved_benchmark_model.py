#!/usr/bin/env python3
"""
Benchmark Improved Model
=======================

This script loads the improved checkpoint and runs benchmarks.
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
from benchmark_3d_vlm_tasks import Standard3DVLMBenchmark

def load_improved_model(checkpoint_path: str = None):
    """
    Load the improved model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint (defaults to best checkpoint)
    """
    if checkpoint_path is None:
        # Find best checkpoint (try fixed model first, then improved)
        checkpoint_dir = Path("/home/alasfour/scratch/distilled-llava3d/checkpoints")
        best_checkpoint = checkpoint_dir / "fixed_model_best.pt"
        if not best_checkpoint.exists():
            best_checkpoint = checkpoint_dir / "improved_model_best.pt"
        if not best_checkpoint.exists():
            # Try epoch checkpoints
            best_checkpoint = checkpoint_dir / "fixed_model_epoch_40.pt"
        if not best_checkpoint.exists():
            best_checkpoint = checkpoint_dir / "improved_model_epoch_50.pt"
        checkpoint_path = str(best_checkpoint)
    
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    
    # Initialize model
    config = DistilledLLaVA3DConfig()
    model = DistilledLLaVA3D(config)
    
    # Load checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"   Loss: {checkpoint.get('loss', 'unknown'):.6f}")
    else:
        print(f"⚠️  Checkpoint not found: {checkpoint_path}")
        print("   Using untrained model")
    
    model.eval()
    return model

def run_benchmarks_with_improved_model():
    """Run all benchmarks with the improved model."""
    print("🚀 BENCHMARKING IMPROVED MODEL")
    print("=" * 60)
    
    # Load improved model
    model = load_improved_model()
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"   Device: {device}")
    
    results = {}
    
    # 1. Basic Benchmark
    print("\n📋 1. Running Basic Benchmark...")
    print("-" * 40)
    try:
        benchmark = BenchmarkFramework(model, device=device)
        basic_results = benchmark.run_comprehensive_benchmark()
        results['basic_benchmark'] = basic_results
        print("✅ Basic benchmark completed")
    except Exception as e:
        print(f"❌ Basic benchmark failed: {str(e)}")
        import traceback
        traceback.print_exc()
        results['basic_benchmark'] = {'error': str(e)}
    
    # 2. Standard 3D VLM Benchmark
    print("\n📋 2. Running Standard 3D VLM Benchmark...")
    print("-" * 40)
    try:
        vlm_benchmark = Standard3DVLMBenchmark(model, device=device)
        vlm_results = vlm_benchmark.run_standard_benchmarks()
        results['3d_vlm_benchmark'] = vlm_results
        print("✅ 3D VLM benchmark completed")
    except Exception as e:
        print(f"❌ 3D VLM benchmark failed: {str(e)}")
        import traceback
        traceback.print_exc()
        results['3d_vlm_benchmark'] = {'error': str(e)}
    
    # Save results
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"improved_model_benchmark_results_{timestamp}.json"
    
    save_data = {
        'timestamp': timestamp,
        'checkpoint_used': 'improved_model_best.pt or improved_model_epoch_50.pt',
        'results': results,
        'model_info': {
            'model_type': 'Distilled LLaVA-3D Student (Improved)',
            'parameters': '~3B',
            'device': device
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 BENCHMARK SUMMARY")
    print("=" * 60)
    
    if 'basic_benchmark' in results and 'error' not in results['basic_benchmark']:
        basic = results['basic_benchmark']
        print("\n📋 Basic Benchmark Results:")
        for task_name, task_result in basic.items():
            if isinstance(task_result, dict) and 'accuracy' in task_result:
                acc = task_result['accuracy']
                print(f"   {task_name}: {acc:.2%}")
    
    if '3d_vlm_benchmark' in results and 'error' not in results['3d_vlm_benchmark']:
        vlm = results['3d_vlm_benchmark']
        print("\n📋 3D VLM Benchmark Results:")
        for task_name, task_result in vlm.items():
            if isinstance(task_result, dict) and 'accuracy' in task_result:
                acc = task_result['accuracy']
                print(f"   {task_name}: {acc:.2%}")
    
    return results

if __name__ == "__main__":
    run_benchmarks_with_improved_model()