#!/usr/bin/env python3
"""Run all benchmarks and assessments for distilled LLaVA-3D."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from datetime import datetime
import json

def run_all_benchmarks():
    """Run all benchmarks and assessments."""
    print("🚀 COMPREHENSIVE BENCHMARKING SUITE")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    start_time = time.time()
    results = {}
    
    # 1. Basic Benchmark
    print("\n📋 1. Running Basic Benchmark...")
    print("-" * 40)
    try:
        from benchmark_framework import run_benchmark
        basic_results = run_benchmark()
        results['basic_benchmark'] = basic_results
        print("✅ Basic benchmark completed")
    except Exception as e:
        print(f"❌ Basic benchmark failed: {str(e)}")
        results['basic_benchmark'] = {'error': str(e)}
    
    # 2. Standard 3D VLM Benchmark
    print("\n📋 2. Running Standard 3D VLM Benchmark...")
    print("-" * 40)
    try:
        from benchmark_3d_vlm_tasks import run_3d_vlm_benchmark
        vlm_results = run_3d_vlm_benchmark()
        results['3d_vlm_benchmark'] = vlm_results
        print("✅ 3D VLM benchmark completed")
    except Exception as e:
        print(f"❌ 3D VLM benchmark failed: {str(e)}")
        results['3d_vlm_benchmark'] = {'error': str(e)}
    
    # 3. Paper-Worthiness Assessment
    print("\n📋 3. Running Paper-Worthiness Assessment...")
    print("-" * 40)
    try:
        from paper_worthiness_assessment import run_paper_worthiness_assessment
        paper_results = run_paper_worthiness_assessment()
        results['paper_worthiness'] = paper_results
        print("✅ Paper-worthiness assessment completed")
    except Exception as e:
        print(f"❌ Paper-worthiness assessment failed: {str(e)}")
        results['paper_worthiness'] = {'error': str(e)}
    
    total_time = time.time() - start_time
    
    # Generate final report
    print("\n" + "=" * 60)
    print("📊 FINAL BENCHMARKING REPORT")
    print("=" * 60)
    print(f"Total Time: {total_time:.2f}s")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Overall assessment
    if 'paper_worthiness' in results and 'error' not in results['paper_worthiness']:
        paper_score = results['paper_worthiness'].get('overall_score', 0)
        print(f"\n🎯 Overall Paper-Worthiness Score: {paper_score:.1f}/10")
        
        if paper_score >= 8.0:
            print("🏆 EXCELLENT: Highly paper-worthy!")
            print("   - Ready for top-tier venue submission")
            print("   - Strong contribution to the field")
        elif paper_score >= 7.0:
            print("✅ GOOD: Paper-worthy!")
            print("   - Suitable for good venue submission")
            print("   - Solid contribution to the field")
        elif paper_score >= 6.0:
            print("⚠️  MODERATE: Needs improvement")
            print("   - May be suitable for workshop or poster")
            print("   - Some contribution to the field")
        else:
            print("❌ NEEDS WORK: Not ready for publication")
            print("   - Significant improvements required")
            print("   - Limited contribution to the field")
    
    # Save comprehensive results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"comprehensive_benchmark_results_{timestamp}.json"
    
    save_data = {
        'timestamp': timestamp,
        'total_time': total_time,
        'results': results,
        'model_info': {
            'model_type': 'Distilled LLaVA-3D Student',
            'parameters': '~3B',
            'device': 'cuda' if os.system('nvidia-smi > /dev/null 2>&1') == 0 else 'cpu'
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    
    print(f"\n💾 Comprehensive results saved to: {results_file}")
    
    return results

if __name__ == "__main__":
    run_all_benchmarks()
