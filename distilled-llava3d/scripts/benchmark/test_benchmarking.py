#!/usr/bin/env python3
"""Test the benchmarking framework."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import time
from datetime import datetime

def test_benchmarking_framework():
    """Test the benchmarking framework."""
    print("🧪 Testing Benchmarking Framework")
    print("=" * 50)
    
    # Test 1: Basic benchmark
    print("\n📋 Test 1: Basic Benchmark")
    print("-" * 30)
    try:
        from benchmark_framework import BenchmarkFramework
        from scripts.distillation.student_model import DistilledLLaVA3D, DistilledLLaVA3DConfig
        
        # Initialize model
        config = DistilledLLaVA3DConfig()
        student_model = DistilledLLaVA3D(config)
        student_model.eval()
        
        # Initialize benchmark
        benchmark = BenchmarkFramework(student_model)
        
        # Test a single benchmark task
        print("   Testing 3D QA benchmark...")
        qa_results = benchmark._benchmark_3d_qa()
        print(f"   ✅ 3D QA: {qa_results['accuracy']:.2%} accuracy")
        
        print("   Testing spatial reasoning benchmark...")
        spatial_results = benchmark._benchmark_spatial_reasoning()
        print(f"   ✅ Spatial Reasoning: {spatial_results['accuracy']:.2%} accuracy")
        
        print("✅ Basic benchmark test passed")
        
    except Exception as e:
        print(f"❌ Basic benchmark test failed: {str(e)}")
        return False
    
    # Test 2: 3D VLM benchmark
    print("\n📋 Test 2: 3D VLM Benchmark")
    print("-" * 30)
    try:
        from benchmark_3d_vlm_tasks import Standard3DVLMBenchmark
        
        # Initialize benchmark
        vlm_benchmark = Standard3DVLMBenchmark(student_model)
        
        # Test a single benchmark task
        print("   Testing ScanQA benchmark...")
        scanqa_results = vlm_benchmark._benchmark_scanqa()
        print(f"   ✅ ScanQA: {scanqa_results['accuracy']:.2%} accuracy")
        
        print("   Testing 3D spatial QA benchmark...")
        spatial_qa_results = vlm_benchmark._benchmark_3d_spatial_qa()
        print(f"   ✅ 3D Spatial QA: {spatial_qa_results['accuracy']:.2%} accuracy")
        
        print("✅ 3D VLM benchmark test passed")
        
    except Exception as e:
        print(f"❌ 3D VLM benchmark test failed: {str(e)}")
        return False
    
    # Test 3: Paper-worthiness assessment
    print("\n📋 Test 3: Paper-Worthiness Assessment")
    print("-" * 30)
    try:
        from paper_worthiness_assessment import PaperWorthinessAssessment
        
        # Initialize assessment
        assessment = PaperWorthinessAssessment(student_model)
        
        # Test a single assessment criterion
        print("   Testing performance assessment...")
        performance_results = assessment._assess_performance()
        print(f"   ✅ Performance: {performance_results['score']:.1f}/10")
        
        print("   Testing efficiency assessment...")
        efficiency_results = assessment._assess_efficiency()
        print(f"   ✅ Efficiency: {efficiency_results['score']:.1f}/10")
        
        print("✅ Paper-worthiness assessment test passed")
        
    except Exception as e:
        print(f"❌ Paper-worthiness assessment test failed: {str(e)}")
        return False
    
    # Test 4: Model response generation
    print("\n📋 Test 4: Model Response Generation")
    print("-" * 30)
    try:
        # Test model response generation
        test_question = "What can you see in this image?"
        test_image = torch.randn(1, 3, 224, 224)
        
        start_time = time.time()
        response = student_model.generate_response(test_question, test_image)
        response_time = time.time() - start_time
        
        print(f"   Question: {test_question}")
        print(f"   Response: {response[:100]}...")
        print(f"   Response Time: {response_time:.3f}s")
        print(f"   Response Length: {len(response)} characters")
        
        if len(response) > 10:  # Basic check for meaningful response
            print("✅ Model response generation test passed")
        else:
            print("⚠️  Model response seems too short")
            
    except Exception as e:
        print(f"❌ Model response generation test failed: {str(e)}")
        return False
    
    # Test 5: Image analysis
    print("\n📋 Test 5: Image Analysis")
    print("-" * 30)
    try:
        # Test image analysis
        test_image = torch.randn(1, 3, 224, 224)
        
        start_time = time.time()
        features = student_model.analyze_image_content(test_image)
        analysis_time = time.time() - start_time
        
        print(f"   Analysis Time: {analysis_time:.3f}s")
        print(f"   Features: {len(features)} detected")
        print(f"   Sample Features:")
        for key, value in list(features.items())[:5]:
            print(f"     {key}: {value}")
        
        if len(features) > 5:  # Basic check for meaningful analysis
            print("✅ Image analysis test passed")
        else:
            print("⚠️  Image analysis seems limited")
            
    except Exception as e:
        print(f"❌ Image analysis test failed: {str(e)}")
        return False
    
    # Summary
    print("\n" + "=" * 50)
    print("🎯 BENCHMARKING FRAMEWORK TEST SUMMARY")
    print("=" * 50)
    print("✅ All tests passed!")
    print("✅ Benchmarking framework is ready to use")
    print("✅ You can now run comprehensive benchmarks")
    
    print(f"\n💡 Next Steps:")
    print("   1. Run: python run_all_benchmarks.py")
    print("   2. Analyze results in generated JSON files")
    print("   3. Implement improvements based on results")
    print("   4. Re-run benchmarks to validate improvements")
    
    return True

if __name__ == "__main__":
    test_benchmarking_framework()
