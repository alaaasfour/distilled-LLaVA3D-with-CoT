#!/usr/bin/env python3
"""
Test the complete distilled LLaVA-3D pipeline.
"""

import os
import sys
import subprocess

def test_component(script_path, description):
    """Test a component of the pipeline."""
    print(f"\n🧪 Testing {description}...")
    print(f"Script: {script_path}")
    
    if not os.path.exists(script_path):
        print(f"❌ Script not found: {script_path}")
        return False
        
    try:
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print(f"✅ {description} - PASSED")
            return True
        else:
            print(f"❌ {description} - FAILED")
            print(f"Error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} - TIMEOUT")
        return False
    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return False

def main():
    """Test the complete pipeline."""
    print("🚀 Testing Complete Distilled LLaVA-3D Pipeline")
    print("=" * 60)
    
    # Test components
    tests = [
        ("scripts/distillation/student_model.py", "Student Model Architecture"),
        ("scripts/distillation/distillation_loss.py", "Distillation Loss Functions"),
        ("scripts/distillation/load_teacher.py", "Teacher Model Loading"),
        ("scripts/evaluation/evaluate_distilled.py", "Evaluation System"),
    ]
    
    passed = 0
    total = len(tests)
    
    for script, description in tests:
        if test_component(script, description):
            passed += 1
    
    # Summary
    print(f"\n📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Pipeline is ready.")
        print("\nNext steps:")
        print("1. Run: python scripts/distillation/train_distilled_optimized.py")
        print("2. Run: python scripts/evaluation/evaluate_3d_tasks.py")
        print("3. Check logs/ and models/checkpoints/ for results")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
