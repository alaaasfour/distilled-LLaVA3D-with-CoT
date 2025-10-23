#!/usr/bin/env python3
"""Comprehensive test script for the improved distilled LLaVA-3D model."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchvision.transforms as transforms
from PIL import Image
import json
import time
from datetime import datetime
from pathlib import Path

# Import the improved model
from scripts.distillation.student_model import DistilledLLaVA3D, DistilledLLaVA3DConfig

class ImprovedModelTester:
    """Test the improved distilled LLaVA-3D model."""
    
    def __init__(self):
        self.config = DistilledLLaVA3DConfig()
        self.model = DistilledLLaVA3D(self.config)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        # Test cases
        self.test_cases = [
            {
                "image": "/scratch/alasfour/llava-3d/LLaVA-3D/demo/my_images/IMG_001.png",
                "questions": [
                    "What can you see in this image?",
                    "Describe the spatial relationships in this scene.",
                    "What are the things I should be cautious about when I visit here?",
                    "Is this an indoor or outdoor scene?",
                    "What objects are visible in this image?"
                ]
            },
            {
                "image": "https://llava-vl.github.io/static/images/view.jpg",
                "questions": [
                    "What can you see in this image?",
                    "Describe the spatial relationships in this scene.",
                    "What are the things I should be cautious about when I visit here?",
                    "Is this an indoor or outdoor scene?",
                    "What objects are visible in this image?"
                ]
            }
        ]
    
    def load_and_preprocess_image(self, image_path):
        """Load and preprocess image for testing."""
        try:
            if image_path.startswith("http"):
                # For web images, we'll use a placeholder for now
                print(f"⚠️  Web image {image_path} - using placeholder")
                # Create a random image as placeholder
                pixel_values = torch.randn(1, 3, 224, 224)
                return pixel_values, True
            else:
                # Load local image
                image = Image.open(image_path).convert('RGB')
                pixel_values = self.transform(image).unsqueeze(0)
                return pixel_values, False
        except Exception as e:
            print(f"❌ Error loading image {image_path}: {e}")
            return None, False
    
    def test_single_image(self, image_path, questions):
        """Test a single image with multiple questions."""
        print(f"\n🖼️  Testing image: {image_path}")
        print("=" * 80)
        
        # Load image
        pixel_values, is_placeholder = self.load_and_preprocess_image(image_path)
        if pixel_values is None:
            return None
        
        results = {
            "image_path": image_path,
            "is_placeholder": is_placeholder,
            "questions": []
        }
        
        # Test each question
        for i, question in enumerate(questions, 1):
            print(f"\n❓ Question {i}: {question}")
            print("-" * 60)
            
            try:
                start_time = time.time()
                
                # Get model response
                response = self.model.generate_response(question, pixel_values)
                
                # Get detailed analysis
                with torch.no_grad():
                    features = self.model.analyze_image_content(pixel_values)
                
                end_time = time.time()
                response_time = end_time - start_time
                
                print(f"🤖 Response: {response}")
                print(f"⏱️  Response time: {response_time:.2f}s")
                
                # Print detailed analysis
                print(f"🔍 Analysis:")
                print(f"   - Indoor: {features['is_indoor']}")
                print(f"   - Outdoor: {features['is_outdoor']}")
                print(f"   - Has Sky: {features['has_sky']}")
                print(f"   - Has Person: {features['has_person']}")
                print(f"   - Has Buildings: {features['has_buildings']}")
                print(f"   - Has Natural Elements: {features['has_natural_elements']}")
                print(f"   - Outdoor Score: {features['outdoor_score']}")
                print(f"   - Indoor Score: {features['indoor_score']}")
                print(f"   - Confidence: {features['outdoor_confidence']:.3f}")
                
                # Store results
                results["questions"].append({
                    "question": question,
                    "response": response,
                    "response_time": response_time,
                    "features": features
                })
                
            except Exception as e:
                print(f"❌ Error processing question: {e}")
                results["questions"].append({
                    "question": question,
                    "response": f"Error: {str(e)}",
                    "response_time": 0,
                    "features": {}
                })
        
        return results
    
    def run_comprehensive_test(self):
        """Run comprehensive tests on all test cases."""
        print("🚀 Starting Comprehensive Test of Improved Distilled LLaVA-3D Model")
        print("=" * 80)
        
        all_results = {
            "test_timestamp": datetime.now().isoformat(),
            "model_type": "Improved Distilled LLaVA-3D",
            "test_cases": []
        }
        
        for i, test_case in enumerate(self.test_cases, 1):
            print(f"\n📋 Test Case {i}/{len(self.test_cases)}")
            results = self.test_single_image(test_case["image"], test_case["questions"])
            if results:
                all_results["test_cases"].append(results)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"improved_model_test_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {results_file}")
        
        # Print summary
        self.print_test_summary(all_results)
        
        return all_results
    
    def print_test_summary(self, results):
        """Print a summary of test results."""
        print("\n📊 TEST SUMMARY")
        print("=" * 50)
        
        total_questions = 0
        successful_responses = 0
        outdoor_detections = 0
        sky_detections = 0
        person_detections = 0
        
        for test_case in results["test_cases"]:
            for question_result in test_case["questions"]:
                total_questions += 1
                
                if not question_result["response"].startswith("Error:"):
                    successful_responses += 1
                
                features = question_result.get("features", {})
                if features.get("is_outdoor", False):
                    outdoor_detections += 1
                if features.get("has_sky", False):
                    sky_detections += 1
                if features.get("has_person", False):
                    person_detections += 1
        
        print(f"Total Questions: {total_questions}")
        print(f"Successful Responses: {successful_responses} ({successful_responses/total_questions*100:.1f}%)")
        print(f"Outdoor Detections: {outdoor_detections}")
        print(f"Sky Detections: {sky_detections}")
        print(f"Person Detections: {person_detections}")
        
        # Average response time
        total_time = sum(
            sum(q["response_time"] for q in tc["questions"]) 
            for tc in results["test_cases"]
        )
        avg_time = total_time / total_questions if total_questions > 0 else 0
        print(f"Average Response Time: {avg_time:.2f}s")

def main():
    """Main test function."""
    print("🧪 Improved Distilled LLaVA-3D Model Comprehensive Test")
    print("=" * 60)
    
    tester = ImprovedModelTester()
    results = tester.run_comprehensive_test()
    
    print("\n✅ Comprehensive test completed!")
    return results

if __name__ == "__main__":
    main()



