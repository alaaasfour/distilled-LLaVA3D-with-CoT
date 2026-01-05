#!/usr/bin/env python3
"""
Specialized 3D Spatial Reasoning Benchmark Evaluation
Evaluates on SpatialBench/3DSRBench-style tasks:
- Proximity reasoning
- Contact relationships
- Size comparisons
- Spatial orientation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image
import torchvision.transforms as transforms
import logging
from collections import defaultdict

from scripts.distillation.student_model import DistilledLLaVA3D, DistilledLLaVA3DConfig, MockVisionEncoder
from real_llava3d_teacher import RealLLaVA3DTeacher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpatialBenchmarkEvaluator:
    """
    Evaluates 3D spatial reasoning capabilities on specialized benchmarks.
    Tests: proximity, contact, size comparison, orientation.
    """
    
    def __init__(self,
                 student_checkpoint: str,
                 device: str = "cuda",
                 teacher_device: str = "cpu"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.teacher_device = teacher_device
        
        # Load models
        logger.info(f"Loading student model from {student_checkpoint}...")
        self.student_model = self._load_student_model(student_checkpoint)
        self.student_model.eval()
        
        logger.info("Loading teacher model...")
        self.teacher_model = RealLLaVA3DTeacher(
            model_path="/home/alasfour/scratch/llava-3d/LLaVA-3D",
            device=teacher_device
        )
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    
    def _load_student_model(self, checkpoint_path: str) -> DistilledLLaVA3D:
        """Load student model with architecture compatibility."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        has_old_arch = 'vision_encoder.conv_layers.0.weight' in state_dict
        has_new_arch = 'vision_encoder.vggt_model' in state_dict
        
        config = DistilledLLaVA3DConfig()
        
        if has_old_arch and not has_new_arch:
            logger.info("⚠️  Detected old architecture (MockVisionEncoder)")
            model = DistilledLLaVA3D(config)
            model.vision_encoder = MockVisionEncoder(config)
            model.load_state_dict(state_dict, strict=False)
        else:
            logger.info("✅ Detected new architecture (VGGTVisionEncoder)")
            model = DistilledLLaVA3D(config)
            model.load_state_dict(state_dict, strict=False)
        
        model.to(self.device)
        return model
    
    def evaluate_proximity(self, samples: List[Dict]) -> Dict:
        """
        Evaluate proximity reasoning: "Is object A near object B?"
        """
        logger.info("Evaluating proximity reasoning...")
        
        proximity_questions = [
            "Is there a chair near the table?",
            "Is the lamp close to the bed?",
            "Are there objects near the window?",
            "Is the sofa near the TV?",
            "Are there items close to the door?"
        ]
        
        results = {
            'correct': 0,
            'total': 0,
            'student_answers': [],
            'teacher_answers': []
        }
        
        for sample in samples:
            try:
                img_path = Path(sample["image_path"])
                if not img_path.exists():
                    continue
                
                image = Image.open(img_path).convert('RGB')
                image_tensor = self.transform(image).unsqueeze(0).to(self.device).float()
                
                for question in proximity_questions:
                    # Student response
                    with torch.no_grad():
                        student_response = self.student_model.generate_response(question, image_tensor)
                        if isinstance(student_response, dict):
                            student_text = student_response.get('response', str(student_response))
                        else:
                            student_text = str(student_response)
                    
                    # Teacher response
                    teacher_response_dict = self.teacher_model.generate_response(question, str(img_path))
                    if isinstance(teacher_response_dict, dict):
                        teacher_text = teacher_response_dict.get('response', str(teacher_response_dict))
                    else:
                        teacher_text = str(teacher_response_dict)
                    
                    # Simple keyword matching for proximity (can be enhanced with better parsing)
                    proximity_keywords = ['near', 'close', 'next to', 'beside', 'adjacent']
                    student_has_proximity = any(kw in student_text.lower() for kw in proximity_keywords)
                    teacher_has_proximity = any(kw in teacher_text.lower() for kw in proximity_keywords)
                    
                    results['student_answers'].append({
                        'question': question,
                        'answer': student_text,
                        'has_proximity': student_has_proximity
                    })
                    results['teacher_answers'].append({
                        'question': question,
                        'answer': teacher_text,
                        'has_proximity': teacher_has_proximity
                    })
                    
                    if student_has_proximity == teacher_has_proximity:
                        results['correct'] += 1
                    results['total'] += 1
                    
            except Exception as e:
                logger.warning(f"Error in proximity evaluation: {e}")
                continue
        
        results['accuracy'] = results['correct'] / results['total'] if results['total'] > 0 else 0.0
        return results
    
    def evaluate_contact(self, samples: List[Dict]) -> Dict:
        """
        Evaluate contact relationships: "Is object A touching/on/under object B?"
        """
        logger.info("Evaluating contact relationships...")
        
        contact_questions = [
            "Is there an object on the table?",
            "Is something touching the wall?",
            "Are there items on the floor?",
            "Is there a lamp on the desk?",
            "Are there objects on the shelf?"
        ]
        
        results = {
            'correct': 0,
            'total': 0,
            'student_answers': [],
            'teacher_answers': []
        }
        
        for sample in samples:
            try:
                img_path = Path(sample["image_path"])
                if not img_path.exists():
                    continue
                
                image = Image.open(img_path).convert('RGB')
                image_tensor = self.transform(image).unsqueeze(0).to(self.device).float()
                
                for question in contact_questions:
                    with torch.no_grad():
                        student_response = self.student_model.generate_response(question, image_tensor)
                        if isinstance(student_response, dict):
                            student_text = student_response.get('response', str(student_response))
                        else:
                            student_text = str(student_response)
                    
                    teacher_response_dict = self.teacher_model.generate_response(question, str(img_path))
                    if isinstance(teacher_response_dict, dict):
                        teacher_text = teacher_response_dict.get('response', str(teacher_response_dict))
                    else:
                        teacher_text = str(teacher_response_dict)
                    
                    contact_keywords = ['on', 'touching', 'contact', 'above', 'below', 'under', 'over']
                    student_has_contact = any(kw in student_text.lower() for kw in contact_keywords)
                    teacher_has_contact = any(kw in teacher_text.lower() for kw in contact_keywords)
                    
                    results['student_answers'].append({
                        'question': question,
                        'answer': student_text,
                        'has_contact': student_has_contact
                    })
                    results['teacher_answers'].append({
                        'question': question,
                        'answer': teacher_text,
                        'has_contact': teacher_has_contact
                    })
                    
                    if student_has_contact == teacher_has_contact:
                        results['correct'] += 1
                    results['total'] += 1
                    
            except Exception as e:
                logger.warning(f"Error in contact evaluation: {e}")
                continue
        
        results['accuracy'] = results['correct'] / results['total'] if results['total'] > 0 else 0.0
        return results
    
    def evaluate_size_comparison(self, samples: List[Dict]) -> Dict:
        """
        Evaluate size comparison: "Is object A larger than object B?"
        """
        logger.info("Evaluating size comparison...")
        
        size_questions = [
            "What is the largest object in the scene?",
            "Are there small objects in the room?",
            "Is the table bigger than the chair?",
            "What is the smallest item you can see?",
            "Compare the sizes of objects in this scene."
        ]
        
        results = {
            'correct': 0,
            'total': 0,
            'student_answers': [],
            'teacher_answers': []
        }
        
        for sample in samples:
            try:
                img_path = Path(sample["image_path"])
                if not img_path.exists():
                    continue
                
                image = Image.open(img_path).convert('RGB')
                image_tensor = self.transform(image).unsqueeze(0).to(self.device).float()
                
                for question in size_questions:
                    with torch.no_grad():
                        student_response = self.student_model.generate_response(question, image_tensor)
                        if isinstance(student_response, dict):
                            student_text = student_response.get('response', str(student_response))
                        else:
                            student_text = str(student_response)
                    
                    teacher_response_dict = self.teacher_model.generate_response(question, str(img_path))
                    if isinstance(teacher_response_dict, dict):
                        teacher_text = teacher_response_dict.get('response', str(teacher_response_dict))
                    else:
                        teacher_text = str(teacher_response_dict)
                    
                    size_keywords = ['large', 'small', 'big', 'bigger', 'smaller', 'size', 'compare']
                    student_has_size = any(kw in student_text.lower() for kw in size_keywords)
                    teacher_has_size = any(kw in teacher_text.lower() for kw in size_keywords)
                    
                    results['student_answers'].append({
                        'question': question,
                        'answer': student_text,
                        'has_size': student_has_size
                    })
                    results['teacher_answers'].append({
                        'question': question,
                        'answer': teacher_text,
                        'has_size': teacher_has_size
                    })
                    
                    if student_has_size == teacher_has_size:
                        results['correct'] += 1
                    results['total'] += 1
                    
            except Exception as e:
                logger.warning(f"Error in size comparison evaluation: {e}")
                continue
        
        results['accuracy'] = results['correct'] / results['total'] if results['total'] > 0 else 0.0
        return results
    
    def evaluate_orientation(self, samples: List[Dict]) -> Dict:
        """
        Evaluate spatial orientation: "Is object A to the left/right/front/back of object B?"
        """
        logger.info("Evaluating spatial orientation...")
        
        orientation_questions = [
            "What is to the left of the window?",
            "What is to the right of the door?",
            "Describe the spatial layout of objects.",
            "What is in front of the camera?",
            "What objects are positioned on the sides?"
        ]
        
        results = {
            'correct': 0,
            'total': 0,
            'student_answers': [],
            'teacher_answers': []
        }
        
        for sample in samples:
            try:
                img_path = Path(sample["image_path"])
                if not img_path.exists():
                    continue
                
                image = Image.open(img_path).convert('RGB')
                image_tensor = self.transform(image).unsqueeze(0).to(self.device).float()
                
                for question in orientation_questions:
                    with torch.no_grad():
                        student_response = self.student_model.generate_response(question, image_tensor)
                        if isinstance(student_response, dict):
                            student_text = student_response.get('response', str(student_response))
                        else:
                            student_text = str(student_response)
                    
                    teacher_response_dict = self.teacher_model.generate_response(question, str(img_path))
                    if isinstance(teacher_response_dict, dict):
                        teacher_text = teacher_response_dict.get('response', str(teacher_response_dict))
                    else:
                        teacher_text = str(teacher_response_dict)
                    
                    orientation_keywords = ['left', 'right', 'front', 'back', 'behind', 'side', 'position']
                    student_has_orientation = any(kw in student_text.lower() for kw in orientation_keywords)
                    teacher_has_orientation = any(kw in teacher_text.lower() for kw in orientation_keywords)
                    
                    results['student_answers'].append({
                        'question': question,
                        'answer': student_text,
                        'has_orientation': student_has_orientation
                    })
                    results['teacher_answers'].append({
                        'question': question,
                        'answer': teacher_text,
                        'has_orientation': teacher_has_orientation
                    })
                    
                    if student_has_orientation == teacher_has_orientation:
                        results['correct'] += 1
                    results['total'] += 1
                    
            except Exception as e:
                logger.warning(f"Error in orientation evaluation: {e}")
                continue
        
        results['accuracy'] = results['correct'] / results['total'] if results['total'] > 0 else 0.0
        return results
    
    def run_full_evaluation(self, samples: List[Dict]) -> Dict:
        """Run full spatial reasoning benchmark evaluation."""
        logger.info("=" * 60)
        logger.info("Spatial Reasoning Benchmark Evaluation")
        logger.info("=" * 60)
        
        results = {
            'proximity': self.evaluate_proximity(samples),
            'contact': self.evaluate_contact(samples),
            'size_comparison': self.evaluate_size_comparison(samples),
            'orientation': self.evaluate_orientation(samples)
        }
        
        # Overall accuracy
        total_correct = sum(r['correct'] for r in results.values())
        total_questions = sum(r['total'] for r in results.values())
        results['overall_accuracy'] = total_correct / total_questions if total_questions > 0 else 0.0
        
        return results


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Spatial Reasoning Benchmark Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to student model checkpoint')
    parser.add_argument('--data_root', type=str, default='/home/alasfour/scratch/distilled-llava3d/data',
                       help='Root directory for evaluation data')
    parser.add_argument('--num_samples', type=int, default=50,
                       help='Number of samples to evaluate')
    parser.add_argument('--output', type=str, default='results/spatial_benchmark_results.json',
                       help='Output JSON file')
    
    args = parser.parse_args()
    
    # Load samples (similar to comprehensive_evaluation.py)
    samples = []
    data_root = Path(args.data_root)
    for dataset_dir in ['scannet', '3d_front', 'matterport3d']:
        dataset_path = data_root / dataset_dir
        if not dataset_path.exists():
            continue
        for scene_dir in dataset_path.iterdir():
            if not scene_dir.is_dir():
                continue
            images_dir = scene_dir / 'images'
            if images_dir.exists():
                image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
            else:
                image_files = list(scene_dir.glob('*.jpg')) + list(scene_dir.glob('*.png'))
            for img_file in image_files[:2]:  # 2 per scene
                if len(samples) >= args.num_samples:
                    break
                samples.append({
                    'image_path': str(img_file),
                    'scene': scene_dir.name,
                    'dataset': dataset_dir
                })
            if len(samples) >= args.num_samples:
                break
        if len(samples) >= args.num_samples:
            break
    
    logger.info(f"✅ Loaded {len(samples)} evaluation samples")
    
    # Run evaluation
    evaluator = SpatialBenchmarkEvaluator(
        student_checkpoint=args.checkpoint,
        device='cuda',
        teacher_device='cpu'
    )
    
    results = evaluator.run_full_evaluation(samples)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("SPATIAL REASONING BENCHMARK RESULTS")
    logger.info("=" * 60)
    logger.info(f"Proximity Accuracy: {results['proximity']['accuracy']:.4f}")
    logger.info(f"Contact Accuracy: {results['contact']['accuracy']:.4f}")
    logger.info(f"Size Comparison Accuracy: {results['size_comparison']['accuracy']:.4f}")
    logger.info(f"Orientation Accuracy: {results['orientation']['accuracy']:.4f}")
    logger.info(f"Overall Accuracy: {results['overall_accuracy']:.4f}")
    logger.info(f"\n💾 Results saved to {output_path}")


if __name__ == "__main__":
    main()




