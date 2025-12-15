#!/usr/bin/env python3
"""
Comprehensive evaluation script for distilled LLaVA-3D model.
Evaluates text generation, depth estimation, object detection, and efficiency metrics.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image
import torchvision.transforms as transforms
import logging
from collections import defaultdict
import argparse

from scripts.distillation.student_model import DistilledLLaVA3D, DistilledLLaVA3DConfig, MockVisionEncoder
from real_llava3d_teacher import RealLLaVA3DTeacher
from scripts.evaluation.metrics import (
    compute_text_metrics,
    aggregate_metrics,
    compute_depth_metrics,
    compute_detection_metrics
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ComprehensiveEvaluator:
    """Comprehensive evaluation including all metrics."""
    
    def __init__(self,
                 student_checkpoint: str,
                 device: str = "cuda",
                 teacher_device: str = "cpu",
                 data_root: str = "/home/alasfour/scratch/distilled-llava3d/data"):
        # Auto-detect device
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, using CPU")
            device = "cpu"
        self.device = device
        self.teacher_device = teacher_device
        self.data_root = Path(data_root)
        
        # Load models
        logger.info(f"Loading student model from {student_checkpoint}...")
        self.student_model = self._load_student_model(student_checkpoint)
        self.student_model.eval()
        
        logger.info("Loading teacher model...")
        self.teacher_model = RealLLaVA3DTeacher(
            model_path="/home/alasfour/scratch/llava-3d/LLaVA-3D",
            device=teacher_device
        )
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    
    def _load_student_model(self, checkpoint_path: str) -> DistilledLLaVA3D:
        """Load student model with architecture compatibility."""
        # Use CPU for loading, then move to device
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Detect architecture
        has_old_arch = 'vision_encoder.conv_layers.0.weight' in state_dict
        has_new_arch = 'vision_encoder.vggt_model' in state_dict or 'vision_encoder.feature_projection.weight' in state_dict
        
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
    
    def load_evaluation_samples(self, num_samples: int = 100) -> List[Dict]:
        """Load evaluation samples from data directory."""
        samples = []
        
        # Load from various datasets
        for dataset_dir in ['scannet', '3d_front', 'matterport3d']:
            dataset_path = self.data_root / dataset_dir
            if not dataset_path.exists():
                continue
            
            for scene_dir in dataset_path.iterdir():
                if not scene_dir.is_dir():
                    continue
                
                # Look for images - try both 'images' subdirectory and direct in scene dir
                images_dir = scene_dir / 'images'
                if images_dir.exists():
                    image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
                else:
                    # Images directly in scene directory
                    image_files = list(scene_dir.glob('*.jpg')) + list(scene_dir.glob('*.png'))
                
                if not image_files:
                    continue
                
                # Get first few images from each scene
                for img_file in image_files[:3]:  # Max 3 per scene
                    if len(samples) >= num_samples:
                        break
                    
                    samples.append({
                        'image_path': str(img_file),
                        'scene': scene_dir.name,
                        'dataset': dataset_dir
                    })
                
                if len(samples) >= num_samples:
                    break
            
            if len(samples) >= num_samples:
                break
        
        logger.info(f"✅ Loaded {len(samples)} evaluation samples")
        return samples
    
    def evaluate_text_generation(self, samples: List[Dict]) -> Dict:
        """Evaluate text generation metrics."""
        logger.info("Evaluating text generation...")
        
        student_text_metrics = []
        teacher_text_metrics = []
        
        for i, sample in enumerate(samples):
            try:
                img_path = Path(sample["image_path"])
                if not img_path.exists():
                    continue
                
                question = sample.get("question", "Describe this 3D scene.")
                
                # Load image
                image = Image.open(img_path).convert('RGB')
                image_tensor = self.transform(image).unsqueeze(0).to(self.device).float()
                
                # Get student response
                with torch.no_grad():
                    student_response = self.student_model.generate_response(
                        question,
                        image_tensor
                    )
                
                # Get teacher response
                teacher_response_dict = self.teacher_model.generate_response(
                    question,
                    str(img_path)
                )
                
                # Extract text
                if isinstance(student_response, dict):
                    student_text = student_response.get('response', str(student_response))
                else:
                    student_text = str(student_response)
                
                if isinstance(teacher_response_dict, dict):
                    teacher_text = teacher_response_dict.get('response', str(teacher_response_dict))
                else:
                    teacher_text = str(teacher_response_dict)
                
                # Compute metrics (using teacher as reference)
                student_metrics = compute_text_metrics(teacher_text, student_text)
                teacher_metrics = compute_text_metrics(teacher_text, teacher_text)
                
                student_text_metrics.append(student_metrics)
                teacher_text_metrics.append(teacher_metrics)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"  Processed {i + 1}/{len(samples)} samples")
            
            except Exception as e:
                logger.warning(f"  Error processing sample {i}: {e}")
                continue
        
        # Aggregate
        student_agg = aggregate_metrics(student_text_metrics)
        teacher_agg = aggregate_metrics(teacher_text_metrics)
        
        return {
            'student': student_agg,
            'teacher': teacher_agg
        }
    
    def evaluate_depth_estimation(self, samples: List[Dict]) -> Dict:
        """Evaluate depth estimation metrics."""
        logger.info("Evaluating depth estimation...")
        
        student_depth_metrics = []
        teacher_depth_metrics = []
        
        for i, sample in enumerate(samples):
            try:
                img_path = Path(sample["image_path"])
                if not img_path.exists():
                    continue
                
                # Load image
                image = Image.open(img_path).convert('RGB')
                image_tensor = self.transform(image).unsqueeze(0).to(self.device).float()
                
                # Get student depth
                with torch.no_grad():
                    student_output = self.student_model(image_tensor)
                    student_depth = student_output.get('depth', None)
                
                # Get teacher depth (if available)
                teacher_depth = None
                try:
                    teacher_output = self.teacher_model.generate_response(
                        "Estimate depth for this scene.",
                        str(img_path)
                    )
                    if isinstance(teacher_output, dict):
                        teacher_depth = teacher_output.get('depth', None)
                except:
                    pass
                
                # For now, we'll need ground truth depth for proper evaluation
                # This is a placeholder - you'll need to load actual depth maps
                if student_depth is not None:
                    # Placeholder metrics - replace with actual depth comparison
                    metrics = {
                        'rmse': 0.0,
                        'mae': 0.0,
                        'delta1': 0.0,
                        'delta2': 0.0,
                        'delta3': 0.0
                    }
                    student_depth_metrics.append(metrics)
            
            except Exception as e:
                logger.warning(f"  Error processing sample {i}: {e}")
                continue
        
        if student_depth_metrics:
            student_agg = aggregate_metrics(student_depth_metrics)
        else:
            student_agg = {}
        
        return {
            'student': student_agg,
            'teacher': {}
        }
    
    def evaluate_object_detection(self, samples: List[Dict]) -> Dict:
        """Evaluate object detection metrics."""
        logger.info("Evaluating object detection...")
        
        student_detection_metrics = []
        
        for i, sample in enumerate(samples):
            try:
                img_path = Path(sample["image_path"])
                if not img_path.exists():
                    continue
                
                # Load image
                image = Image.open(img_path).convert('RGB')
                image_tensor = self.transform(image).unsqueeze(0).to(self.device).float()
                
                # Get student detections
                with torch.no_grad():
                    student_output = self.student_model(image_tensor)
                    student_detections = student_output.get('detections', None)
                
                # Placeholder - need ground truth for proper evaluation
                if student_detections is not None:
                    metrics = {
                        'map': 0.0,
                        'precision': 0.0,
                        'recall': 0.0,
                        'f1': 0.0
                    }
                    student_detection_metrics.append(metrics)
            
            except Exception as e:
                logger.warning(f"  Error processing sample {i}: {e}")
                continue
        
        if student_detection_metrics:
            student_agg = aggregate_metrics(student_detection_metrics)
        else:
            student_agg = {}
        
        return {
            'student': student_agg,
            'teacher': {}
        }
    
    def evaluate_efficiency(self, samples: List[Dict], num_warmup: int = 5) -> Dict:
        """Evaluate efficiency metrics (speed, memory, size)."""
        logger.info("Evaluating efficiency metrics...")
        
        # Warmup
        for sample in samples[:num_warmup]:
            try:
                img_path = Path(sample["image_path"])
                if not img_path.exists():
                    continue
                image = Image.open(img_path).convert('RGB')
                image_tensor = self.transform(image).unsqueeze(0).to(self.device).float()
                with torch.no_grad():
                    _ = self.student_model.generate_response("Describe this scene.", image_tensor)
            except:
                pass
        
        # Student inference speed
        student_times = []
        for sample in samples[num_warmup:num_warmup+20]:
            try:
                img_path = Path(sample["image_path"])
                if not img_path.exists():
                    continue
                image = Image.open(img_path).convert('RGB')
                image_tensor = self.transform(image).unsqueeze(0).to(self.device).float()
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()
                
                with torch.no_grad():
                    _ = self.student_model.generate_response("Describe this scene.", image_tensor)
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                elapsed = time.time() - start_time
                student_times.append(elapsed)
            except:
                pass
        
        # Teacher inference speed (on CPU, slower)
        teacher_times = []
        for sample in samples[num_warmup:num_warmup+10]:
            try:
                img_path = Path(sample["image_path"])
                if not img_path.exists():
                    continue
                
                start_time = time.time()
                _ = self.teacher_model.generate_response("Describe this scene.", str(img_path))
                elapsed = time.time() - start_time
                teacher_times.append(elapsed)
            except:
                pass
        
        # Memory usage
        student_params = sum(p.numel() for p in self.student_model.parameters())
        student_size_mb = student_params * 4 / (1024 ** 2)  # Assuming float32
        
        teacher_params = 7_000_000_000  # Approximate for LLaVA-3D-7B
        teacher_size_mb = teacher_params * 4 / (1024 ** 2)
        
        # Peak GPU memory
        if torch.cuda.is_available():
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        else:
            peak_memory_mb = 0
        
        return {
            'speed': {
                'student_avg_time_ms': np.mean(student_times) * 1000 if student_times else 0,
                'teacher_avg_time_ms': np.mean(teacher_times) * 1000 if teacher_times else 0,
                'speedup': np.mean(teacher_times) / np.mean(student_times) if student_times and teacher_times and np.mean(student_times) > 0 else 0,
                'student_fps': 1.0 / np.mean(student_times) if student_times and np.mean(student_times) > 0 else 0,
                'teacher_fps': 1.0 / np.mean(teacher_times) if teacher_times and np.mean(teacher_times) > 0 else 0
            },
            'memory': {
                'student_params': student_params,
                'student_size_mb': student_size_mb,
                'teacher_params': teacher_params,
                'teacher_size_mb': teacher_size_mb,
                'compression_ratio': teacher_size_mb / student_size_mb if student_size_mb > 0 else 0,
                'peak_gpu_memory_mb': peak_memory_mb
            }
        }
    
    def run_comprehensive_evaluation(self, num_samples: int = 100) -> Dict:
        """Run comprehensive evaluation."""
        logger.info("=" * 60)
        logger.info("Starting Comprehensive Evaluation")
        logger.info("=" * 60)
        
        # Load samples
        samples = self.load_evaluation_samples(num_samples)
        
        # Evaluate all tasks
        results = {
            'text_generation': self.evaluate_text_generation(samples),
            'depth_estimation': self.evaluate_depth_estimation(samples),
            'object_detection': self.evaluate_object_detection(samples),
            'efficiency': self.evaluate_efficiency(samples),
            'num_samples': len(samples)
        }
        
        # Compute comparison ratios
        text_student = results['text_generation']['student']
        text_teacher = results['text_generation']['teacher']
        
        comparison = {}
        for metric in ['bleu-1', 'bleu-4', 'rouge-1', 'rouge-l', 'meteor']:
            if metric in text_student and metric in text_teacher:
                student_val = text_student[metric]
                teacher_val = text_teacher[metric]
                if teacher_val > 0:
                    comparison[f'{metric}_ratio'] = student_val / teacher_val
        
        results['comparison'] = comparison
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Comprehensive evaluation of distilled LLaVA-3D')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to student model checkpoint')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of evaluation samples')
    parser.add_argument('--output', type=str, default='results/comprehensive_evaluation.json',
                       help='Output JSON file path')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device for student model')
    parser.add_argument('--teacher_device', type=str, default='cpu',
                       help='Device for teacher model')
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Run evaluation
    evaluator = ComprehensiveEvaluator(
        student_checkpoint=args.checkpoint,
        device=args.device,
        teacher_device=args.teacher_device
    )
    
    results = evaluator.run_comprehensive_evaluation(num_samples=args.num_samples)
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION RESULTS SUMMARY")
    logger.info("=" * 60)
    
    logger.info("\n📊 Text Generation Metrics:")
    logger.info(f"  Student BLEU-1: {results['text_generation']['student'].get('bleu-1', 0):.4f}")
    logger.info(f"  Student ROUGE-1: {results['text_generation']['student'].get('rouge-1', 0):.4f}")
    logger.info(f"  Student METEOR: {results['text_generation']['student'].get('meteor', 0):.4f}")
    
    logger.info("\n⚡ Efficiency Metrics:")
    logger.info(f"  Student FPS: {results['efficiency']['speed'].get('student_fps', 0):.2f}")
    logger.info(f"  Teacher FPS: {results['efficiency']['speed'].get('teacher_fps', 0):.2f}")
    logger.info(f"  Speedup: {results['efficiency']['speed'].get('speedup', 0):.2f}x")
    logger.info(f"  Compression: {results['efficiency']['memory'].get('compression_ratio', 0):.2f}x")
    
    logger.info(f"\n💾 Results saved to {output_path}")


if __name__ == "__main__":
    main()

