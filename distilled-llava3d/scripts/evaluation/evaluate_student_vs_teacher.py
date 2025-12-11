#!/usr/bin/env python3
"""
Evaluate student model vs teacher model on validation/test set.
Computes comprehensive metrics for publication.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import logging

from scripts.distillation.student_model import DistilledLLaVA3D, DistilledLLaVA3DConfig
from real_llava3d_teacher import RealLLaVA3DTeacher
from scripts.evaluation.metrics import (
    compute_text_metrics, 
    aggregate_metrics,
    compute_depth_metrics,
    compute_detection_metrics
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StudentTeacherEvaluator:
    """Evaluate student model against teacher model."""
    
    def __init__(self, 
                 student_checkpoint: str,
                 device: str = "cuda",
                 teacher_device: str = "cpu",
                 data_root: str = "/home/alasfour/scratch/distilled-llava3d/data"):
        self.device = device
        self.teacher_device = teacher_device
        self.data_root = Path(data_root)
        
        # Load student model
        logger.info(f"Loading student model from {student_checkpoint}...")
        self.student_model = self._load_student_model(student_checkpoint)
        
        # Load teacher model
        logger.info("Loading teacher model...")
        self.teacher_model = self._load_teacher_model()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        # Results storage
        self.results = {
            'student_metrics': [],
            'teacher_metrics': [],
            'comparison': {}
        }
    
    def _load_student_model(self, checkpoint_path: str) -> DistilledLLaVA3D:
        """Load student model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        config = DistilledLLaVA3DConfig()
        model = DistilledLLaVA3D(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        logger.info(f"✅ Student model loaded (epoch {checkpoint.get('epoch', 'unknown')})")
        return model
    
    def _load_teacher_model(self) -> RealLLaVA3DTeacher:
        """Load teacher model."""
        try:
            teacher = RealLLaVA3DTeacher(
                model_path="ChaimZhu/LLaVA-3D-7B",
                device=self.teacher_device
            )
            if teacher.model is None:
                raise ValueError("Teacher model not loaded")
            logger.info("✅ Teacher model loaded")
            return teacher
        except Exception as e:
            logger.error(f"❌ Failed to load teacher model: {e}")
            raise
    
    def load_evaluation_samples(self, num_samples: int = 100) -> List[Dict]:
        """Load samples for evaluation."""
        logger.info(f"Loading {num_samples} evaluation samples...")
        
        samples = []
        
        # Load from validation set or test set
        scannet_path = self.data_root / "scannet_real" / "expanded"
        front_path = self.data_root / "3d_front_real" / "expanded"
        
        # Load ScanNet samples
        if scannet_path.exists():
            scene_dirs = sorted([d for d in scannet_path.glob("scene*") if d.is_dir()])
            for scene_dir in scene_dirs[:20]:  # Limit scenes
                images = sorted(list(scene_dir.glob("*.jpg")))
                for img_path in images[:2]:  # 2 images per scene
                    if len(samples) >= num_samples:
                        break
                    samples.append({
                        "image_path": str(img_path),
                        "question": "Describe this 3D scene and identify objects.",
                        "dataset": "scannet"
                    })
                if len(samples) >= num_samples:
                    break
        
        # Load 3D-FRONT samples
        if len(samples) < num_samples and front_path.exists():
            scene_dirs = sorted([d for d in front_path.glob("*") if d.is_dir()])
            for scene_dir in scene_dirs[:20]:
                images = sorted(list(scene_dir.glob("*.jpg")))
                for img_path in images[:2]:
                    if len(samples) >= num_samples:
                        break
                    samples.append({
                        "image_path": str(img_path),
                        "question": "Describe this 3D scene and identify objects.",
                        "dataset": "3d_front"
                    })
                if len(samples) >= num_samples:
                    break
        
        logger.info(f"✅ Loaded {len(samples)} evaluation samples")
        return samples
    
    def evaluate_text_generation(self, samples: List[Dict]) -> Tuple[Dict, Dict]:
        """
        Evaluate text generation on samples.
        
        Returns:
            (student_metrics, teacher_metrics)
        """
        logger.info("Evaluating text generation...")
        
        student_text_metrics = []
        teacher_text_metrics = []
        
        for i, sample in enumerate(samples):
            if i % 10 == 0:
                logger.info(f"  Processing sample {i+1}/{len(samples)}")
            
            try:
                img_path = Path(sample["image_path"])
                if not img_path.exists():
                    continue
                
                question = sample.get("question", "Describe this 3D scene.")
                
                # Load and preprocess image
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
                
                # Extract text from responses
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
                teacher_metrics = compute_text_metrics(teacher_text, teacher_text)  # Perfect match
                
                student_text_metrics.append(student_metrics)
                teacher_text_metrics.append(teacher_metrics)
                
            except Exception as e:
                logger.warning(f"  Error processing sample {i}: {e}")
                continue
        
        # Aggregate metrics
        student_agg = aggregate_metrics(student_text_metrics)
        teacher_agg = aggregate_metrics(teacher_text_metrics)
        
        return student_agg, teacher_agg
    
    def evaluate_inference_speed(self, samples: List[Dict], num_warmup: int = 5) -> Dict:
        """Evaluate inference speed for student and teacher."""
        logger.info("Evaluating inference speed...")
        
        # Warmup
        for sample in samples[:num_warmup]:
            try:
                img_path = Path(sample["image_path"])
                if not img_path.exists():
                    continue
                image = Image.open(img_path).convert('RGB')
                image_tensor = self.transform(image).unsqueeze(0).to(self.device).float()
                
                with torch.no_grad():
                    _ = self.student_model.generate_response(
                        "Describe this scene.",
                        image_tensor
                    )
            except:
                pass
        
        # Student inference speed
        student_times = []
        for sample in samples[num_warmup:num_warmup+20]:  # Test on 20 samples
            try:
                img_path = Path(sample["image_path"])
                if not img_path.exists():
                    continue
                image = Image.open(img_path).convert('RGB')
                image_tensor = self.transform(image).unsqueeze(0).to(self.device).float()
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()
                
                with torch.no_grad():
                    _ = self.student_model.generate_response(
                        "Describe this scene.",
                        image_tensor
                    )
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                elapsed = time.time() - start_time
                student_times.append(elapsed)
            except:
                pass
        
        # Teacher inference speed (on CPU, so slower)
        teacher_times = []
        for sample in samples[num_warmup:num_warmup+10]:  # Test on 10 samples (teacher is slower)
            try:
                img_path = Path(sample["image_path"])
                if not img_path.exists():
                    continue
                
                start_time = time.time()
                _ = self.teacher_model.generate_response(
                    "Describe this scene.",
                    str(img_path)
                )
                elapsed = time.time() - start_time
                teacher_times.append(elapsed)
            except:
                pass
        
        student_avg_time = np.mean(student_times) if student_times else float('inf')
        teacher_avg_time = np.mean(teacher_times) if teacher_times else float('inf')
        
        speedup = teacher_avg_time / student_avg_time if student_avg_time > 0 else 0.0
        
        return {
            'student_avg_time_ms': student_avg_time * 1000,
            'teacher_avg_time_ms': teacher_avg_time * 1000,
            'speedup': speedup,
            'student_fps': 1.0 / student_avg_time if student_avg_time > 0 else 0.0,
            'teacher_fps': 1.0 / teacher_avg_time if teacher_avg_time > 0 else 0.0
        }
    
    def evaluate_memory_usage(self) -> Dict:
        """Evaluate memory usage."""
        logger.info("Evaluating memory usage...")
        
        # Student model size
        student_params = sum(p.numel() for p in self.student_model.parameters())
        student_size_mb = student_params * 4 / (1024 ** 2)  # Assuming float32
        
        # Teacher model size (approximate - 7B parameters)
        teacher_params = 7_000_000_000  # 7B
        teacher_size_mb = teacher_params * 4 / (1024 ** 2)
        
        # Peak GPU memory during inference
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            # Run a dummy inference
            dummy_image = torch.randn(1, 3, 224, 224).to(self.device)
            with torch.no_grad():
                _ = self.student_model.generate_response("Test", dummy_image)
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        else:
            peak_memory_mb = 0.0
        
        return {
            'student_params': int(student_params),
            'student_size_mb': student_size_mb,
            'teacher_params': int(teacher_params),
            'teacher_size_mb': teacher_size_mb,
            'compression_ratio': teacher_size_mb / student_size_mb if student_size_mb > 0 else 0.0,
            'peak_gpu_memory_mb': peak_memory_mb
        }
    
    def run_evaluation(self, num_samples: int = 100, output_path: str = None):
        """Run full evaluation."""
        logger.info("=" * 60)
        logger.info("Starting Student vs Teacher Evaluation")
        logger.info("=" * 60)
        
        # Load samples
        samples = self.load_evaluation_samples(num_samples)
        
        if len(samples) == 0:
            logger.error("No evaluation samples found!")
            return
        
        # Evaluate text generation
        student_text_metrics, teacher_text_metrics = self.evaluate_text_generation(samples)
        
        # Evaluate inference speed
        speed_metrics = self.evaluate_inference_speed(samples)
        
        # Evaluate memory usage
        memory_metrics = self.evaluate_memory_usage()
        
        # Compile results
        results = {
            'student_metrics': {
                'text_generation': student_text_metrics
            },
            'teacher_metrics': {
                'text_generation': teacher_text_metrics
            },
            'efficiency': {
                'speed': speed_metrics,
                'memory': memory_metrics
            },
            'comparison': {
                'text_metrics_ratio': {
                    k: student_text_metrics.get(k, 0) / teacher_text_metrics.get(k, 1) 
                    for k in student_text_metrics.keys() 
                    if k in teacher_text_metrics and teacher_text_metrics.get(k, 0) > 0
                }
            },
            'num_samples': len(samples)
        }
        
        # Print results
        logger.info("\n" + "=" * 60)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 60)
        
        logger.info("\n📊 Text Generation Metrics:")
        logger.info("  Student:")
        for key, value in student_text_metrics.items():
            if not key.endswith('_std'):
                logger.info(f"    {key}: {value:.4f}")
        
        logger.info("  Teacher:")
        for key, value in teacher_text_metrics.items():
            if not key.endswith('_std'):
                logger.info(f"    {key}: {value:.4f}")
        
        logger.info("\n⚡ Efficiency Metrics:")
        logger.info(f"  Student FPS: {speed_metrics['student_fps']:.2f}")
        logger.info(f"  Teacher FPS: {speed_metrics['teacher_fps']:.2f}")
        logger.info(f"  Speedup: {speed_metrics['speedup']:.2f}x")
        logger.info(f"  Student Size: {memory_metrics['student_size_mb']:.2f} MB")
        logger.info(f"  Teacher Size: {memory_metrics['teacher_size_mb']:.2f} MB")
        logger.info(f"  Compression: {memory_metrics['compression_ratio']:.2f}x")
        
        logger.info("\n📈 Performance Ratio (Student/Teacher):")
        for key, value in results['comparison']['text_metrics_ratio'].items():
            logger.info(f"  {key}: {value:.2%}")
        
        # Save results
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"\n💾 Results saved to {output_path}")
        
        return results


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate student vs teacher model")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to student model checkpoint")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of samples to evaluate")
    parser.add_argument("--output", type=str, default="evaluation_results.json",
                        help="Output path for results JSON")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for student model")
    parser.add_argument("--teacher_device", type=str, default="cpu",
                        help="Device for teacher model")
    
    args = parser.parse_args()
    
    evaluator = StudentTeacherEvaluator(
        student_checkpoint=args.checkpoint,
        device=args.device,
        teacher_device=args.teacher_device
    )
    
    results = evaluator.run_evaluation(
        num_samples=args.num_samples,
        output_path=args.output
    )


if __name__ == "__main__":
    main()

