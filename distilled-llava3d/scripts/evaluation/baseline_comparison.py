#!/usr/bin/env python3
"""
Baseline comparison script.
Compares student model against teacher and other SOTA methods.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import json
import numpy as np
from pathlib import Path
from typing import Dict, List
import logging
import argparse

from scripts.distillation.student_model import DistilledLLaVA3D, DistilledLLaVA3DConfig, MockVisionEncoder
from real_llava3d_teacher import RealLLaVA3DTeacher
from scripts.evaluation.metrics import compute_text_metrics, aggregate_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaselineComparator:
    """Compare student model against baselines."""
    
    def __init__(self, student_checkpoint: str, device: str = "cuda"):
        self.device = device
        self.student_model = self._load_student_model(student_checkpoint)
        self.student_model.eval()
        
        # Load teacher for comparison
        self.teacher_model = RealLLaVA3DTeacher(
            model_path="/home/alasfour/scratch/llava-3d/LLaVA-3D",
            device="cpu"
        )
    
    def _load_student_model(self, checkpoint_path: str) -> DistilledLLaVA3D:
        """Load student model."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        has_old_arch = 'vision_encoder.conv_layers.0.weight' in state_dict
        config = DistilledLLaVA3DConfig()
        
        if has_old_arch:
            model = DistilledLLaVA3D(config)
            model.vision_encoder = MockVisionEncoder(config)
            model.load_state_dict(state_dict, strict=False)
        else:
            model = DistilledLLaVA3D(config)
            model.load_state_dict(state_dict, strict=False)
        
        model.to(self.device)
        return model
    
    def compare_student_vs_teacher(self, samples: List[Dict]) -> Dict:
        """Compare student vs teacher."""
        logger.info("Comparing student vs teacher...")
        
        student_metrics = []
        teacher_metrics = []
        
        for i, sample in enumerate(samples):
            try:
                img_path = Path(sample["image_path"])
                question = sample.get("question", "Describe this 3D scene.")
                
                # Student
                from PIL import Image
                import torchvision.transforms as transforms
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                ])
                image = Image.open(img_path).convert('RGB')
                image_tensor = transform(image).unsqueeze(0).to(self.device).float()
                
                with torch.no_grad():
                    student_response = self.student_model.generate_response(question, image_tensor)
                
                # Teacher
                teacher_response = self.teacher_model.generate_response(question, str(img_path))
                
                # Extract text
                student_text = student_response.get('response', str(student_response)) if isinstance(student_response, dict) else str(student_response)
                teacher_text = teacher_response.get('response', str(teacher_response)) if isinstance(teacher_response, dict) else str(teacher_response)
                
                # Metrics
                student_met = compute_text_metrics(teacher_text, student_text)
                teacher_met = compute_text_metrics(teacher_text, teacher_text)
                
                student_metrics.append(student_met)
                teacher_metrics.append(teacher_met)
                
            except Exception as e:
                logger.warning(f"Error on sample {i}: {e}")
                continue
        
        student_agg = aggregate_metrics(student_metrics)
        teacher_agg = aggregate_metrics(teacher_metrics)
        
        # Compute ratios
        ratios = {}
        for key in student_agg:
            if key in teacher_agg and teacher_agg[key] > 0:
                ratios[key] = student_agg[key] / teacher_agg[key]
        
        return {
            'student': student_agg,
            'teacher': teacher_agg,
            'ratios': ratios
        }
    
    def get_sota_baselines(self) -> Dict:
        """Get SOTA baseline results from literature.
        These are placeholder values - replace with actual published results."""
        return {
            'LLaVA-3D-7B': {
                'bleu-1': 0.85,
                'bleu-4': 0.72,
                'rouge-1': 0.88,
                'rouge-l': 0.85,
                'meteor': 0.82,
                'params': 7_000_000_000,
                'fps': 0.01
            },
            'LLaVA-1.5': {
                'bleu-1': 0.78,
                'bleu-4': 0.65,
                'rouge-1': 0.82,
                'rouge-l': 0.79,
                'meteor': 0.75,
                'params': 7_000_000_000,
                'fps': 0.02
            },
            '3D-LLM': {
                'bleu-1': 0.72,
                'bleu-4': 0.58,
                'rouge-1': 0.76,
                'rouge-l': 0.73,
                'meteor': 0.70,
                'params': 13_000_000_000,
                'fps': 0.005
            }
        }
    
    def compare_with_sota(self, student_results: Dict) -> Dict:
        """Compare student with SOTA baselines."""
        logger.info("Comparing with SOTA baselines...")
        
        sota_baselines = self.get_sota_baselines()
        comparisons = {}
        
        for method_name, baseline_metrics in sota_baselines.items():
            comparison = {}
            for metric in ['bleu-1', 'bleu-4', 'rouge-1', 'rouge-l', 'meteor']:
                if metric in student_results and metric in baseline_metrics:
                    student_val = student_results[metric]
                    baseline_val = baseline_metrics[metric]
                    comparison[metric] = {
                        'student': student_val,
                        'baseline': baseline_val,
                        'ratio': student_val / baseline_val if baseline_val > 0 else 0
                    }
            
            # Efficiency comparison
            student_params = sum(p.numel() for p in self.student_model.parameters())
            baseline_params = baseline_metrics.get('params', 0)
            comparison['efficiency'] = {
                'student_params': student_params,
                'baseline_params': baseline_params,
                'compression_ratio': baseline_params / student_params if student_params > 0 else 0
            }
            
            comparisons[method_name] = comparison
        
        return comparisons
    
    def generate_comparison_table(self, results: Dict) -> str:
        """Generate LaTeX table for paper."""
        table = []
        table.append("\\begin{table}[h]")
        table.append("\\centering")
        table.append("\\caption{Comparison with Baselines}")
        table.append("\\begin{tabular}{lcccc}")
        table.append("\\hline")
        table.append("Method & BLEU-1 & ROUGE-1 & METEOR & Params (B) \\\\")
        table.append("\\hline")
        
        # Student
        student = results.get('student', {})
        table.append(f"Student (Ours) & {student.get('bleu-1', 0):.3f} & {student.get('rouge-1', 0):.3f} & {student.get('meteor', 0):.3f} & {sum(p.numel() for p in self.student_model.parameters()) / 1e9:.2f} \\\\")
        
        # Teacher
        teacher = results.get('teacher', {})
        table.append(f"Teacher (LLaVA-3D) & {teacher.get('bleu-1', 0):.3f} & {teacher.get('rouge-1', 0):.3f} & {teacher.get('meteor', 0):.3f} & 7.00 \\\\")
        
        # SOTA
        sota = results.get('sota_comparison', {})
        for method_name, comp in sota.items():
            baseline = comp.get('bleu-1', {}).get('baseline', 0)
            table.append(f"{method_name} & {baseline:.3f} & - & - & - \\\\")
        
        table.append("\\hline")
        table.append("\\end{tabular}")
        table.append("\\end{table}")
        
        return '\n'.join(table)


def main():
    parser = argparse.ArgumentParser(description='Baseline comparison')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Student model checkpoint')
    parser.add_argument('--num_samples', type=int, default=50,
                       help='Number of samples for evaluation')
    parser.add_argument('--output', type=str, default='results/baseline_comparison.json',
                       help='Output JSON file')
    
    args = parser.parse_args()
    
    # Load samples (simplified - you may want to load from actual dataset)
    samples = []
    data_root = Path("/home/alasfour/scratch/distilled-llava3d/data")
    for dataset_dir in ['scannet', '3d_front']:
        dataset_path = data_root / dataset_dir
        if dataset_path.exists():
            for scene_dir in dataset_path.iterdir():
                if scene_dir.is_dir():
                    images_dir = scene_dir / 'images'
                    if images_dir.exists():
                        for img_file in list(images_dir.glob('*.jpg'))[:2]:
                            if len(samples) >= args.num_samples:
                                break
                            samples.append({'image_path': str(img_file)})
                if len(samples) >= args.num_samples:
                    break
        if len(samples) >= args.num_samples:
            break
    
    # Run comparison
    comparator = BaselineComparator(args.checkpoint)
    
    # Student vs Teacher
    student_teacher = comparator.compare_student_vs_teacher(samples)
    
    # Student vs SOTA
    sota_comparison = comparator.compare_with_sota(student_teacher['student'])
    
    # Combine results
    results = {
        'student_vs_teacher': student_teacher,
        'sota_comparison': sota_comparison,
        'num_samples': len(samples)
    }
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate table
    table = comparator.generate_comparison_table(results)
    table_path = output_path.parent / 'baseline_comparison_table.tex'
    with open(table_path, 'w') as f:
        f.write(table)
    
    logger.info(f"✅ Results saved to {output_path}")
    logger.info(f"✅ LaTeX table saved to {table_path}")


if __name__ == "__main__":
    main()


