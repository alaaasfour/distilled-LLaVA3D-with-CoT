#!/usr/bin/env python3
"""
Qualitative Evaluation with Success/Failure Case Analysis
Identifies and analyzes cases where the student model succeeds or fails
compared to the teacher model.
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
from scripts.evaluation.metrics import compute_text_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QualitativeEvaluator:
    """
    Qualitative evaluation that identifies success and failure cases.
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
    
    def evaluate_samples(self, samples: List[Dict], num_cases: int = 10) -> Dict:
        """
        Evaluate samples and identify success/failure cases.
        
        Args:
            samples: List of evaluation samples
            num_cases: Number of success/failure cases to identify
        
        Returns:
            Dictionary with success and failure cases
        """
        logger.info(f"Evaluating {len(samples)} samples for qualitative analysis...")
        
        all_cases = []
        
        for i, sample in enumerate(samples):
            try:
                img_path = Path(sample["image_path"])
                if not img_path.exists():
                    continue
                
                image = Image.open(img_path).convert('RGB')
                image_tensor = self.transform(image).unsqueeze(0).to(self.device).float()
                
                question = sample.get("question", "Describe this 3D scene and identify objects.")
                
                # Get student response
                with torch.no_grad():
                    student_response = self.student_model.generate_response(question, image_tensor)
                    if isinstance(student_response, dict):
                        student_text = student_response.get('response', str(student_response))
                    else:
                        student_text = str(student_response)
                
                # Get teacher response
                teacher_response_dict = self.teacher_model.generate_response(question, str(img_path))
                if isinstance(teacher_response_dict, dict):
                    teacher_text = teacher_response_dict.get('response', str(teacher_response_dict))
                else:
                    teacher_text = str(teacher_response_dict)
                
                # Compute metrics
                metrics = compute_text_metrics(teacher_text, student_text)
                
                # Compute similarity score (average of BLEU-1, ROUGE-1, METEOR)
                similarity_score = (
                    metrics.get('bleu-1', 0.0) +
                    metrics.get('rouge-1', 0.0) +
                    metrics.get('meteor', 0.0)
                ) / 3.0
                
                all_cases.append({
                    'image_path': str(img_path),
                    'scene': sample.get('scene', 'unknown'),
                    'dataset': sample.get('dataset', 'unknown'),
                    'question': question,
                    'student_response': student_text,
                    'teacher_response': teacher_text,
                    'metrics': metrics,
                    'similarity_score': similarity_score,
                    'bleu_1': metrics.get('bleu-1', 0.0),
                    'rouge_1': metrics.get('rouge-1', 0.0),
                    'meteor': metrics.get('meteor', 0.0)
                })
                
                if (i + 1) % 10 == 0:
                    logger.info(f"  Processed {i + 1}/{len(samples)} samples")
                    
            except Exception as e:
                logger.warning(f"Error processing sample {i}: {e}")
                continue
        
        # Sort by similarity score
        all_cases.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Identify success and failure cases
        # Success: top cases (high similarity)
        # Failure: bottom cases (low similarity)
        success_cases = all_cases[:num_cases]
        failure_cases = all_cases[-num_cases:] if len(all_cases) >= num_cases else []
        
        # Analyze failure patterns
        failure_analysis = self._analyze_failures(failure_cases)
        
        return {
            'total_samples': len(all_cases),
            'success_cases': success_cases,
            'failure_cases': failure_cases,
            'failure_analysis': failure_analysis,
            'average_similarity': np.mean([c['similarity_score'] for c in all_cases]) if all_cases else 0.0,
            'median_similarity': np.median([c['similarity_score'] for c in all_cases]) if all_cases else 0.0
        }
    
    def _analyze_failures(self, failure_cases: List[Dict]) -> Dict:
        """
        Analyze failure patterns to identify common issues.
        """
        if not failure_cases:
            return {}
        
        analysis = {
            'common_issues': [],
            'response_length_analysis': {
                'student_avg_length': np.mean([len(c['student_response'].split()) for c in failure_cases]),
                'teacher_avg_length': np.mean([len(c['teacher_response'].split()) for c in failure_cases])
            },
            'missing_keywords': [],
            'hallucination_keywords': []
        }
        
        # Identify common missing keywords in student responses
        teacher_keywords = set()
        student_keywords = set()
        
        for case in failure_cases:
            teacher_words = set(case['teacher_response'].lower().split())
            student_words = set(case['student_response'].lower().split())
            teacher_keywords.update(teacher_words)
            student_keywords.update(student_words)
        
        missing_keywords = teacher_keywords - student_keywords
        analysis['missing_keywords'] = list(missing_keywords)[:20]  # Top 20
        
        # Identify potential hallucinations (words in student but not in teacher)
        hallucination_keywords = student_keywords - teacher_keywords
        analysis['hallucination_keywords'] = list(hallucination_keywords)[:20]
        
        # Common issues
        if analysis['response_length_analysis']['student_avg_length'] < \
           analysis['response_length_analysis']['teacher_avg_length'] * 0.5:
            analysis['common_issues'].append("Student responses are too short")
        
        if len(missing_keywords) > len(teacher_keywords) * 0.3:
            analysis['common_issues'].append("Student missing many important keywords")
        
        return analysis


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Qualitative Evaluation with Success/Failure Cases')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to student model checkpoint')
    parser.add_argument('--data_root', type=str, default='/home/alasfour/scratch/distilled-llava3d/data',
                       help='Root directory for evaluation data')
    parser.add_argument('--num_samples', type=int, default=50,
                       help='Number of samples to evaluate')
    parser.add_argument('--num_cases', type=int, default=10,
                       help='Number of success/failure cases to identify')
    parser.add_argument('--output', type=str, default='results/qualitative_evaluation.json',
                       help='Output JSON file')
    
    args = parser.parse_args()
    
    # Load samples
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
            for img_file in image_files[:2]:
                if len(samples) >= args.num_samples:
                    break
                samples.append({
                    'image_path': str(img_file),
                    'scene': scene_dir.name,
                    'dataset': dataset_dir,
                    'question': "Describe this 3D scene and identify objects."
                })
            if len(samples) >= args.num_samples:
                break
        if len(samples) >= args.num_samples:
            break
    
    logger.info(f"✅ Loaded {len(samples)} evaluation samples")
    
    # Run evaluation
    evaluator = QualitativeEvaluator(
        student_checkpoint=args.checkpoint,
        device='cuda',
        teacher_device='cpu'
    )
    
    results = evaluator.evaluate_samples(samples, num_cases=args.num_cases)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("QUALITATIVE EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total Samples: {results['total_samples']}")
    logger.info(f"Average Similarity: {results['average_similarity']:.4f}")
    logger.info(f"Median Similarity: {results['median_similarity']:.4f}")
    logger.info(f"\n✅ Success Cases: {len(results['success_cases'])}")
    logger.info(f"❌ Failure Cases: {len(results['failure_cases'])}")
    
    if results['failure_analysis']:
        logger.info("\n📊 Failure Analysis:")
        logger.info(f"  Common Issues: {results['failure_analysis'].get('common_issues', [])}")
        logger.info(f"  Student Avg Response Length: {results['failure_analysis']['response_length_analysis']['student_avg_length']:.1f} words")
        logger.info(f"  Teacher Avg Response Length: {results['failure_analysis']['response_length_analysis']['teacher_avg_length']:.1f} words")
    
    logger.info(f"\n💾 Results saved to {output_path}")


if __name__ == "__main__":
    main()




