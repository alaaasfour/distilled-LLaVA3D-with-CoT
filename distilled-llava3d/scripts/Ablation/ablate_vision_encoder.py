#!/usr/bin/env python3
"""
Ablation Study: Vision Encoder Impact
Compares VGGT vs CLIP ViT vs ResNet vs Custom CNN
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import json
import time
from pathlib import Path
from typing import Dict, List
import logging

from scripts.distillation.student_model import DistilledLLaVA3D, DistilledLLaVA3DConfig
from fixed_training_pipeline import FixedTrainingPipeline
from scripts.evaluation.metrics import compute_text_metrics, aggregate_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VisionEncoderAblation:
    """Ablation study for different vision encoders."""
    
    def __init__(self, data_root: str = "/home/alasfour/scratch/distilled-llava3d/data",
                 checkpoint_dir: str = "/home/alasfour/scratch/distilled-llava3d/checkpoints/ablation"):
        self.data_root = Path(data_root)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {}
    
    def train_with_encoder(self, encoder_type: str, epochs: int = 20) -> Dict:
        """
        Train model with specific vision encoder.
        
        Args:
            encoder_type: 'vggt', 'clip', 'resnet', 'cnn'
            epochs: Number of epochs to train
            
        Returns:
            Training results dictionary
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Training with {encoder_type.upper()} encoder")
        logger.info(f"{'='*60}")
        
        # Create config with specific encoder
        config = DistilledLLaVA3DConfig()
        
        # Modify config based on encoder type
        if encoder_type == 'vggt':
            # Use VGGT (default)
            config.vggt_device = 'cpu'  # CPU offloading
        elif encoder_type == 'clip':
            # TODO: Implement CLIP encoder option
            logger.warning("CLIP encoder not yet implemented, using VGGT")
            encoder_type = 'vggt'
        elif encoder_type == 'resnet':
            # TODO: Implement ResNet encoder option
            logger.warning("ResNet encoder not yet implemented, using VGGT")
            encoder_type = 'vggt'
        elif encoder_type == 'cnn':
            # Use fallback CNN (if VGGT fails to load)
            # This requires modifying student_model.py to force CNN
            logger.warning("CNN encoder requires code modification, using VGGT")
            encoder_type = 'vggt'
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
        
        # Create training pipeline
        pipeline = FixedTrainingPipeline(
            data_root=str(self.data_root),
            checkpoint_dir=str(self.checkpoint_dir / encoder_type)
        )
        
        # Override epochs for faster ablation
        pipeline.epochs = epochs
        pipeline.validation_split = 0.2
        
        # Train
        start_time = time.time()
        pipeline.train()
        training_time = time.time() - start_time
        
        # Get best checkpoint
        best_checkpoint = pipeline.checkpoint_dir / "fixed_model_best.pt"
        
        # Evaluate on validation set
        logger.info(f"Evaluating {encoder_type} encoder...")
        val_metrics = self.evaluate_checkpoint(best_checkpoint, encoder_type)
        
        results = {
            'encoder_type': encoder_type,
            'training_time': training_time,
            'best_train_loss': pipeline.training_stats.get('best_loss', float('inf')),
            'best_val_loss': pipeline.training_stats.get('best_val_loss', float('inf')),
            'epochs_completed': pipeline.training_stats.get('epochs_completed', 0),
            'validation_metrics': val_metrics,
            'checkpoint_path': str(best_checkpoint)
        }
        
        return results
    
    def evaluate_checkpoint(self, checkpoint_path: Path, encoder_type: str, num_samples: int = 50) -> Dict:
        """Evaluate checkpoint on validation samples."""
        from real_llava3d_teacher import RealLLaVA3DTeacher
        from PIL import Image
        import torchvision.transforms as transforms
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = DistilledLLaVA3DConfig()
        model = DistilledLLaVA3D(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        # Load teacher for reference
        try:
            teacher = RealLLaVA3DTeacher(device="cpu")
        except:
            logger.warning("Teacher not available, skipping evaluation")
            return {}
        
        # Load validation samples
        pipeline = FixedTrainingPipeline()
        _, val_samples = pipeline.load_expanded_datasets()
        val_samples = val_samples[:num_samples]  # Limit for speed
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        # Evaluate
        text_metrics_list = []
        
        for i, sample in enumerate(val_samples):
            if i % 10 == 0:
                logger.info(f"  Evaluating sample {i+1}/{len(val_samples)}")
            
            try:
                img_path = Path(sample["image_path"])
                if not img_path.exists():
                    continue
                
                image = Image.open(img_path).convert('RGB')
                image_tensor = transform(image).unsqueeze(0).to(device).float()
                
                question = "Describe this 3D scene and identify objects."
                
                # Get student response
                with torch.no_grad():
                    student_response = model.generate_response(question, image_tensor)
                
                # Get teacher response
                teacher_response_dict = teacher.generate_response(question, str(img_path))
                
                # Extract text
                if isinstance(student_response, dict):
                    student_text = student_response.get('response', str(student_response))
                else:
                    student_text = str(student_response)
                
                if isinstance(teacher_response_dict, dict):
                    teacher_text = teacher_response_dict.get('response', str(teacher_response_dict))
                else:
                    teacher_text = str(teacher_response_dict)
                
                # Compute metrics
                metrics = compute_text_metrics(teacher_text, student_text)
                text_metrics_list.append(metrics)
                
            except Exception as e:
                logger.warning(f"  Error evaluating sample {i}: {e}")
                continue
        
        # Aggregate
        if text_metrics_list:
            aggregated = aggregate_metrics(text_metrics_list)
            return aggregated
        else:
            return {}
    
    def run_ablation(self, encoders: List[str] = None, epochs: int = 20):
        """Run ablation study for all encoders."""
        if encoders is None:
            encoders = ['vggt']  # Start with VGGT only for now
        
        logger.info("="*60)
        logger.info("Vision Encoder Ablation Study")
        logger.info("="*60)
        logger.info(f"Encoders to test: {encoders}")
        logger.info(f"Epochs per encoder: {epochs}")
        logger.info("")
        
        results = {}
        
        for encoder in encoders:
            try:
                result = self.train_with_encoder(encoder, epochs=epochs)
                results[encoder] = result
                
                # Save intermediate results
                results_path = self.checkpoint_dir / "vision_encoder_ablation_results.json"
                with open(results_path, 'w') as f:
                    json.dump(results, f, indent=2)
                logger.info(f"💾 Intermediate results saved to {results_path}")
                
            except Exception as e:
                logger.error(f"❌ Failed to train with {encoder}: {e}")
                results[encoder] = {'error': str(e)}
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("ABLATION STUDY SUMMARY")
        logger.info("="*60)
        
        for encoder, result in results.items():
            if 'error' in result:
                logger.info(f"\n{encoder.upper()}: ERROR - {result['error']}")
            else:
                logger.info(f"\n{encoder.upper()}:")
                logger.info(f"  Best Val Loss: {result.get('best_val_loss', 'N/A'):.6f}")
                logger.info(f"  Training Time: {result.get('training_time', 0):.2f}s")
                if 'validation_metrics' in result:
                    metrics = result['validation_metrics']
                    logger.info(f"  BLEU-4: {metrics.get('bleu-4', 0):.4f}")
                    logger.info(f"  ROUGE-L: {metrics.get('rouge-l', 0):.4f}")
                    logger.info(f"  METEOR: {metrics.get('meteor', 0):.4f}")
        
        # Save final results
        results_path = self.checkpoint_dir / "vision_encoder_ablation_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\n💾 Final results saved to {results_path}")
        
        return results


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Vision encoder ablation study")
    parser.add_argument("--encoders", nargs="+", default=['vggt'],
                        choices=['vggt', 'clip', 'resnet', 'cnn'],
                        help="Encoders to test")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of epochs per encoder")
    parser.add_argument("--data_root", type=str,
                        default="/home/alasfour/scratch/distilled-llava3d/data",
                        help="Data root directory")
    
    args = parser.parse_args()
    
    ablation = VisionEncoderAblation(data_root=args.data_root)
    results = ablation.run_ablation(encoders=args.encoders, epochs=args.epochs)


if __name__ == "__main__":
    main()

