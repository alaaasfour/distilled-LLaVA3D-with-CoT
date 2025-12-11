#!/usr/bin/env python3
"""
Ablation Study: Loss Weight Sensitivity
Tests different combinations of loss weights
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
import itertools

from fixed_training_pipeline import FixedTrainingPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LossWeightAblation:
    """Ablation study for loss weights."""
    
    def __init__(self, data_root: str = "/home/alasfour/scratch/distilled-llava3d/data",
                 checkpoint_dir: str = "/home/alasfour/scratch/distilled-llava3d/checkpoints/ablation"):
        self.data_root = Path(data_root)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {}
    
    def train_with_weights(self, weights: Dict[str, float], name: str, epochs: int = 15) -> Dict:
        """Train model with specific loss weights."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Training with weights: {name}")
        logger.info(f"Weights: {weights}")
        logger.info(f"{'='*60}")
        
        pipeline = FixedTrainingPipeline(
            data_root=str(self.data_root),
            checkpoint_dir=str(self.checkpoint_dir / name)
        )
        
        pipeline.epochs = epochs
        pipeline.validation_split = 0.2
        
        # Set weights
        pipeline.lambda_det = weights.get('lambda_det', 0.35)
        pipeline.lambda_depth_ce = weights.get('lambda_depth_ce', 0.25)
        pipeline.lambda_depth_reg = weights.get('lambda_depth_reg', 0.15)
        pipeline.lambda_depth_kl = weights.get('lambda_depth_kl', 0.0125)
        pipeline.lambda_spatial = weights.get('lambda_spatial', 0.25)
        pipeline.lambda_text = weights.get('lambda_text', 0.0)
        pipeline.lambda_mv = weights.get('lambda_mv', 0.1)
        pipeline.lambda_feat = weights.get('lambda_feat', 0.3)
        
        start_time = time.time()
        pipeline.train()
        training_time = time.time() - start_time
        
        results = {
            'config_name': name,
            'weights': weights,
            'training_time': training_time,
            'best_train_loss': pipeline.training_stats.get('best_loss', float('inf')),
            'best_val_loss': pipeline.training_stats.get('best_val_loss', float('inf')),
            'epochs_completed': pipeline.training_stats.get('epochs_completed', 0),
            'checkpoint_path': str(pipeline.checkpoint_dir / "fixed_model_best.pt")
        }
        
        return results
    
    def run_ablation(self, epochs: int = 15):
        """Run loss weight sensitivity analysis."""
        logger.info("="*60)
        logger.info("Loss Weight Sensitivity Analysis")
        logger.info("="*60)
        logger.info(f"Epochs per configuration: {epochs}")
        logger.info("")
        
        # Baseline weights
        baseline = {
            'lambda_det': 0.35,
            'lambda_depth_ce': 0.25,
            'lambda_depth_reg': 0.15,
            'lambda_depth_kl': 0.0125,
            'lambda_spatial': 0.25,
            'lambda_text': 0.0,
            'lambda_mv': 0.1,
            'lambda_feat': 0.3
        }
        
        # Test different weight combinations
        # Focus on main components: detection, depth, spatial, feature distillation
        weight_configs = {
            'baseline': baseline,
            # Increase detection weight
            'high_det': {**baseline, 'lambda_det': 0.5, 'lambda_feat': 0.2},
            'low_det': {**baseline, 'lambda_det': 0.2, 'lambda_feat': 0.4},
            # Increase depth weight
            'high_depth': {
                **baseline,
                'lambda_depth_ce': 0.35,
                'lambda_depth_reg': 0.2,
                'lambda_feat': 0.2
            },
            'low_depth': {
                **baseline,
                'lambda_depth_ce': 0.15,
                'lambda_depth_reg': 0.1,
                'lambda_feat': 0.4
            },
            # Increase spatial weight
            'high_spatial': {**baseline, 'lambda_spatial': 0.4, 'lambda_feat': 0.2},
            'low_spatial': {**baseline, 'lambda_spatial': 0.1, 'lambda_feat': 0.4},
            # Increase feature distillation
            'high_feat': {**baseline, 'lambda_feat': 0.5, 'lambda_det': 0.25},
            'low_feat': {**baseline, 'lambda_feat': 0.1, 'lambda_det': 0.45},
            # Balanced (equal weights for main components)
            'balanced': {
                'lambda_det': 0.25,
                'lambda_depth_ce': 0.25,
                'lambda_depth_reg': 0.15,
                'lambda_depth_kl': 0.0125,
                'lambda_spatial': 0.25,
                'lambda_text': 0.0,
                'lambda_mv': 0.1,
                'lambda_feat': 0.25
            }
        }
        
        results = {}
        
        for config_name, weights in weight_configs.items():
            try:
                result = self.train_with_weights(weights, config_name, epochs=epochs)
                results[config_name] = result
                
                # Save intermediate results
                results_path = self.checkpoint_dir / "loss_weight_ablation_results.json"
                with open(results_path, 'w') as f:
                    json.dump(results, f, indent=2)
                logger.info(f"💾 Intermediate results saved to {results_path}")
                
            except Exception as e:
                logger.error(f"❌ Failed to train with {config_name}: {e}")
                results[config_name] = {'error': str(e)}
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("LOSS WEIGHT SENSITIVITY SUMMARY")
        logger.info("="*60)
        
        # Find best configuration
        best_config = None
        best_val_loss = float('inf')
        
        for config_name, result in results.items():
            if 'error' not in result:
                val_loss = result.get('best_val_loss', float('inf'))
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_config = config_name
        
        logger.info(f"\nBest Configuration: {best_config} (Val Loss: {best_val_loss:.6f})")
        logger.info("\nAll Configurations:")
        
        for config_name, result in results.items():
            if 'error' in result:
                logger.info(f"\n{config_name.upper()}: ERROR - {result['error']}")
            else:
                val_loss = result.get('best_val_loss', float('inf'))
                marker = " ⭐ BEST" if config_name == best_config else ""
                logger.info(f"\n{config_name.upper()}:{marker}")
                logger.info(f"  Best Val Loss: {val_loss:.6f}")
                logger.info(f"  Training Time: {result.get('training_time', 0):.2f}s")
                if 'weights' in result:
                    logger.info(f"  Weights: {result['weights']}")
        
        # Save final results
        results_path = self.checkpoint_dir / "loss_weight_ablation_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\n💾 Final results saved to {results_path}")
        
        return results


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Loss weight sensitivity analysis")
    parser.add_argument("--epochs", type=int, default=15,
                        help="Number of epochs per configuration")
    parser.add_argument("--data_root", type=str,
                        default="/home/alasfour/scratch/distilled-llava3d/data",
                        help="Data root directory")
    
    args = parser.parse_args()
    
    ablation = LossWeightAblation(data_root=args.data_root)
    results = ablation.run_ablation(epochs=args.epochs)


if __name__ == "__main__":
    main()

