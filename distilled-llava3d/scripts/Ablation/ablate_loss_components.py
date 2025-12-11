#!/usr/bin/env python3
"""
Ablation Study: Multi-task Loss Components
Tests impact of each loss component by removing it
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

from fixed_training_pipeline import FixedTrainingPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LossComponentAblation:
    """Ablation study for loss components."""
    
    def __init__(self, data_root: str = "/home/alasfour/scratch/distilled-llava3d/data",
                 checkpoint_dir: str = "/home/alasfour/scratch/distilled-llava3d/checkpoints/ablation"):
        self.data_root = Path(data_root)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {}
    
    def train_with_loss_config(self, loss_config: Dict[str, float], name: str, epochs: int = 20) -> Dict:
        """
        Train model with specific loss configuration.
        
        Args:
            loss_config: Dictionary of loss weights (lambda_*)
            name: Name for this configuration
            epochs: Number of epochs
            
        Returns:
            Training results dictionary
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Training with loss config: {name}")
        logger.info(f"Loss weights: {loss_config}")
        logger.info(f"{'='*60}")
        
        # Create training pipeline
        pipeline = FixedTrainingPipeline(
            data_root=str(self.data_root),
            checkpoint_dir=str(self.checkpoint_dir / name)
        )
        
        # Override epochs
        pipeline.epochs = epochs
        pipeline.validation_split = 0.2
        
        # Set loss weights
        pipeline.lambda_det = loss_config.get('lambda_det', 0.35)
        pipeline.lambda_depth_ce = loss_config.get('lambda_depth_ce', 0.25)
        pipeline.lambda_depth_reg = loss_config.get('lambda_depth_reg', 0.15)
        pipeline.lambda_depth_kl = loss_config.get('lambda_depth_kl', 0.0125)
        pipeline.lambda_spatial = loss_config.get('lambda_spatial', 0.25)
        pipeline.lambda_text = loss_config.get('lambda_text', 0.0)
        pipeline.lambda_mv = loss_config.get('lambda_mv', 0.1)
        pipeline.lambda_feat = loss_config.get('lambda_feat', 0.3)
        
        # Train
        start_time = time.time()
        pipeline.train()
        training_time = time.time() - start_time
        
        results = {
            'config_name': name,
            'loss_config': loss_config,
            'training_time': training_time,
            'best_train_loss': pipeline.training_stats.get('best_loss', float('inf')),
            'best_val_loss': pipeline.training_stats.get('best_val_loss', float('inf')),
            'epochs_completed': pipeline.training_stats.get('epochs_completed', 0),
            'checkpoint_path': str(pipeline.checkpoint_dir / "fixed_model_best.pt")
        }
        
        return results
    
    def run_ablation(self, epochs: int = 20):
        """Run ablation study for all loss components."""
        logger.info("="*60)
        logger.info("Loss Component Ablation Study")
        logger.info("="*60)
        logger.info(f"Epochs per configuration: {epochs}")
        logger.info("")
        
        # Baseline: All components enabled
        baseline_config = {
            'lambda_det': 0.35,
            'lambda_depth_ce': 0.25,
            'lambda_depth_reg': 0.15,
            'lambda_depth_kl': 0.0125,
            'lambda_spatial': 0.25,
            'lambda_text': 0.0,  # Currently disabled
            'lambda_mv': 0.1,
            'lambda_feat': 0.3
        }
        
        # Ablation configurations: Remove one component at a time
        ablation_configs = {
            'baseline': baseline_config,
            'no_detection': {**baseline_config, 'lambda_det': 0.0},
            'no_depth': {
                **baseline_config,
                'lambda_depth_ce': 0.0,
                'lambda_depth_reg': 0.0,
                'lambda_depth_kl': 0.0
            },
            'no_spatial': {**baseline_config, 'lambda_spatial': 0.0},
            'no_multiview': {**baseline_config, 'lambda_mv': 0.0},
            'no_feature_distill': {**baseline_config, 'lambda_feat': 0.0},
            'only_feature_distill': {
                'lambda_det': 0.0,
                'lambda_depth_ce': 0.0,
                'lambda_depth_reg': 0.0,
                'lambda_depth_kl': 0.0,
                'lambda_spatial': 0.0,
                'lambda_text': 0.0,
                'lambda_mv': 0.0,
                'lambda_feat': 1.0
            }
        }
        
        results = {}
        
        for config_name, loss_config in ablation_configs.items():
            try:
                result = self.train_with_loss_config(loss_config, config_name, epochs=epochs)
                results[config_name] = result
                
                # Save intermediate results
                results_path = self.checkpoint_dir / "loss_component_ablation_results.json"
                with open(results_path, 'w') as f:
                    json.dump(results, f, indent=2)
                logger.info(f"💾 Intermediate results saved to {results_path}")
                
            except Exception as e:
                logger.error(f"❌ Failed to train with {config_name}: {e}")
                results[config_name] = {'error': str(e)}
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("LOSS COMPONENT ABLATION SUMMARY")
        logger.info("="*60)
        
        baseline_val_loss = results.get('baseline', {}).get('best_val_loss', float('inf'))
        
        for config_name, result in results.items():
            if 'error' in result:
                logger.info(f"\n{config_name.upper()}: ERROR - {result['error']}")
            else:
                val_loss = result.get('best_val_loss', float('inf'))
                if config_name != 'baseline' and baseline_val_loss != float('inf'):
                    delta = val_loss - baseline_val_loss
                    delta_pct = (delta / baseline_val_loss) * 100 if baseline_val_loss > 0 else 0
                    logger.info(f"\n{config_name.upper()}:")
                    logger.info(f"  Best Val Loss: {val_loss:.6f} (Δ {delta:+.6f}, {delta_pct:+.2f}%)")
                else:
                    logger.info(f"\n{config_name.upper()}:")
                    logger.info(f"  Best Val Loss: {val_loss:.6f}")
                logger.info(f"  Training Time: {result.get('training_time', 0):.2f}s")
        
        # Save final results
        results_path = self.checkpoint_dir / "loss_component_ablation_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\n💾 Final results saved to {results_path}")
        
        return results


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Loss component ablation study")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of epochs per configuration")
    parser.add_argument("--data_root", type=str,
                        default="/home/alasfour/scratch/distilled-llava3d/data",
                        help="Data root directory")
    
    args = parser.parse_args()
    
    ablation = LossComponentAblation(data_root=args.data_root)
    results = ablation.run_ablation(epochs=args.epochs)


if __name__ == "__main__":
    main()

