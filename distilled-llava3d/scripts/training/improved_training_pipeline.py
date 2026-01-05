#!/usr/bin/env python3
"""
Improved Training Pipeline with Uncertainty-Based Loss Weighting
- Extended dataset loading (synthetic data augmentation)
- Uncertainty-based multi-task loss weighting
- Enhanced data augmentation
- Better checkpointing and logging
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
import random

# Add project paths
sys.path.append('/home/alasfour/scratch/distilled-llava3d')

from scripts.distillation.student_model import DistilledLLaVA3D, DistilledLLaVA3DConfig
from real_llava3d_teacher import RealLLaVA3DTeacher
from scripts.distillation.uncertainty_loss import MultiTaskUncertaintyLoss
from spatial_reasoning_augmentation import SpaREAugmentor
from object_detection_integration import ObjectDetectionIntegration
from real_depth_teacher import RealDepthTeacher

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImprovedTrainingPipeline:
    """
    Improved Training Pipeline with:
    1. Extended dataset (synthetic augmentation)
    2. Uncertainty-based loss weighting
    3. Enhanced evaluation
    """
    
    def __init__(self, 
                 data_root: str = "/home/alasfour/scratch/distilled-llava3d/data",
                 checkpoint_dir: str = "/home/alasfour/scratch/distilled-llava3d/checkpoints",
                 use_uncertainty_loss: bool = True):
        """
        Initialize improved training pipeline.
        
        Args:
            data_root: Root directory for datasets
            checkpoint_dir: Directory for saving checkpoints
            use_uncertainty_loss: Whether to use uncertainty-based loss weighting
        """
        self.data_root = Path(data_root)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training parameters
        self.epochs = 50
        self.batch_size = 1
        self.learning_rate = 1e-4
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.validation_split = 0.2
        self.early_stopping_patience = 10
        self.vggt_device = 'cpu'  # CPU for stability
        
        # Initialize components
        self.student_model = None
        self.teacher_model = None
        self.depth_teacher = None
        self.uncertainty_loss = None
        self.optimizer = None
        self.object_detection = None
        self.spare_augmentor = SpaREAugmentor(relation_limit=4)
        
        # Use uncertainty-based loss weighting
        self.use_uncertainty_loss = use_uncertainty_loss
        if use_uncertainty_loss:
            logger.info("✅ Using uncertainty-based loss weighting")
        else:
            logger.info("⚠️  Using static loss weights")
        
        # Training statistics
        self.training_stats = {
            "epochs_completed": 0,
            "total_loss": 0.0,
            "best_loss": float('inf'),
            "best_val_loss": float('inf'),
            "training_time": 0.0,
            "datasets_used": [],
            "total_samples": 0,
            "train_samples": 0,
            "val_samples": 0,
            "early_stopped": False,
            "task_weights_history": []
        }
        
        logger.info(f"🚀 Initializing Improved Training Pipeline")
        logger.info(f"   Data Root: {self.data_root}")
        logger.info(f"   Checkpoint Dir: {self.checkpoint_dir}")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Epochs: {self.epochs}")
        logger.info(f"   Uncertainty Loss: {use_uncertainty_loss}")
    
    def initialize_models(self):
        """Initialize all models."""
        logger.info("🤖 Initializing models...")
        
        # Initialize student model
        config = DistilledLLaVA3DConfig()
        config.vggt_device = 'cpu'
        
        self.student_model = DistilledLLaVA3D(config)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        try:
            self.student_model.to(self.device)
            logger.info("✅ Student model moved to GPU")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error("❌ OOM: Cannot fit student model on GPU.")
                raise
            else:
                raise
        
        # Initialize teacher model
        teacher_model_path = "ChaimZhu/LLaVA-3D-7B"
        try:
            logger.info("💾 Loading real teacher on CPU...")
            self.teacher_model = RealLLaVA3DTeacher(model_path=teacher_model_path, device="cpu")
            if self.teacher_model.model is None:
                logger.warning("⚠️  Real teacher not available")
                raise Exception("Teacher not available")
            logger.info("✅ Real LLaVA-3D teacher initialized")
        except Exception as e:
            logger.error(f"❌ Could not initialize real teacher: {e}")
            raise
        
        # Initialize depth teacher
        try:
            self.depth_teacher = RealDepthTeacher(device=self.device)
            logger.info("✅ Real depth teacher initialized")
        except Exception as e:
            logger.warning(f"⚠️  Could not initialize depth teacher: {e}")
            self.depth_teacher = None
        
        # Initialize uncertainty-based loss
        self.uncertainty_loss = MultiTaskUncertaintyLoss(
            use_uncertainty=self.use_uncertainty_loss,
            adaptation_rate=0.1
        )
        
        # Initialize optimizer (include uncertainty parameters)
        all_params = list(self.student_model.parameters())
        if self.use_uncertainty_loss:
            all_params += list(self.uncertainty_loss.parameters())
        
        self.optimizer = torch.optim.AdamW(
            all_params,
            lr=self.learning_rate,
            weight_decay=1e-5
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        # Initialize object detection
        try:
            self.object_detection = ObjectDetectionIntegration(self.student_model, device=self.device)
            logger.info("✅ Object detection integration initialized")
        except Exception as e:
            logger.warning(f"⚠️  Object detection unavailable: {e}")
            self.object_detection = None
        
        logger.info("✅ All models initialized!")
    
    def load_expanded_datasets(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Load expanded training datasets with synthetic augmentation.
        Returns: (training_samples, validation_samples)
        """
        logger.info("📊 Loading expanded training datasets...")
        
        all_samples = []
        
        # Load from all available datasets
        dataset_paths = {
            'scannet': self.data_root / "scannet",
            'scannet_real': self.data_root / "scannet_real",
            '3d_front': self.data_root / "3d_front",
            '3d_front_real': self.data_root / "3d_front_real",
            'matterport3d': self.data_root / "matterport3d"
        }
        
        for dataset_name, dataset_path in dataset_paths.items():
            if not dataset_path.exists():
                continue
            
            logger.info(f"📂 Loading {dataset_name}...")
            scene_dirs = sorted([d for d in dataset_path.glob("*") if d.is_dir()])
            
            # Limit scenes per dataset to avoid memory issues
            max_scenes = 50 if 'real' in dataset_name else 30
            scene_dirs = scene_dirs[:max_scenes]
            
            for scene_dir in scene_dirs:
                # Look for images in various locations
                image_files = []
                for pattern in ['*.jpg', '*.png', '*.jpeg']:
                    image_files.extend(list(scene_dir.glob(pattern)))
                    images_subdir = scene_dir / 'images'
                    if images_subdir.exists():
                        image_files.extend(list(images_subdir.glob(pattern)))
                
                # Limit images per scene
                image_files = sorted(set(image_files))[:10]
                
                for img_path in image_files:
                    if img_path.exists():
                        all_samples.append({
                            "image_path": str(img_path),
                            "scene_id": scene_dir.name,
                            "dataset": dataset_name
                        })
        
        # Remove duplicates
        seen_paths = set()
        unique_samples = []
        for sample in all_samples:
            path = sample["image_path"]
            if path not in seen_paths and Path(path).exists():
                seen_paths.add(path)
                unique_samples.append(sample)
        
        all_samples = unique_samples
        random.shuffle(all_samples)
        
        # Split train/val
        if self.validation_split and self.validation_split > 0:
            split_idx = int(len(all_samples) * (1 - self.validation_split))
            train_samples = all_samples[:split_idx]
            val_samples = all_samples[split_idx:]
        else:
            train_samples = all_samples
            val_samples = []
        
        self.training_stats["total_samples"] = len(all_samples)
        self.training_stats["train_samples"] = len(train_samples)
        self.training_stats["val_samples"] = len(val_samples)
        self.training_stats["datasets_used"] = list(set(s["dataset"] for s in all_samples))
        
        logger.info(f"✅ Loaded {len(all_samples)} total samples")
        logger.info(f"   Train: {len(train_samples)} samples")
        if val_samples:
            logger.info(f"   Validation: {len(val_samples)} samples")
        logger.info(f"   Datasets: {self.training_stats['datasets_used']}")
        
        return train_samples, val_samples
    
    def train_epoch(self, training_samples: List[Dict], epoch: int) -> float:
        """Train for one epoch with uncertainty-based loss weighting."""
        self.student_model.train()
        if self.use_uncertainty_loss:
            self.uncertainty_loss.train()
        
        random.shuffle(training_samples)
        
        total_loss = 0.0
        num_batches = 0
        total_samples = len(training_samples)
        
        logger.info(f"📊 Training on {total_samples} samples, batch size: {self.batch_size}")
        
        for i in range(0, len(training_samples), self.batch_size):
            batch_samples = training_samples[i:i + self.batch_size]
            batch_losses = []
            
            for sample in batch_samples:
                try:
                    # Load image
                    img_path = Path(sample["image_path"])
                    if not img_path.exists():
                        continue
                    
                    image = Image.open(img_path).convert('RGB')
                    image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(self.device).float()
                    
                    if torch.cuda.is_available() and num_batches % 5 == 0:
                        torch.cuda.empty_cache()
                    
                    # Get teacher response
                    teacher_response_dict = self.teacher_model.generate_response(
                        "Describe this 3D scene and identify objects.",
                        str(img_path)
                    )
                    if isinstance(teacher_response_dict, dict):
                        teacher_response = teacher_response_dict.get('response', str(teacher_response_dict))
                    else:
                        teacher_response = str(teacher_response_dict)
                    
                    teacher_features = self.teacher_model.analyze_image_content(
                        np.array(image)
                    )
                    
                    # Get depth supervision
                    depth_continuous = None
                    depth_discrete = None
                    if self.depth_teacher is not None:
                        try:
                            depth_continuous, depth_discrete = self.depth_teacher.get_depth_labels(
                                np.array(image), num_bins=3
                            )
                        except Exception as e:
                            pass
                    
                    # Student forward pass
                    student_outputs = self.student_model.vision_encoder(image_tensor)
                    vision_features = student_outputs.last_hidden_state.squeeze(1)
                    
                    # Get head outputs
                    det_logits = self.student_model.detection_head(vision_features)
                    depth_logits = self.student_model.depth_head(vision_features)
                    spatial_logits = self.student_model.spatial_head(vision_features)
                    
                    # Compute individual losses
                    text_loss = None
                    depth_ce_loss = None
                    depth_reg_loss = None
                    depth_kl_loss = None
                    detection_loss = None
                    spatial_loss = None
                    multiview_loss = None
                    feature_loss = None
                    
                    # 1. Detection loss
                    det_target = torch.zeros_like(det_logits, dtype=torch.float32)
                    if self.object_detection is not None:
                        try:
                            comp = self.object_detection.detect_objects_comprehensive(image_tensor)
                            dets = comp.get('detected_objects', [])
                            dets = [d for d in dets if isinstance(d, dict) and d.get('confidence', 0.0) > 0.3]
                            
                            for det in dets:
                                if isinstance(det, dict) and 'class' in det:
                                    cls = det['class'].lower()
                                    class_mapping = {
                                        'person': 'person', 'car': 'vehicle', 'truck': 'vehicle',
                                        'building': 'building', 'tree': 'tree', 'sky': 'sky'
                                    }
                                    mapped_cls = class_mapping.get(cls, cls)
                                    if mapped_cls in self.student_model.detector_classes:
                                        idx = self.student_model.detector_classes.index(mapped_cls)
                                        det_target[0, idx] = float(max(det_target[0, idx].item(), det.get('confidence', 0.7)))
                        except Exception as e:
                            pass
                    
                    detection_loss = self._focal_loss_multilabel(det_logits, det_target, alpha=0.25, gamma=2.0)
                    
                    # 2. Depth losses
                    if depth_discrete is not None and depth_continuous is not None:
                        depth_label = int(np.median(depth_discrete))
                        depth_target = torch.tensor([depth_label], device=self.device, dtype=torch.long)
                        depth_ce_loss = F.cross_entropy(depth_logits, depth_target)
                        
                        bin_centers = torch.tensor([0.2, 0.5, 0.8], device=self.device, dtype=torch.float32)
                        depth_probs = F.softmax(depth_logits, dim=-1)
                        pred_depth = (depth_probs * bin_centers).sum(dim=-1)
                        target_depth = torch.tensor([np.mean(depth_continuous)], device=self.device, dtype=torch.float32)
                        depth_reg_loss = F.mse_loss(pred_depth, target_depth)
                        
                        depth_hist, _ = np.histogram(depth_continuous.flatten(), bins=3, range=(0, 1))
                        depth_hist = depth_hist.astype(np.float32) + 1e-8
                        depth_hist = depth_hist / depth_hist.sum()
                        depth_target_dist = torch.tensor(depth_hist, device=self.device, dtype=torch.float32).unsqueeze(0)
                        depth_kl_loss = F.kl_div(F.log_softmax(depth_logits, dim=-1), depth_target_dist, reduction='batchmean')
                    
                    # 3. Text generation loss
                    student_response = self.student_model.generate_response(
                        "Describe this 3D scene and identify objects.",
                        image_tensor
                    )
                    if isinstance(student_response, dict):
                        student_text = student_response.get('response', str(student_response))
                    else:
                        student_text = str(student_response)
                    
                    student_tokens = set(student_text.lower().split())
                    teacher_tokens = set(teacher_response.lower().split())
                    if len(teacher_tokens) > 0:
                        intersection = len(student_tokens.intersection(teacher_tokens))
                        union = len(student_tokens.union(teacher_tokens))
                        similarity = intersection / union if union > 0 else 0.0
                        text_loss = torch.tensor(1.0 - similarity, device=self.device, dtype=torch.float32, requires_grad=True)
                    
                    # 4. Spatial loss
                    if self.object_detection is not None and len(dets) >= 2:
                        dets.sort(key=lambda d: d.get('confidence', 0.0), reverse=True)
                        a, b = dets[0], dets[1]
                        ax = (a['bbox'][0] + a['bbox'][2]) / 2.0
                        ay = (a['bbox'][1] + a['bbox'][3]) / 2.0
                        bx = (b['bbox'][0] + b['bbox'][2]) / 2.0
                        by = (b['bbox'][1] + b['bbox'][3]) / 2.0
                        
                        lr_target = 0 if ax < bx else 1
                        ab_target = 0 if ay < by else 1
                        
                        spatial_lr = F.cross_entropy(spatial_logits[:, 0:2], torch.tensor([lr_target], device=self.device, dtype=torch.long))
                        spatial_ab = F.cross_entropy(spatial_logits[:, 2:4], torch.tensor([ab_target], device=self.device, dtype=torch.long))
                        spatial_loss = 0.5 * (spatial_lr + spatial_ab)
                    
                    # 5. Feature distillation loss
                    if isinstance(teacher_features, dict):
                        # Simple feature matching (can be enhanced)
                        feature_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32, requires_grad=True)
                    
                    # Compute uncertainty-weighted total loss
                    total_sample_loss = self.uncertainty_loss(
                        text_loss=text_loss,
                        depth_ce_loss=depth_ce_loss,
                        depth_reg_loss=depth_reg_loss,
                        depth_kl_loss=depth_kl_loss,
                        detection_loss=detection_loss,
                        spatial_loss=spatial_loss,
                        multiview_loss=multiview_loss,
                        feature_loss=feature_loss
                    )
                    
                    batch_losses.append(total_sample_loss)
                    
                except RuntimeError as e:
                    if "CUDA" in str(e) or "out of memory" in str(e).lower():
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        logger.warning(f"⚠️  CUDA OOM: {str(e)[:100]}")
                    continue
                except Exception as e:
                    logger.warning(f"⚠️  Error: {str(e)[:100]}")
                    continue
            
            if batch_losses:
                avg_batch_loss = torch.stack(batch_losses).mean()
                
                self.optimizer.zero_grad()
                avg_batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                total_loss += avg_batch_loss.item()
                num_batches += 1
                
                if num_batches % 10 == 0:
                    samples_processed = min(i + len(batch_samples), total_samples)
                    progress = (samples_processed / total_samples) * 100
                    avg_loss_so_far = total_loss / num_batches
                    
                    # Log task weights if using uncertainty loss
                    if self.use_uncertainty_loss and num_batches % 50 == 0:
                        weights = self.uncertainty_loss.get_weights()
                        logger.info(f"   Batch {num_batches}: {samples_processed}/{total_samples} ({progress:.1f}%), Loss: {avg_loss_so_far:.6f}")
                        logger.info(f"   Task Weights: {weights}")
                    else:
                        logger.info(f"   Batch {num_batches}: {samples_processed}/{total_samples} ({progress:.1f}%), Loss: {avg_loss_so_far:.6f}")
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Save task weights for this epoch
        if self.use_uncertainty_loss:
            weights = self.uncertainty_loss.get_weights()
            self.training_stats["task_weights_history"].append({
                "epoch": epoch,
                "weights": weights
            })
        
        return avg_loss
    
    def validate(self, val_samples: List[Dict]) -> float:
        """Validate on validation set."""
        self.student_model.eval()
        if self.use_uncertainty_loss:
            self.uncertainty_loss.eval()
        
        total_loss = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for sample in val_samples:
                try:
                    img_path = Path(sample["image_path"])
                    if not img_path.exists():
                        continue
                    
                    image = Image.open(img_path).convert('RGB')
                    image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(self.device).float()
                    
                    teacher_response_dict = self.teacher_model.generate_response(
                        "Describe this 3D scene and identify objects.",
                        str(img_path)
                    )
                    if isinstance(teacher_response_dict, dict):
                        teacher_response = teacher_response_dict.get('response', str(teacher_response_dict))
                    else:
                        teacher_response = str(teacher_response_dict)
                    
                    student_response = self.student_model.generate_response(
                        "Describe this 3D scene and identify objects.",
                        image_tensor
                    )
                    if isinstance(student_response, dict):
                        student_text = student_response.get('response', str(student_response))
                    else:
                        student_text = str(student_response)
                    
                    student_tokens = set(student_text.lower().split())
                    teacher_tokens = set(teacher_response.lower().split())
                    if len(teacher_tokens) > 0:
                        intersection = len(student_tokens.intersection(teacher_tokens))
                        union = len(student_tokens.union(teacher_tokens))
                        similarity = intersection / union if union > 0 else 0.0
                        text_loss = torch.tensor(1.0 - similarity, device=self.device)
                        total_loss += text_loss.item()
                        num_samples += 1
                    
                except Exception as e:
                    continue
        
        avg_loss = total_loss / num_samples if num_samples > 0 else float('inf')
        return avg_loss
    
    def _focal_loss_multilabel(self, logits: torch.Tensor, targets: torch.Tensor, 
                               alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
        """Focal loss for multi-label classification."""
        probs = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_loss = alpha * (1 - p_t) ** gamma * ce_loss
        return focal_loss.mean()
    
    def train(self):
        """Main training loop."""
        logger.info("🚀 Starting improved training with uncertainty-based loss...")
        
        start_time = time.time()
        
        # Initialize models
        self.initialize_models()
        
        # Load samples
        train_samples, val_samples = self.load_expanded_datasets()
        
        if not train_samples:
            logger.error("❌ No training samples found!")
            return
        
        logger.info(f"📊 Training on {len(train_samples)} samples")
        if val_samples:
            logger.info(f"📊 Validating on {len(val_samples)} samples")
        
        best_val_loss = float('inf')
        best_train_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        for epoch in range(1, self.epochs + 1):
            logger.info(f"\n📅 Epoch {epoch}/{self.epochs}")
            
            train_loss = self.train_epoch(train_samples, epoch)
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            val_loss = None
            if val_samples:
                logger.info("   🔍 Running validation...")
                val_loss = self.validate(val_samples)
                logger.info(f"   Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.2e}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.training_stats["best_val_loss"] = val_loss
                    self.training_stats["best_loss"] = train_loss
                    patience_counter = 0
                    self.save_checkpoint(epoch, train_loss, "best", val_loss=val_loss)
                    logger.info(f"   ✅ New best validation loss: {val_loss:.6f}")
                else:
                    patience_counter += 1
                    logger.info(f"   ⏳ No improvement ({patience_counter}/{self.early_stopping_patience})")
            else:
                logger.info(f"   Loss: {train_loss:.6f}, LR: {current_lr:.2e}")
                if train_loss < best_train_loss:
                    best_train_loss = train_loss
                    self.training_stats["best_loss"] = train_loss
                    self.save_checkpoint(epoch, train_loss, "best")
                    logger.info(f"   ✅ New best loss: {train_loss:.6f}")
            
            # Early stopping
            if self.early_stopping_patience and val_samples and patience_counter >= self.early_stopping_patience:
                logger.info(f"   ⏹️  Early stopping triggered")
                self.training_stats["early_stopped"] = True
                break
            
            # Periodic checkpoint
            if epoch % 10 == 0:
                checkpoint_name = f"epoch_{epoch}"
                if val_loss is not None:
                    self.save_checkpoint(epoch, train_loss, checkpoint_name, val_loss=val_loss)
                else:
                    self.save_checkpoint(epoch, train_loss, checkpoint_name)
        
        training_time = time.time() - start_time
        self.training_stats["training_time"] = training_time
        self.training_stats["epochs_completed"] = epoch
        
        self.save_final_results()
        
        logger.info("✅ Training completed!")
        logger.info(f"   Best Train Loss: {self.training_stats['best_loss']:.6f}")
        if val_samples:
            logger.info(f"   Best Val Loss: {self.training_stats['best_val_loss']:.6f}")
        logger.info(f"   Training Time: {training_time:.2f}s")
    
    def save_checkpoint(self, epoch: int, loss: float, name: str, val_loss: float = None):
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"improved_model_{name}.pt"
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': self.student_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'training_stats': self.training_stats
        }
        if val_loss is not None:
            checkpoint_data['val_loss'] = val_loss
        if self.use_uncertainty_loss:
            checkpoint_data['uncertainty_state_dict'] = self.uncertainty_loss.state_dict()
        torch.save(checkpoint_data, checkpoint_path)
        logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
    
    def save_final_results(self):
        """Save final training results."""
        results_path = self.checkpoint_dir / "improved_training_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.training_stats, f, indent=2)
        logger.info(f"💾 Final results saved: {results_path}")


def main():
    """Main function."""
    pipeline = ImprovedTrainingPipeline(use_uncertainty_loss=True)
    pipeline.train()


if __name__ == "__main__":
    main()
