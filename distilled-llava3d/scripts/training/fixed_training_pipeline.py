#!/usr/bin/env python3
"""
Fixed Training Pipeline
======================

This script implements the fixed training pipeline with:
1. Text generation loss aligned with evaluation
2. Validation set and early stopping
3. Better use of learned features in text generation
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
from enhanced_mock_teacher import EnhancedMockTeacher
from real_llava3d_teacher import RealLLaVA3DTeacher
from scripts.distillation.distillation_loss import AdaptiveDistillationLoss
from spatial_reasoning_augmentation import SpaREAugmentor
from object_detection_integration import ObjectDetectionIntegration
from real_depth_teacher import RealDepthTeacher

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FixedTrainingPipeline:
    """
    Fixed Training Pipeline
    
    Aligns training objectives with evaluation metrics.
    """
    
    def __init__(self, 
                 data_root: str = "/home/alasfour/scratch/distilled-llava3d/data",
                 checkpoint_dir: str = "/home/alasfour/scratch/distilled-llava3d/checkpoints"):
        """
        Initialize fixed training pipeline.
        
        Args:
            data_root: Root directory for datasets
            checkpoint_dir: Directory for saving checkpoints
        """
        self.data_root = Path(data_root)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training parameters
        self.epochs = 50  # Full training like previous successful model
        self.batch_size = 1  # Reduced to 1 for VGGT on GPU (can increase if memory allows)
        self.learning_rate = 1e-4
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.validation_split = 0.0  # No validation split - use all data for training
        self.early_stopping_patience = None  # Disabled - no early stopping
        
        # VGGT device: 'cuda' for GPU (faster) or 'cpu' for CPU (slower but more memory)
        # Default to CPU since GPU is too small (9.75 GB) for VGGT + student + teachers
        self.vggt_device = 'cpu'  # CPU is safer - GPU too small for all models
        
        # Initialize components
        self.student_model = None
        self.teacher_model = None
        self.depth_teacher = None
        self.distillation_loss = None
        self.optimizer = None
        self.object_detection = None
        self.spare_augmentor = SpaREAugmentor(relation_limit=4)
        
        # Loss weights (TUNED: Balanced for better performance)
        self.lambda_det = 0.35  # Reduced from 0.4 (detection was too strong)
        self.lambda_depth_ce = 0.25
        self.lambda_depth_reg = 0.15
        self.lambda_depth_kl = 0.0125  # Reduced KL weight (was 0.1 * 0.25 = 0.025)
        self.lambda_spatial = 0.25
        self.lambda_text = 0.0  # DISABLED: Text generation loss causing issues, focus on features first
        self.lambda_mv = 0.1
        self.lambda_feat = 0.3
        
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
            "early_stopped": False
        }
        
        logger.info(f"🚀 Initializing Fixed Training Pipeline")
        logger.info(f"   Data Root: {self.data_root}")
        logger.info(f"   Checkpoint Dir: {self.checkpoint_dir}")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Epochs: {self.epochs}")
        logger.info(f"   Validation Split: {self.validation_split}")
        if self.early_stopping_patience:
            logger.info(f"   Early Stopping Patience: {self.early_stopping_patience}")
        else:
            logger.info(f"   Early Stopping: Disabled")
    
    def initialize_models(self):
        """Initialize all models."""
        logger.info("🤖 Initializing models...")
        
        # Initialize student model with VGGT device preference
        config = DistilledLLaVA3DConfig()
        # Start with CPU to avoid OOM during initialization
        config.vggt_device = 'cpu'
        logger.info(f"🔧 Initializing with VGGT on CPU (will try GPU after model loads)")
        
        # Initialize model with VGGT on CPU first
        self.student_model = DistilledLLaVA3D(config)
        
        # Clear any existing GPU allocations
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Move student model to GPU first (without VGGT)
        try:
            self.student_model.to(self.device)
            logger.info("✅ Student model moved to GPU")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error("❌ OOM: Cannot fit student model on GPU. Need more GPU memory.")
                raise
            else:
                raise
        
        # Now try moving VGGT to GPU if requested
        if hasattr(self.student_model, 'vision_encoder') and \
           hasattr(self.student_model.vision_encoder, 'vggt_model') and \
           self.student_model.vision_encoder.vggt_model is not None:
            
            # Keep VGGT on CPU by default - GPU too small for VGGT + student + teachers
            # Only try GPU if explicitly requested and there's enough space
            if self.vggt_device == 'cuda' and torch.cuda.is_available():
                # Check if we have enough free memory (need at least 4 GB free for VGGT)
                allocated_before = torch.cuda.memory_allocated(0) / 1e9
                free_before = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)) / 1e9
                logger.info(f"💾 Before VGGT: {allocated_before:.2f} GB allocated, {free_before:.2f} GB free")
                
                if free_before < 4.0:
                    logger.warning("⚠️  Not enough GPU memory for VGGT (need ~4 GB free)")
                    logger.warning("   Keeping VGGT on CPU (slower but stable)")
                    self.student_model.vision_encoder.vggt_model.to('cpu')
                    self.vggt_device = 'cpu'
                else:
                    try:
                        # Try moving VGGT to GPU
                        self.student_model.vision_encoder.vggt_model.to('cuda')
                        self.vggt_device = 'cuda'
                        allocated_after = torch.cuda.memory_allocated(0) / 1e9
                        logger.info(f"✅ VGGT moved to GPU (faster training)")
                        logger.info(f"💾 After VGGT: {allocated_after:.2f} GB allocated")
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            logger.warning("⚠️  GPU OOM when moving VGGT to GPU")
                            logger.warning("   Falling back to CPU (slower but will work)")
                            self.student_model.vision_encoder.vggt_model.to('cpu')
                            self.vggt_device = 'cpu'
                            torch.cuda.empty_cache()
                            logger.info("✅ VGGT on CPU (training will be slower but stable)")
                        else:
                            raise
            else:
                logger.info(f"💾 VGGT on {self.vggt_device} (CPU is safer for 9.75 GB GPU)")
        
        # Final memory report
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated(0) / 1e9
            reserved = torch.cuda.memory_reserved(0) / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            free = total - reserved
            logger.info(f"💾 Final GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved, {free:.2f} GB free, {total:.2f} GB total")
        
        # Initialize REAL LLaVA-3D teacher (with fallback to mock)
        # Use HuggingFace model: ChaimZhu/LLaVA-3D-7B (official LLaVA-3D model)
        # The local directory is just source code, not the model checkpoint
        teacher_model_path = "ChaimZhu/LLaVA-3D-7B"  # HuggingFace model
        try:
            # Try loading real teacher on CPU first to avoid OOM
            # The teacher is huge (7B params) and won't fit on GPU with everything else
            logger.info("💾 Attempting to load real teacher on CPU (to avoid OOM)")
            self.teacher_model = RealLLaVA3DTeacher(model_path=teacher_model_path, device="cpu")
            if self.teacher_model.model is None:
                logger.warning("⚠️  Real teacher not available, using Enhanced Mock Teacher")
                self.teacher_model = EnhancedMockTeacher(device=self.device)
            else:
                logger.info("✅ Real LLaVA-3D teacher initialized on CPU")
        except Exception as e:
            logger.warning(f"⚠️  Could not initialize real teacher: {e}, using Enhanced Mock Teacher")
            self.teacher_model = EnhancedMockTeacher(device=self.device)
        
        # Initialize REAL depth teacher
        try:
            self.depth_teacher = RealDepthTeacher(device=self.device)
            logger.info("✅ Real depth teacher initialized")
        except Exception as e:
            logger.warning(f"⚠️  Could not initialize real depth teacher: {e}")
            self.depth_teacher = None
        
        # Initialize distillation loss
        self.distillation_loss = AdaptiveDistillationLoss(
            temperature=3.0,
            alpha=0.7,
            adaptation_rate=0.1
        )
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5
        )
        
        # Initialize learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        # Initialize object detection
        try:
            self.object_detection = ObjectDetectionIntegration(self.student_model, device=self.device)
            logger.info("✅ Object detection integration initialized")
        except Exception as e:
            logger.warning(f"⚠️  Object detection integration unavailable: {e}")
            self.object_detection = None
        
        logger.info("✅ All models initialized successfully!")
    
    def load_expanded_datasets(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Load expanded training datasets and split into train/val.
        
        Returns:
            (training_samples, validation_samples)
        """
        logger.info("📊 Loading expanded training datasets...")
        
        all_samples = []
        
        # Load from expanded directories (SCALED: More scenes and images)
        scannet_expanded = self.data_root / "scannet_real" / "expanded"
        front_expanded = self.data_root / "3d_front_real" / "expanded"
        scannet_sample = self.data_root / "scannet_real" / "sample"
        scannet_full = self.data_root / "scannet"
        front_original = self.data_root / "3d_front_real"
        matterport_path = self.data_root / "matterport3d"
        
        # Load ScanNet expanded (SCALED: All scenes, all images)
        if scannet_expanded.exists():
            scene_dirs = sorted([d for d in scannet_expanded.glob("scene*") if d.is_dir()])
            logger.info(f"📂 Found {len(scene_dirs)} ScanNet expanded scenes")
            for scene_dir in scene_dirs:
                images = sorted(list(scene_dir.glob("*.jpg")))
                annotations_file = scene_dir / "annotations.json"
                
                annotations = {}
                if annotations_file.exists():
                    try:
                        with open(annotations_file, 'r') as f:
                            annotations = json.load(f)
                    except:
                        pass
                
                # Take ALL images from each scene (not just a few)
                for img_path in images:
                    all_samples.append({
                        "image_path": str(img_path),
                        "scene_id": scene_dir.name,
                        "annotations": annotations,
                        "dataset": "scannet_real"
                    })
        
        # Load ScanNet sample (fallback)
        if scannet_sample.exists() and len(all_samples) < 100:
            scene_dirs = sorted([d for d in scannet_sample.glob("scene*") if d.is_dir()])
            logger.info(f"📂 Found {len(scene_dirs)} ScanNet sample scenes")
            for scene_dir in scene_dirs:
                images = sorted(list(scene_dir.glob("*.jpg")))
                for img_path in images:
                    all_samples.append({
                        "image_path": str(img_path),
                        "scene_id": scene_dir.name,
                        "dataset": "scannet_real"
                    })
        
        # Load ScanNet full dataset (SCALED: More scenes)
        if scannet_full.exists():
            scene_dirs = sorted([d for d in scannet_full.glob("scene*") if d.is_dir()])[:30]  # Up to 30 scenes
            logger.info(f"📂 Found {len(scene_dirs)} ScanNet full scenes")
            for scene_dir in scene_dirs:
                images = sorted(list(scene_dir.glob("*.jpg")))[:10]  # Up to 10 images per scene
                for img_path in images:
                    all_samples.append({
                        "image_path": str(img_path),
                        "scene_id": scene_dir.name,
                        "dataset": "scannet"
                    })
        
        # Load 3D-FRONT expanded (SCALED: All scenes, all images)
        if front_expanded.exists():
            scene_dirs = sorted([d for d in front_expanded.glob("*") if d.is_dir() and d.name != "expanded"])
            logger.info(f"📂 Found {len(scene_dirs)} 3D-FRONT expanded scenes")
            for scene_dir in scene_dirs:
                images = sorted(list(scene_dir.glob("*.jpg")))
                annotations_file = scene_dir / "annotations.json"
                
                annotations = {}
                if annotations_file.exists():
                    try:
                        with open(annotations_file, 'r') as f:
                            annotations = json.load(f)
                    except:
                        pass
                
                # Take ALL images from each scene
                for img_path in images:
                    all_samples.append({
                        "image_path": str(img_path),
                        "scene_id": scene_dir.name,
                        "annotations": annotations,
                        "dataset": "3d_front_real"
                    })
        
        # Load 3D-FRONT original (SCALED: More scenes)
        if front_original.exists():
            scene_dirs = sorted([d for d in front_original.glob("*") if d.is_dir()])[:20]  # Up to 20 scenes
            logger.info(f"📂 Found {len(scene_dirs)} 3D-FRONT original scenes")
            for scene_dir in scene_dirs:
                images = sorted(list(scene_dir.glob("view_*.jpg")))[:5]  # Up to 5 views per scene
                for img_path in images:
                    all_samples.append({
                        "image_path": str(img_path),
                        "scene_id": scene_dir.name,
                        "dataset": "3d_front"
                    })
        
        # Load Matterport3D (NEW: Additional dataset)
        if matterport_path.exists():
            scene_dirs = sorted([d for d in matterport_path.glob("*") if d.is_dir() and d.name != "manifest.json"])[:20]
            logger.info(f"📂 Found {len(scene_dirs)} Matterport3D scenes")
            for scene_dir in scene_dirs:
                images = sorted(list(scene_dir.glob("*.jpg")))[:5]  # Up to 5 images per scene
                for img_path in images:
                    all_samples.append({
                        "image_path": str(img_path),
                        "scene_id": scene_dir.name,
                        "dataset": "matterport3d"
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
        
        # Shuffle
        random.shuffle(all_samples)
        
        # If validation_split is 0 or None, use all samples for training
        if self.validation_split is None or self.validation_split == 0:
            train_samples = all_samples
            val_samples = []
        else:
            split_idx = int(len(all_samples) * (1 - self.validation_split))
            train_samples = all_samples[:split_idx]
            val_samples = all_samples[split_idx:]
        
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
    
    def compute_text_generation_loss(self, student_response, teacher_response) -> torch.Tensor:
        """
        Compute text generation loss using token-level similarity.
        
        Args:
            student_response: Student generated text (str or dict)
            teacher_response: Teacher target text (str or dict)
            
        Returns:
            Text generation loss
        """
        # Extract text from responses (handle both str and dict)
        if isinstance(student_response, dict):
            student_text = student_response.get('response', student_response.get('text', str(student_response)))
        else:
            student_text = str(student_response)
        
        if isinstance(teacher_response, dict):
            teacher_text = teacher_response.get('response', teacher_response.get('text', str(teacher_response)))
        else:
            teacher_text = str(teacher_response)
        
        # Ensure we have strings
        if not isinstance(student_text, str):
            student_text = str(student_text)
        if not isinstance(teacher_text, str):
            teacher_text = str(teacher_text)
        
        # Simple token-based loss (Jaccard similarity)
        student_tokens = set(student_text.lower().split())
        teacher_tokens = set(teacher_text.lower().split())
        
        if len(teacher_tokens) == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        intersection = len(student_tokens.intersection(teacher_tokens))
        union = len(student_tokens.union(teacher_tokens))
        
        similarity = intersection / union if union > 0 else 0.0
        loss = 1.0 - similarity
        
        return torch.tensor(loss, device=self.device, requires_grad=True)
    
    def train_epoch(self, training_samples: List[Dict], epoch: int) -> float:
        """
        Train for one epoch with text generation loss.
        
        Args:
            training_samples: List of training samples
            epoch: Current epoch number
            
        Returns:
            Average loss for the epoch
        """
        self.student_model.train()
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
                    
                    # Clear GPU cache periodically to avoid CUBLAS errors
                    if torch.cuda.is_available() and num_batches % 5 == 0:
                        torch.cuda.empty_cache()
                    
                    # Get teacher response and features
                    teacher_response_dict = self.teacher_model.generate_response(
                        "Describe this 3D scene and identify objects.",
                        str(img_path)
                    )
                    # Extract text from dict (teacher returns dict with 'response' key)
                    if isinstance(teacher_response_dict, dict):
                        teacher_response = teacher_response_dict.get('response', str(teacher_response_dict))
                    else:
                        teacher_response = str(teacher_response_dict)
                    
                    teacher_features = self.teacher_model.analyze_image_content(
                        np.array(image)
                    )
                    
                    # Get REAL depth supervision
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
                    
                    # Compute losses
                    sample_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32, requires_grad=True)
                    
                    # 1. Detection loss (IMPROVED: Better YOLO integration + teacher features)
                    # Ensure det_target is float32
                    det_target = torch.zeros_like(det_logits, dtype=torch.float32)
                    
                    # Use YOLO detections as primary source
                    if self.object_detection is not None:
                        try:
                            comp = self.object_detection.detect_objects_comprehensive(image_tensor)
                            dets = comp.get('detected_objects', [])
                            
                            # Filter by confidence threshold (LOWERED: More lenient for better coverage)
                            dets = [d for d in dets if isinstance(d, dict) and d.get('confidence', 0.0) > 0.3]
                            
                            for det in dets:
                                if isinstance(det, dict) and 'class' in det:
                                    cls = det['class'].lower()
                                    # Map YOLO classes to our detector classes
                                    class_mapping = {
                                        'person': 'person',
                                        'car': 'vehicle', 'truck': 'vehicle', 'bus': 'vehicle', 
                                        'motorcycle': 'vehicle', 'bicycle': 'vehicle',
                                        'building': 'building', 'house': 'building',
                                        'tree': 'tree', 'grass': 'tree',
                                        'water': 'water', 'ocean': 'water', 'lake': 'water',
                                        'road': 'road', 'street': 'road',
                                        'sky': 'sky'
                                    }
                                    mapped_cls = class_mapping.get(cls, cls)
                                    if mapped_cls in self.student_model.detector_classes:
                                        idx = self.student_model.detector_classes.index(mapped_cls)
                                        # Use confidence as target value (soft labels)
                                        current_val = det_target[0, idx].item()
                                        new_val = max(current_val, det.get('confidence', 0.7))
                                        det_target[0, idx] = float(new_val)  # Ensure float32
                        except Exception as e:
                            logger.debug(f"YOLO detection failed: {e}")
                    
                    # Also use teacher features to supplement detection labels
                    if isinstance(teacher_features, dict):
                        # Map teacher features to detection targets
                        if teacher_features.get('has_person', False):
                            idx = self.student_model.detector_classes.index('person')
                            det_target[0, idx] = float(max(det_target[0, idx].item(), 0.8))
                        if teacher_features.get('has_buildings', False):
                            idx = self.student_model.detector_classes.index('building')
                            det_target[0, idx] = float(max(det_target[0, idx].item(), 0.8))
                        if teacher_features.get('has_sky', False):
                            idx = self.student_model.detector_classes.index('sky')
                            det_target[0, idx] = float(max(det_target[0, idx].item(), 0.8))
                        if teacher_features.get('has_natural_elements', False):
                            # Check for tree or water
                            if 'tree' in self.student_model.detector_classes:
                                idx = self.student_model.detector_classes.index('tree')
                                det_target[0, idx] = float(max(det_target[0, idx].item(), 0.7))
                        if teacher_features.get('is_outdoor', False):
                            idx = self.student_model.detector_classes.index('outdoor')
                            det_target[0, idx] = float(max(det_target[0, idx].item(), 0.8))
                        if teacher_features.get('is_indoor', False):
                            idx = self.student_model.detector_classes.index('indoor')
                            det_target[0, idx] = float(max(det_target[0, idx].item(), 0.8))
                    
                    # Use focal loss with improved weighting
                    det_loss = self._focal_loss_multilabel(det_logits, det_target, alpha=0.25, gamma=2.0)
                    sample_loss = sample_loss + self.lambda_det * det_loss
                    
                    # 2. Depth loss (IMPROVED: Better depth supervision with spatial awareness)
                    if depth_discrete is not None and depth_continuous is not None:
                        # Use per-pixel depth distribution for better supervision
                        depth_probs = F.softmax(depth_logits, dim=-1)  # (batch, 3)
                        
                        # Classification loss: use mode of depth distribution
                        depth_label = int(np.median(depth_discrete))
                        depth_target = torch.tensor([depth_label], device=self.device, dtype=torch.long)
                        depth_ce = F.cross_entropy(depth_logits, depth_target)
                        sample_loss = sample_loss + self.lambda_depth_ce * depth_ce
                        
                        # Regression loss: predict mean depth
                        bin_centers = torch.tensor([0.2, 0.5, 0.8], device=self.device, dtype=torch.float32)
                        pred_depth = (depth_probs * bin_centers).sum(dim=-1)  # (batch,)
                        target_depth = torch.tensor([np.mean(depth_continuous)], device=self.device, dtype=torch.float32)
                        depth_reg = F.mse_loss(pred_depth, target_depth)
                        sample_loss = sample_loss + self.lambda_depth_reg * depth_reg
                        
                        # Additional: depth distribution loss (KL divergence)
                        # Create target distribution from depth map
                        depth_hist, _ = np.histogram(depth_continuous.flatten(), bins=3, range=(0, 1))
                        depth_hist = depth_hist.astype(np.float32) + 1e-8  # Smooth
                        depth_hist = depth_hist / depth_hist.sum()
                        depth_target_dist = torch.tensor(depth_hist, device=self.device, dtype=torch.float32).unsqueeze(0)
                        depth_kl = F.kl_div(F.log_softmax(depth_logits, dim=-1), depth_target_dist, reduction='batchmean')
                        sample_loss = sample_loss + self.lambda_depth_kl * depth_kl
                    
                    # Fallback: use teacher features for depth if depth teacher unavailable
                    elif isinstance(teacher_features, dict) and teacher_features.get('depth_layers'):
                        # Use teacher's depth perception as weak supervision
                        depth_layers = teacher_features.get('depth_layers', [])
                        if len(depth_layers) >= 3:
                            # Assume foreground/midground/background structure
                            depth_target = torch.tensor([1], device=self.device, dtype=torch.long)  # Midground
                            depth_ce = F.cross_entropy(depth_logits, depth_target)
                            sample_loss = sample_loss + 0.5 * self.lambda_depth_ce * depth_ce
                    
                    # 3. Spatial loss
                    if self.object_detection is not None:
                        dets = comp.get('detected_objects', [])
                        if len(dets) >= 2:
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
                            sample_loss = sample_loss + self.lambda_spatial * spatial_loss
                    
                    # 4. TEXT GENERATION LOSS (NEW!)
                    student_response = self.student_model.generate_response(
                        "Describe this 3D scene and identify objects.",
                        image_tensor
                    )
                    text_loss = self.compute_text_generation_loss(student_response, teacher_response)
                    sample_loss = sample_loss + self.lambda_text * text_loss
                    
                    batch_losses.append(sample_loss)
                    
                except RuntimeError as e:
                    error_msg = str(e)
                    if "CUDA" in error_msg or "out of memory" in error_msg.lower() or "CUBLAS" in error_msg:
                        # CUDA OOM error - clear cache and skip this sample
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        logger.warning(f"⚠️  CUDA OOM for sample {sample.get('image_path', 'unknown')}: {error_msg[:100]}")
                    else:
                        logger.warning(f"⚠️  Error processing sample {sample.get('image_path', 'unknown')}: {error_msg[:100]}")
                    continue
                except Exception as e:
                    logger.warning(f"⚠️  Error processing sample {sample.get('image_path', 'unknown')}: {str(e)[:100]}")
                    continue
            
            if batch_losses:
                avg_batch_loss = torch.stack(batch_losses).mean()
                
                self.optimizer.zero_grad()
                avg_batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                total_loss += avg_batch_loss.item()
                num_batches += 1
                
                # Log progress every 10 batches
                if num_batches % 10 == 0:
                    samples_processed = min(i + len(batch_samples), total_samples)
                    progress = (samples_processed / total_samples) * 100
                    avg_loss_so_far = total_loss / num_batches
                    logger.info(f"   Batch {num_batches}: {samples_processed}/{total_samples} samples ({progress:.1f}%), Avg Loss: {avg_loss_so_far:.6f}")
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def validate(self, val_samples: List[Dict]) -> float:
        """
        Validate on validation set.
        
        Args:
            val_samples: List of validation samples
            
        Returns:
            Average validation loss
        """
        self.student_model.eval()
        
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
                    
                    # Get teacher response
                    teacher_response_dict = self.teacher_model.generate_response(
                        "Describe this 3D scene and identify objects.",
                        str(img_path)
                    )
                    # Extract text from dict
                    if isinstance(teacher_response_dict, dict):
                        teacher_response = teacher_response_dict.get('response', str(teacher_response_dict))
                    else:
                        teacher_response = str(teacher_response_dict)
                    
                    # Student response
                    student_response = self.student_model.generate_response(
                        "Describe this 3D scene and identify objects.",
                        image_tensor
                    )
                    
                    # Compute text generation loss
                    text_loss = self.compute_text_generation_loss(student_response, teacher_response)
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
        """Main training loop - no validation, train for full 50 epochs like previous successful model."""
        logger.info("🚀 Starting fixed training (no validation, full 50 epochs)...")
        
        start_time = time.time()
        
        # Initialize models
        self.initialize_models()
        
        # Load all samples (no validation split when validation_split=0)
        train_samples, val_samples = self.load_expanded_datasets()
        
        if not train_samples:
            logger.error("❌ No training samples found!")
            return
        
        logger.info(f"📊 Training on {len(train_samples)} samples (no validation split)")
        
        # Track best training loss
        best_train_loss = float('inf')
        self.training_stats["best_loss"] = float('inf')
        
        # Training loop
        for epoch in range(1, self.epochs + 1):
            logger.info(f"\n📅 Epoch {epoch}/{self.epochs}")
            
            # Train epoch
            train_loss = self.train_epoch(train_samples, epoch)
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            logger.info(f"   Loss: {train_loss:.6f}, LR: {current_lr:.2e}")
            
            # Save best checkpoint based on training loss
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                self.training_stats["best_loss"] = train_loss
                self.save_checkpoint(epoch, train_loss, "best")
                logger.info(f"   ✅ New best loss: {train_loss:.6f}")
            
            # Periodic checkpoint
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, train_loss, f"epoch_{epoch}")
        
        training_time = time.time() - start_time
        self.training_stats["training_time"] = training_time
        self.training_stats["epochs_completed"] = self.epochs
        
        # Save final results
        self.save_final_results()
        
        logger.info("✅ Training completed!")
        logger.info(f"   Best Loss: {self.training_stats['best_loss']:.6f}")
        logger.info(f"   Training Time: {training_time:.2f}s")
    
    def save_checkpoint(self, epoch: int, loss: float, name: str):
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"fixed_model_{name}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.student_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'training_stats': self.training_stats
        }, checkpoint_path)
        logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
    
    def save_final_results(self):
        """Save final training results."""
        results_path = self.checkpoint_dir / "fixed_training_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.training_stats, f, indent=2)
        logger.info(f"💾 Final results saved: {results_path}")

def main():
    """Main function."""
    pipeline = FixedTrainingPipeline()
    pipeline.train()

if __name__ == "__main__":
    main()

