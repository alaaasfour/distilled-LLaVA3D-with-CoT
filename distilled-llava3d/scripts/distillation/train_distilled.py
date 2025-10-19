#!/usr/bin/env python3
"""
Complete Distilled LLaVA-3D Training Pipeline
Implements knowledge distillation from LLaVA-3D-7B to 3B parameters.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import os
import logging
import sys
from pathlib import Path
import importlib.machinery as importlib_machinery
import types
import warnings
from tqdm import tqdm
from datetime import datetime
from dataset_loader import create_dataloader

LLAVA3D_REPO = Path("/scratch/alasfour/llava-3d/LLaVA-3D")
if LLAVA3D_REPO.is_dir():
    repo_path = str(LLAVA3D_REPO)
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

try:
    import cv2  # type: ignore
except ModuleNotFoundError:
    warnings.warn(
        "OpenCV (cv2) is not installed; video-specific features will be skipped."
    )

    def _cv2_inpaint(depth, mask, radius, flags=None):
        return depth

    cv2_stub = types.ModuleType("cv2")
    cv2_stub.__spec__ = importlib_machinery.ModuleSpec("cv2", loader=None)
    cv2_stub.inpaint = _cv2_inpaint  # type: ignore[attr-defined]
    cv2_stub.INPAINT_NS = 0  # type: ignore[attr-defined]
    sys.modules.setdefault("cv2", cv2_stub)

try:
    from scipy.spatial.transform import Rotation as R  # type: ignore  # noqa: F401
except ModuleNotFoundError:
    import numpy as _np

    warnings.warn(
        "SciPy is not installed; using a lightweight rotation stub."
    )

    class _Rotation:
        def __init__(self, matrix: _np.ndarray):
            self._matrix = matrix

        @classmethod
        def from_quat(cls, quat):
            x, y, z, w = (float(q) for q in quat)
            norm = (x * x + y * y + z * z + w * w) ** 0.5
            if norm == 0:
                raise ValueError("Zero-norm quaternion is invalid")
            x /= norm
            y /= norm
            z /= norm
            w /= norm
            matrix = _np.array([
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ], dtype=float)
            return cls(matrix)

        def as_matrix(self):
            return self._matrix

    scipy_module = types.ModuleType("scipy")
    scipy_module.__spec__ = importlib_machinery.ModuleSpec("scipy", loader=None)
    spatial_module = types.ModuleType("scipy.spatial")
    spatial_module.__spec__ = importlib_machinery.ModuleSpec("scipy.spatial", loader=None)
    transform_module = types.ModuleType("scipy.spatial.transform")
    transform_module.__spec__ = importlib_machinery.ModuleSpec("scipy.spatial.transform", loader=None)
    transform_module.Rotation = _Rotation  # type: ignore[attr-defined]
    spatial_module.transform = transform_module  # type: ignore[attr-defined]
    scipy_module.spatial = spatial_module  # type: ignore[attr-defined]
    sys.modules.setdefault("scipy", scipy_module)
    sys.modules.setdefault("scipy.spatial", spatial_module)
    sys.modules.setdefault("scipy.spatial.transform", transform_module)

from llava.conversation import conv_templates
from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import process_images, tokenizer_special_token


# Import our custom modules
from student_model import DistilledLLaVA3D, DistilledLLaVA3DConfig
from distillation_loss import create_distillation_loss
from load_teacher import load_llava3d_teacher

class DistillationTrainer:
    """Main trainer class for distillation."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models
        self.teacher_model = None
        self.teacher_tokenizer = None
        self.teacher_processor = None
        self.teacher_context_len = None
        self.teacher_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.student_model = None
        
        # Initialize loss function
        self.distillation_loss = create_distillation_loss(
            loss_type=config["distillation_method"],
            temperature=config.get("temperature", 3.0),
            alpha=config.get("alpha", 0.7)
        )
        
        # Initialize optimizer
        self.optimizer = None
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration."""
        log_dir = "logs/training"
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{log_dir}/distillation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_teacher_model(self):
        """Load the pre-trained LLaVA-3D teacher model."""
        self.logger.info("Loading teacher model...")

        teacher_cfg = self.config.get("teacher", {})
        model_path = teacher_cfg.get("model_path", self.config.get("teacher_model", "ChaimZhu/LLaVA-3D-7B"))
        device = teacher_cfg.get("device", "cuda")
        precision = teacher_cfg.get("precision", "bf16")
        quant = teacher_cfg.get("quant", None)

        tokenizer, model, processor, context_len = load_llava3d_teacher(
            model_path=model_path,
            device=device,
            precision=precision,
            quant=quant,
        )

        self.teacher_model = model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        self.teacher_tokenizer = tokenizer
        self.teacher_processor = processor
        self.teacher_context_len = context_len
        self.teacher_device = next(self.teacher_model.parameters()).device

        self.logger.info(
            "Teacher model loaded from %s on %s (context_len=%s)",
            model_path,
            self.teacher_device,
            context_len,
        )
        
    def load_student_model(self):
        """Load the student model."""
        self.logger.info("Loading student model...")
        
        config = DistilledLLaVA3DConfig()
        self.student_model = DistilledLLaVA3D.from_teacher(
            (self.teacher_tokenizer, self.teacher_model, self.teacher_processor, self.teacher_context_len),
            config=config,
        )
        self.student_model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=0.01
        )
        
        self.logger.info(f"Student model loaded with {sum(p.numel() for p in self.student_model.parameters()):,} parameters")
        
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch."""
        self.student_model.train()
        total_loss = 0.0
        num_batches = len(dataloader)

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

        conversation_template = self.config.get("conversation_template", "llava_v1")
        base_conv_template = conv_templates[conversation_template]
        image_token = DEFAULT_IMAGE_TOKEN
        if self.teacher_model.config.mm_use_im_start_end:
            image_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN

        teacher_dtype = next(self.teacher_model.parameters()).dtype
        student_dtype = next(self.student_model.parameters()).dtype
        vision_dtype = torch.float16

        if isinstance(self.teacher_processor, dict):
            teacher_image_processor = self.teacher_processor.get("image")
        else:
            teacher_image_processor = getattr(self.teacher_processor, "image_processor", None)

        kd_weight = float(self.config.get("kd_weight", 1.0))
        ce_weight = float(self.config.get("ce_weight", 1.0))

        for batch_idx, batch in enumerate(progress_bar):
            images = batch["images"]
            questions = batch["questions"]
            answers = batch.get("answers")

            if teacher_image_processor is None:
                vision_tower = getattr(self.teacher_model.get_model(), "vision_tower", None)
                teacher_image_processor = getattr(vision_tower, "image_processor", None)

            image_tensors = process_images(images, teacher_image_processor, self.teacher_model.config)
            if isinstance(image_tensors, list):
                image_tensors = torch.stack(image_tensors, dim=0)
            if hasattr(image_tensors, "pixel_values"):
                image_tensors = image_tensors.pixel_values

            image_tensors = image_tensors.to(self.teacher_device, dtype=vision_dtype)
            student_images = image_tensors.to(self.device, dtype=student_dtype)

            prompt_ids_list = []
            full_ids_list = []
            labels_list = []
            answers_available = True if answers else False

            iter_answers = answers if answers is not None else [None] * len(questions)
            for question, answer in zip(questions, iter_answers):
                prompt = question
                if image_token not in prompt:
                    prompt = f"{image_token}\n{prompt}"
                conv_prompt = base_conv_template.copy()
                conv_prompt.append_message(conv_prompt.roles[0], prompt)
                conv_prompt.append_message(conv_prompt.roles[1], None)
                prompt_ids = tokenizer_special_token(conv_prompt.get_prompt(), self.teacher_tokenizer, return_tensors="pt")
                prompt_ids_list.append(prompt_ids)

                if ce_weight > 0 and answer:
                    conv_full = base_conv_template.copy()
                    conv_full.append_message(conv_full.roles[0], prompt)
                    conv_full.append_message(conv_full.roles[1], answer)
                    full_ids = tokenizer_special_token(conv_full.get_prompt(), self.teacher_tokenizer, return_tensors="pt")
                    labels = full_ids.clone()
                    labels[: prompt_ids.shape[0]] = -100
                    full_ids_list.append(full_ids)
                    labels_list.append(labels)
                else:
                    answers_available = False

            prompt_input_ids = torch.nn.utils.rnn.pad_sequence(
                prompt_ids_list,
                batch_first=True,
                padding_value=self.teacher_tokenizer.pad_token_id,
            )
            prompt_attention_mask = (prompt_input_ids != self.teacher_tokenizer.pad_token_id).long()

            student_prompt_ids = prompt_input_ids.to(self.device)
            student_prompt_mask = prompt_attention_mask.to(self.device)
            teacher_prompt_ids = prompt_input_ids.to(self.teacher_device)
            teacher_prompt_mask = prompt_attention_mask.to(self.teacher_device)

            loss_tensor = torch.tensor(0.0, device=self.device)
            self.optimizer.zero_grad()

            if kd_weight > 0:
                student_outputs = self.student_model(
                    input_ids=student_prompt_ids,
                    attention_mask=student_prompt_mask,
                    pixel_values=student_images,
                )

                with torch.no_grad():
                    teacher_outputs = self.teacher_model(
                        input_ids=teacher_prompt_ids,
                        attention_mask=teacher_prompt_mask,
                        images=image_tensors,
                    )

                kd_loss = self.distillation_loss(student_outputs, teacher_outputs)
                loss_tensor = loss_tensor + kd_weight * kd_loss

            if ce_weight > 0 and answers_available and full_ids_list:
                full_input_ids = torch.nn.utils.rnn.pad_sequence(
                    full_ids_list,
                    batch_first=True,
                    padding_value=self.teacher_tokenizer.pad_token_id,
                )
                full_attention_mask = (full_input_ids != self.teacher_tokenizer.pad_token_id).long()
                labels = torch.nn.utils.rnn.pad_sequence(
                    labels_list,
                    batch_first=True,
                    padding_value=-100,
                )

                ce_outputs = self.student_model(
                    input_ids=full_input_ids.to(self.device),
                    attention_mask=full_attention_mask.to(self.device),
                    pixel_values=student_images,
                    labels=labels.to(self.device),
                )
                loss_tensor = loss_tensor + ce_weight * ce_outputs.loss

            if not torch.isfinite(loss_tensor):
                continue

            loss_tensor.backward()
            self.optimizer.step()

            total_loss += loss_tensor.item()

            progress_bar.set_postfix({'loss': f'{loss_tensor.item():.4f}'})

            if batch_idx % 50 == 0:
                self.logger.info(
                    "Epoch %s, Batch %s/%s, Loss: %.4f",
                    epoch,
                    batch_idx,
                    num_batches,
                    loss_tensor.item(),
                )

        return total_loss / num_batches
        
    def train(self):
        """Main training loop."""
        self.logger.info("Starting distillation training...")
        
        # Load models
        self.load_teacher_model()
        self.load_student_model()
        
        dataloader = create_dataloader(
            data_dir=self.config.get("data_dir"),
            tokenizer=self.teacher_tokenizer,
            processor=self.teacher_processor,
            batch_size=self.config["batch_size"],
            num_workers=self.config.get("num_workers", 0),
            manifest=self.config.get("data_manifest"),
        )
        
        # Training loop
        for epoch in range(self.config["num_epochs"]):
            self.logger.info(f"Starting epoch {epoch + 1}/{self.config['num_epochs']}")
            
            avg_loss = self.train_epoch(dataloader, epoch + 1)
            
            self.logger.info(f"Epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch + 1, avg_loss)
                
        self.logger.info("Training completed!")
        
    def save_checkpoint(self, epoch, loss):
        """Save model checkpoint."""
        checkpoint_dir = "models/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.student_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'config': self.config
        }
        
        checkpoint_path = f"{checkpoint_dir}/distilled_llava3d_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")

def main():
    """Main function."""
    # Configuration
    config = {
        "teacher": {
            "model_path": "ChaimZhu/LLaVA-3D-7B",
            "device": "cuda",
            "precision": "fp16",
            "quant": "4bit",
        },
        "student_size": "3B",
        "distillation_method": "knowledge_distillation",
        "learning_rate": 1e-4,
        "batch_size": 2,
        "num_epochs": 1,
        "temperature": 3.0,
        "alpha": 0.7,
        "conversation_template": "llava_v1",
        "data_dir": "/scratch/alasfour/llava-3d/LLaVA-3D/demo/scannet",
        "data_manifest": "/scratch/alasfour/distilled-llava3d/data/manifests/scannet_scene0356_demo.json",
        "num_workers": 0,
        "kd_weight": 1.0,
        "ce_weight": 1.0,
    }
    
    print("Distilled LLaVA-3D Training Configuration:")
    print(json.dumps(config, indent=2))
    
    # Initialize trainer
    trainer = DistillationTrainer(config)
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()
