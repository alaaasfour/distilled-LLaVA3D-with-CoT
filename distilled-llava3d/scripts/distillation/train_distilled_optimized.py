#!/usr/bin/env python3
"""
Memory-optimized training script for distilled LLaVA-3D using the real teacher model.
"""

import torch
import json
import os
import logging
from tqdm import tqdm
from datetime import datetime
import gc

from llava.conversation import conv_templates
from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import process_images, tokenizer_special_token

# Import our custom modules
from student_model import DistilledLLaVA3D, DistilledLLaVA3DConfig
from distillation_loss import create_distillation_loss
from load_teacher import load_llava3d_teacher
from dataset_loader import create_dataloader

class DistillationTrainer:
    """Trainer for distilled LLaVA-3D model."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models
        self.teacher_model = None
        self.teacher_tokenizer = None
        self.teacher_processor = None
        self.teacher_context_len = None
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
        """Load the teacher model (mock for now)."""
        self.logger.info("Loading teacher model...")
        
        teacher_model_path = self.config["teacher_model"]
        precision = self.config.get("precision", "bf16")

        self.teacher_tokenizer, self.teacher_model, self.teacher_processor, context_len = load_llava3d_teacher(
            model_path=teacher_model_path,
            device=self.device,
            precision=precision,
            quant=None,
        )
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        self.teacher_device = next(self.teacher_model.parameters()).device
        self.teacher_context_len = context_len
        self.logger.info(
            "Teacher model loaded from %s (context_len=%s)",
            teacher_model_path,
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
        
        # Enable gradient checkpointing to save memory
        if hasattr(self.student_model, 'gradient_checkpointing_enable'):
            self.student_model.gradient_checkpointing_enable()
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=0.01
        )
        
        self.logger.info(f"Student model loaded with {sum(p.numel() for p in self.student_model.parameters()):,} parameters")
        
    def create_dataset(self):
        """Create dataset for training."""
        self.logger.info("Creating training dataloader...")

        return create_dataloader(
            data_dir=self.config.get("data_dir"),
            tokenizer=self.teacher_tokenizer,
            processor=self.teacher_processor,
            batch_size=self.config["batch_size"],
            num_workers=self.config.get("num_workers", 0),
            manifest=self.config.get("data_manifest"),
        )
        
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch."""
        self.student_model.train()
        total_loss = 0.0
        num_batches = len(dataloader)
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

        conversation_template = self.config.get("conversation_template", "llava_v1")
        conv_template = conv_templates[conversation_template]
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

        for batch_idx, batch in enumerate(progress_bar):
            images = batch["images"]
            questions = batch["questions"]

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

            prompts = []
            for question in questions:
                prompt = question
                if image_token not in prompt:
                    prompt = f"{image_token}\n{prompt}"
                conv = conv_template.copy()
                conv.append_message(conv.roles[0], prompt)
                conv.append_message(conv.roles[1], None)
                prompts.append(conv.get_prompt())

            input_ids_list = [
                tokenizer_special_token(p, self.teacher_tokenizer, return_tensors="pt")
                for p in prompts
            ]

            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids_list,
                batch_first=True,
                padding_value=self.teacher_tokenizer.pad_token_id,
            )
            attention_mask = (input_ids != self.teacher_tokenizer.pad_token_id).long()

            student_input_ids = input_ids.to(self.device)
            student_attention_mask = attention_mask.to(self.device)
            teacher_input_ids = input_ids.to(self.teacher_device)
            teacher_attention_mask = attention_mask.to(self.teacher_device)

            student_outputs = self.student_model(
                input_ids=student_input_ids,
                attention_mask=student_attention_mask,
                pixel_values=student_images,
            )

            with torch.no_grad():
                teacher_outputs = self.teacher_model(
                    input_ids=teacher_input_ids,
                    attention_mask=teacher_attention_mask,
                    images=image_tensors,
                )
            
            # Compute distillation loss
            loss = self.distillation_loss(student_outputs, teacher_outputs)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Clear cache and garbage collect
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()
            
            # Log every 10 batches
            if batch_idx % 10 == 0:
                self.logger.info(f"Epoch {epoch}, Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}")
        
        return total_loss / num_batches
        
    def train(self):
        """Main training loop."""
        self.logger.info("Starting memory-optimized distillation training...")
        
        # Load models
        self.load_teacher_model()
        self.load_student_model()
        
        # Create dataset
        dataloader = self.create_dataset()
        
        # Training loop
        for epoch in range(self.config["num_epochs"]):
            self.logger.info(f"Starting epoch {epoch + 1}/{self.config['num_epochs']}")
            
            avg_loss = self.train_epoch(dataloader, epoch + 1)
            
            self.logger.info(f"Epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % 2 == 0:
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
        "teacher_model": "ChaimZhu/LLaVA-3D-7B",
        "student_size": "3B",
        "distillation_method": "knowledge_distillation",
        "learning_rate": 1e-4,
        "batch_size": 2,
        "num_epochs": 5,
        "temperature": 3.0,
        "alpha": 0.7,
        "precision": "bf16",
        "quant": "4bit",
        "conversation_template": "llava_v1",
        "data_dir": None,
        "data_manifest": None,
        "num_workers": 0,
    }
    
    print("Memory-Optimized Distilled LLaVA-3D Training Configuration:")
    print(json.dumps(config, indent=2))
    
    # Initialize trainer
    trainer = DistillationTrainer(config)
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()
