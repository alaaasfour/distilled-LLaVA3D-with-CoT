#!/usr/bin/env python3
"""
Hidden CoT (Scratchpad) Training Pipeline
=========================================
Train the distilled student with learnable "thinking" tokens. We only supervise
and evaluate the final answer; the scratchpad is internal and never shown.

"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
import json
import time
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import random

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.distillation.student_model import DistilledLLaVA3D, DistilledLLaVA3DConfig
from real_llava3d_teacher import RealLLaVA3DTeacher
from scripts.distillation.uncertainty_loss import MultiTaskUncertaintyLoss
from object_detection_integration import ObjectDetectionIntegration
from real_depth_teacher import RealDepthTeacher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_QUESTION = "Describe this 3D scene and identify objects."
MAX_QUESTION_LEN = 16
MAX_ANSWER_LEN = 48


class HiddenCoTTrainingPipeline:
    def __init__(
        self,
        data_root: str = "/home/alasfour/scratch/distilled-llava3d/data",
        checkpoint_dir: str = "/home/alasfour/scratch/distilled-llava3d/checkpoints",
        use_uncertainty_loss: bool = True,
    ):
        self.data_root = Path(data_root)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.epochs = 50
        self.learning_rate = 2e-5
        self.validation_split = 0.2
        self.early_stopping_patience = 10
        self.use_uncertainty_loss = use_uncertainty_loss
        self.use_amp = torch.cuda.is_available()
        self.accumulation_steps = 1

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.vision_device = None
        if torch.cuda.device_count() >= 2:
            self.vision_device = "cuda:1"
            logger.info("📌 2 GPUs: vision encoder on cuda:1, transformer on cuda:0 (split memory)")

        self.student_model = None
        self.teacher_model = None
        self.depth_teacher = None
        self.uncertainty_loss = None
        self.optimizers = None
        self.scheduler = None
        self.tokenizer = None
        self.object_detection = None

        self.training_stats = {
            "best_loss": float("inf"),
            "best_val_loss": float("inf"),
            "training_time": 0.0,
            "datasets_used": [],
            "task_weights_history": [],
        }
        logger.info("🚀 Hidden CoT (Scratchpad) Training Pipeline")
        logger.info("   Only final answer is trained and evaluated; thinking tokens are internal.")
        logger.info("   For ~10GB GPUs: gradient checkpointing + short seq; use run_cot_train.sbatch for 20GB.")

    def initialize_models(self):
        logger.info("🚀 Starting Hidden CoT training...")
        logger.info("🤖 Initializing models (Hidden CoT)...")
        gpu_mem = 0.0
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_mem += torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
        logger.info("📊 GPU Memory: %.2f GB", gpu_mem)

        num_thinking_tokens = 8
        if gpu_mem < 12:
            num_thinking_tokens = 4
        if getattr(self, "_num_thinking_tokens_override", None) is not None:
            num_thinking_tokens = self._num_thinking_tokens_override
        logger.info("   ✅ Large GPU: MAX_QUESTION_LEN=%d, MAX_ANSWER_LEN=%d, num_thinking_tokens=%d",
                    MAX_QUESTION_LEN, MAX_ANSWER_LEN, num_thinking_tokens)

        config = DistilledLLaVA3DConfig()
        config.num_thinking_tokens = num_thinking_tokens
        config.vggt_device = self.vision_device if self.vision_device else "cpu"
        self.student_model = DistilledLLaVA3D(config)
        if self.vision_device is not None:
            self.student_model.vision_encoder.to(self.vision_device)
            logger.info("✅ Student: vision_encoder on %s, rest on %s", self.vision_device, self.device)
        else:
            self.student_model.to(self.device)

        self.teacher_model = RealLLaVA3DTeacher(model_path="ChaimZhu/LLaVA-3D-7B", device="cpu")
        self.tokenizer = getattr(self.teacher_model, "tokenizer", None)
        if self.tokenizer is None and hasattr(self.teacher_model, "model"):
            self.tokenizer = getattr(self.teacher_model.model, "tokenizer", None)
        if self.tokenizer is None:
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=False)
            except Exception:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        logger.info("✅ Real LLaVA-3D teacher and tokenizer (for CoT)")

        self.depth_teacher = RealDepthTeacher(device="cpu")
        logger.info("✅ Depth teacher (CPU)")

        try:
            from bitsandbytes.optim import AdamW8bit
            all_params = list(self.student_model.parameters())
            if self.use_uncertainty_loss:
                self.uncertainty_loss = MultiTaskUncertaintyLoss(use_uncertainty=True, adaptation_rate=0.1)
                all_params += list(self.uncertainty_loss.parameters())
            self.optimizers = [AdamW8bit(all_params, lr=self.learning_rate, weight_decay=1e-5)]
            logger.info("✅ AdamW8bit for GPU student params (saves ~2x optimizer state)")
        except ImportError:
            all_params = list(self.student_model.parameters())
            if self.use_uncertainty_loss:
                self.uncertainty_loss = MultiTaskUncertaintyLoss(use_uncertainty=True, adaptation_rate=0.1)
                all_params += list(self.uncertainty_loss.parameters())
            self.optimizers = [torch.optim.AdamW(all_params, lr=self.learning_rate, weight_decay=1e-5)]

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizers[0], T_max=self.epochs, eta_min=1e-6)
        try:
            self.object_detection = ObjectDetectionIntegration(self.student_model, device=self.device)
            logger.info("✅ Object detection")
        except Exception as e:
            logger.warning("Object detection unavailable: %s", e)
            self.object_detection = None
        logger.info("✅ All models initialized (Hidden CoT).")

    def load_expanded_datasets(self) -> Tuple[List[Dict], List[Dict]]:
        logger.info("📊 Loading datasets...")
        all_samples = []
        dataset_paths = {
            "scannet": self.data_root / "scannet",
            "scannet_real": self.data_root / "scannet_real",
            "3d_front": self.data_root / "3d_front",
            "3d_front_real": self.data_root / "3d_front_real",
        }
        for name, p in dataset_paths.items():
            if not p.exists():
                continue
            scene_dirs = sorted([d for d in p.glob("*") if d.is_dir()])[: (50 if "real" in name else 30)]
            for scene_dir in scene_dirs:
                imgs = []
                for pat in ["*.jpg", "*.png", "*.jpeg"]:
                    imgs.extend(list(scene_dir.glob(pat)))
                    if (scene_dir / "images").exists():
                        imgs.extend(list((scene_dir / "images").glob(pat)))
                for img in sorted(set(imgs))[:10]:
                    if img.exists():
                        all_samples.append({"image_path": str(img), "scene_id": scene_dir.name, "dataset": name})
        seen = set()
        all_samples = [s for s in all_samples if s["image_path"] not in seen and not seen.add(s["image_path"])]
        random.shuffle(all_samples)
        if self.validation_split and self.validation_split > 0:
            split = int(len(all_samples) * (1 - self.validation_split))
            train_samples, val_samples = all_samples[:split], all_samples[split:]
        else:
            train_samples, val_samples = all_samples, []
        self.training_stats["total_samples"] = len(all_samples)
        self.training_stats["train_samples"] = len(train_samples)
        self.training_stats["val_samples"] = len(val_samples)
        self.training_stats["datasets_used"] = list(set(s["dataset"] for s in all_samples))
        logger.info("   Train: %d, Val: %d", len(train_samples), len(val_samples))
        return train_samples, val_samples

    def _tokenize_qa(self, question: str, answer: str):
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not set")
        q_enc = self.tokenizer(question, return_tensors="pt", truncation=True, max_length=MAX_QUESTION_LEN, add_special_tokens=True)
        question_ids = q_enc["input_ids"].to(self.device)
        a_ids = self.tokenizer(answer, truncation=True, max_length=MAX_ANSWER_LEN, add_special_tokens=False)["input_ids"]
        if getattr(self.tokenizer, "eos_token_id", None) and (not a_ids or a_ids[-1] != self.tokenizer.eos_token_id):
            a_ids.append(self.tokenizer.eos_token_id)
        a_t = torch.tensor([a_ids], device=self.device, dtype=torch.long)
        input_ids = torch.cat([question_ids, a_t], dim=1)
        answer_start_index = question_ids.size(1)
        return input_ids, answer_start_index

    def train_epoch(self, train_samples: List[Dict], epoch: int) -> float:
        self.student_model.train()
        if self.use_uncertainty_loss:
            self.uncertainty_loss.train()
        random.shuffle(train_samples)
        total_loss, num_batches = 0.0, 0
        total_samples = len(train_samples)
        logger.info("📊 CoT train on %d samples (accum_steps=%d, 10GB_mode=False)", total_samples, self.accumulation_steps)

        for i, sample in enumerate(train_samples):
            try:
                img_path = Path(sample["image_path"])
                if not img_path.exists():
                    continue
                image = Image.open(img_path).convert("RGB")
                image_tensor = transforms.ToTensor()(image).unsqueeze(0).float()
                if self.vision_device is not None:
                    image_tensor = image_tensor.to(self.vision_device)
                else:
                    image_tensor = image_tensor.to(self.device)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                teacher_response_dict = self.teacher_model.generate_response(DEFAULT_QUESTION, str(img_path))
                teacher_response = teacher_response_dict.get("response", str(teacher_response_dict)) if isinstance(teacher_response_dict, dict) else str(teacher_response_dict)
                input_ids, answer_start_index = self._tokenize_qa(DEFAULT_QUESTION, teacher_response)
                if input_ids.size(1) <= answer_start_index:
                    continue

                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    vision_outputs = self.student_model.vision_encoder(image_tensor)
                    vision_features_raw = vision_outputs.last_hidden_state.to(self.device)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                out = self.student_model.forward(
                    input_ids,
                    pixel_values=None,
                    answer_start_index=answer_start_index,
                    vision_features_precomputed=vision_features_raw,
                )
                text_loss = out.loss
                if text_loss is None:
                    continue
                vision_features = vision_features_raw.squeeze(1)
                det_logits = self.student_model.detection_head(vision_features)
                depth_logits = self.student_model.depth_head(vision_features)
                spatial_logits = self.student_model.spatial_head(vision_features)

                det_target = torch.zeros_like(det_logits, dtype=torch.float32)
                depth_ce_loss = depth_reg_loss = depth_kl_loss = None
                try:
                    if self.depth_teacher is not None:
                        depth_continuous, depth_discrete = self.depth_teacher.get_depth_labels(np.array(image), num_bins=3)
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
                            depth_hist = (depth_hist.astype(np.float32) + 1e-8) / (depth_hist.sum() + 1e-8)
                            depth_target_dist = torch.tensor(depth_hist, device=self.device, dtype=torch.float32).unsqueeze(0).clamp(min=1e-7)
                            depth_kl_loss = F.kl_div(torch.log(F.softmax(depth_logits, dim=-1).clamp(min=1e-7)), depth_target_dist, reduction="batchmean")
                except Exception:
                    pass
                detection_loss = F.binary_cross_entropy_with_logits(det_logits, det_target, reduction="mean")
                spatial_loss = None
                if self.object_detection is not None:
                    try:
                        comp = self.object_detection.detect_objects_comprehensive(image_tensor.to(self.device) if image_tensor.device.type != "cuda" else image_tensor)
                        dets = comp.get("detected_objects", []) or []
                        if len(dets) >= 2:
                            dets_s = sorted(dets, key=lambda x: x.get("confidence", 0.0), reverse=True)
                            a, b = dets_s[0], dets_s[1]
                            bbox = lambda o: o.get("bbox", [0, 0, 0, 0])
                            ax = (bbox(a)[0] + bbox(a)[2]) / 2.0
                            ay = (bbox(a)[1] + bbox(a)[3]) / 2.0
                            bx = (bbox(b)[0] + bbox(b)[2]) / 2.0
                            by = (bbox(b)[1] + bbox(b)[3]) / 2.0
                            lr_t = 0 if ax < bx else 1
                            ab_t = 0 if ay < by else 1
                            spatial_lr = F.cross_entropy(spatial_logits[:, :2], torch.tensor([lr_t], device=self.device, dtype=torch.long))
                            spatial_ab = F.cross_entropy(spatial_logits[:, 2:4], torch.tensor([ab_t], device=self.device, dtype=torch.long))
                            spatial_loss = 0.5 * (spatial_lr + spatial_ab)
                    except Exception:
                        pass
                multiview_loss = torch.tensor(2.5, device=self.device)
                feature_loss = torch.tensor(2.5, device=self.device)
                total_sample_loss = self.uncertainty_loss(
                    text_loss=text_loss,
                    depth_ce_loss=depth_ce_loss,
                    depth_reg_loss=depth_reg_loss,
                    depth_kl_loss=depth_kl_loss,
                    detection_loss=detection_loss,
                    spatial_loss=spatial_loss,
                    multiview_loss=multiview_loss,
                    feature_loss=feature_loss,
                )
                self.optimizers[0].zero_grad()
                total_sample_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), max_norm=1.0)
                self.optimizers[0].step()
                total_loss += total_sample_loss.item()
                num_batches += 1
                if num_batches % 10 == 0:
                    pct = 100.0 * (i + 1) / total_samples
                    w = self.uncertainty_loss.get_weights() if self.use_uncertainty_loss and num_batches % 50 == 0 else None
                    lg = "   Batch %d: %.6f (%.1f%%)" % (num_batches, total_loss / num_batches, pct)
                    if w:
                        lg += " | Weights: %s" % w
                    logger.info(lg)
            except Exception as e:
                if "CUDA" in str(e) or "out of memory" in str(e).lower():
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                continue
        if self.use_uncertainty_loss:
            self.training_stats["task_weights_history"].append({"epoch": epoch, "weights": self.uncertainty_loss.get_weights()})
        return total_loss / num_batches if num_batches > 0 else 0.0

    def validate(self, val_samples: List[Dict]) -> float:
        self.student_model.eval()
        if self.use_uncertainty_loss:
            self.uncertainty_loss.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for sample in val_samples:
                try:
                    img_path = Path(sample["image_path"])
                    if not img_path.exists():
                        continue
                    image = Image.open(img_path).convert("RGB")
                    image_tensor = transforms.ToTensor()(image).unsqueeze(0).float()
                    if self.vision_device is not None:
                        image_tensor = image_tensor.to(self.vision_device)
                    else:
                        image_tensor = image_tensor.to(self.device)
                    teacher_response_dict = self.teacher_model.generate_response(DEFAULT_QUESTION, str(img_path))
                    teacher_response = teacher_response_dict.get("response", str(teacher_response_dict)) if isinstance(teacher_response_dict, dict) else str(teacher_response_dict)
                    input_ids, answer_start_index = self._tokenize_qa(DEFAULT_QUESTION, teacher_response)
                    if input_ids.size(1) <= answer_start_index:
                        continue
                    with torch.amp.autocast("cuda", enabled=self.use_amp):
                        vision_outputs = self.student_model.vision_encoder(image_tensor)
                        vision_features_raw = vision_outputs.last_hidden_state.to(self.device)
                    out = self.student_model.forward(
                        input_ids,
                        pixel_values=None,
                        answer_start_index=answer_start_index,
                        vision_features_precomputed=vision_features_raw,
                    )
                    if out.loss is not None:
                        total += out.loss.item()
                        n += 1
                except Exception as e:
                    if n == 0:
                        logger.warning("   ⚠️ Validation sample failed (first error): %s", e)
                    continue
        if n == 0:
            logger.warning("   ⚠️ Validation: no sample succeeded (missing paths or device errors); Val: inf")
        return total / n if n > 0 else float("inf")

    def train(self):
        logger.info("🚀 Starting Hidden CoT training...")
        start = time.time()
        self.initialize_models()
        train_samples, val_samples = self.load_expanded_datasets()
        if not train_samples:
            logger.error("❌ No samples")
            return
        best_val, patience, last_val = float("inf"), 0, None
        for epoch in range(1, self.epochs + 1):
            logger.info("\n📅 Epoch %d/%d", epoch, self.epochs)
            train_loss = self.train_epoch(train_samples, epoch)
            self.scheduler.step()
            lr = self.optimizers[0].param_groups[0]["lr"]
            if val_samples:
                val_loss = self.validate(val_samples)
                last_val = val_loss
                logger.info("   Train: %.6f, Val: %.6f, LR: %.2e", train_loss, val_loss, lr)
                if val_loss < best_val:
                    best_val = val_loss
                    self.training_stats["best_val_loss"] = val_loss
                    self.training_stats["best_loss"] = train_loss
                    patience = 0
                    self._save("best", epoch, train_loss, val_loss)
                    logger.info("   ✅ Best val: %.6f", val_loss)
                else:
                    patience += 1
                if self.early_stopping_patience and patience >= self.early_stopping_patience:
                    logger.info("   ⏹️  Early stop")
                    self.training_stats["early_stopped"] = True
                    break
            else:
                logger.info("   Loss: %.6f, LR: %.2e", train_loss, lr)
                if train_loss < self.training_stats["best_loss"]:
                    self.training_stats["best_loss"] = train_loss
                    self._save("best", epoch, train_loss)
            self._save("epoch_%d" % epoch, epoch, train_loss, last_val)
            if epoch % 10 == 0:
                logger.info("   💾 Epoch checkpoint: cot_model_epoch_%d.pt", epoch)
        self.training_stats["training_time"] = time.time() - start
        self.training_stats["epochs_completed"] = epoch
        (self.checkpoint_dir / "cot_training_results.json").write_text(json.dumps(self.training_stats, indent=2))
        logger.info("✅ Hidden CoT training done.")

    def _save(self, name: str, epoch: int, loss: float, val_loss: float = None):
        p = self.checkpoint_dir / ("cot_model_%s.pt" % name)
        d = {
            "epoch": epoch,
            "model_state_dict": self.student_model.state_dict(),
            "optimizer_state_dict": [o.state_dict() for o in self.optimizers],
            "loss": loss,
            "training_stats": self.training_stats,
        }
        if val_loss is not None:
            d["val_loss"] = val_loss
        if self.use_uncertainty_loss:
            d["uncertainty_state_dict"] = self.uncertainty_loss.state_dict()
        torch.save(d, p)
        logger.info("💾 %s", p)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Hidden CoT training; optionally run K ablation via --num_thinking_tokens.")
    ap.add_argument("--num_thinking_tokens", type=int, default=None, help="Override K (default: 8 if GPU>=12GB else 4). Use for ablation: 2, 4, 8, 16.")
    ap.add_argument("--max_epochs", type=int, default=None, help="Cap training at this many epochs (for ablation).")
    ap.add_argument("--data_root", type=str, default=None)
    ap.add_argument("--checkpoint_dir", type=str, default=None)
    args = ap.parse_args()

    pipeline = HiddenCoTTrainingPipeline(use_uncertainty_loss=True)
    if args.data_root:
        pipeline.data_root = Path(args.data_root)
    if args.checkpoint_dir:
        pipeline.checkpoint_dir = Path(args.checkpoint_dir)
    if args.max_epochs is not None:
        pipeline.epochs = min(pipeline.epochs, args.max_epochs)

    if args.num_thinking_tokens is not None:
        K = max(1, min(64, args.num_thinking_tokens))
        pipeline.training_stats["num_thinking_tokens"] = K
        # Set K on config before building model (done in initialize_models)
        pipeline._num_thinking_tokens_override = K
    else:
        pipeline._num_thinking_tokens_override = None

    pipeline.train()


if __name__ == "__main__":
    main()
