#!/usr/bin/env python3
"""Command-line interface for the distilled LLaVA-3D student model."""

from __future__ import annotations

import argparse
import os
import sys
from io import BytesIO
from pathlib import Path
from typing import Optional

import requests
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.distillation.student_model import DistilledLLaVA3D, DistilledLLaVA3DConfig
from scripts.distillation.load_teacher import load_llava3d_teacher


class DistilledLLaVA3DCLI:
    """Utility wrapper around the distilled model for quick interactive tests."""

    def __init__(self, model_path: Optional[str] = None, device: str = "cuda") -> None:
        resolved_device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.device = resolved_device
        self.model: Optional[DistilledLLaVA3D] = None
        self.config: Optional[DistilledLLaVA3DConfig] = None

        if model_path is not None:
            self.load_model(model_path)

    def load_model(self, model_path: str) -> None:
        print(f"📚 Loading distilled model from {model_path}...")
        checkpoint_path = self._resolve_checkpoint_path(model_path)

        teacher_bundle = load_llava3d_teacher(
            model_path="ChaimZhu/LLaVA-3D-7B",
            device=str(self.device),
            precision="bf16",
            quant="4bit" if self.device.type == "cuda" else None,
        )

        self.config = DistilledLLaVA3DConfig()
        self.model = DistilledLLaVA3D.from_teacher(teacher_bundle, config=self.config)
        self.model.to(self.device)
        self.model.eval()

        # Release the teacher language backbone once multimodal pieces are bound.
        del teacher_bundle

        print(f"🔧 Processor type: {type(self.model.processor)}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        missing, unexpected = self.model.load_state_dict(checkpoint, strict=False)
        if missing:
            print(f"⚠️  Missing parameters in checkpoint: {len(missing)} entries")
        if unexpected:
            print(f"⚠️  Unexpected parameters in checkpoint: {len(unexpected)} entries")

        print("✅ Model loaded successfully!")
        print("📊 Model loaded from checkpoint")
        print(f"🔢 Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def _resolve_checkpoint_path(self, model_path: str) -> str:
        candidate = Path(model_path)
        if candidate.is_dir():
            checkpoints = sorted(p for p in candidate.iterdir() if p.suffix == ".pt")
            if not checkpoints:
                raise FileNotFoundError(f"No checkpoints found in {model_path}")
            return str(checkpoints[-1])
        if not candidate.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        return str(candidate)

    def load_image(self, image_path: str) -> Optional[Image.Image]:
        try:
            if image_path.startswith("http"):
                response = requests.get(image_path)
                response.raise_for_status()
                return Image.open(BytesIO(response.content)).convert("RGB")
            return Image.open(image_path).convert("RGB")
        except Exception as exc:  # pylint: disable=broad-except
            print(f"❌ Error loading image: {exc}")
            return None

    def generate_response(self, image: Image.Image, query: str) -> str:
        if self.model is None:
            return "Error: Model not loaded"
        try:
            with torch.no_grad():
                return self.model.generate_response(query, image)
        except Exception as exc:  # pylint: disable=broad-except
            import traceback
            traceback.print_exc()
            return f"Error generating response: {exc}"

    def run_inference(self, image_path: str, query: str) -> None:
        print(f"🖼️  Image: {image_path}")
        print(f"❓ Query: {query}")
        print("-" * 50)

        image = self.load_image(image_path)
        if image is None:
            return

        response = self.generate_response(image, query)
        print("🤖 Distilled LLaVA-3D Response:")
        print(f"   {response}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Distilled LLaVA-3D Command Line Interface")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/checkpoints",
        help="Path to a checkpoint file or directory containing checkpoints",
    )
    parser.add_argument("--image-file", type=str, required=True, help="Path or URL of the image")
    parser.add_argument("--query", type=str, required=True, help="Question to ask about the image")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    args = parser.parse_args()

    cli = DistilledLLaVA3DCLI(args.model_path, args.device)
    cli.run_inference(args.image_file, args.query)


if __name__ == "__main__":
    main()
