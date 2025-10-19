#!/usr/bin/env python3
"""Dataset loader for LLaVA-3D distillation with real 3D data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image
from torch.utils.data import Dataset, DataLoader


class LLaVA3DDataset(Dataset):
    """Lightweight dataset that surfaces image/question pairs for distillation."""

    def __init__(
        self,
        data_manifest: Optional[Path],
        tokenizer: Any,
        processor: Any,
        root_dir: Optional[Path] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.processor = processor
        self.root_dir = Path(root_dir) if root_dir else None
        self.samples = self._load_manifest(data_manifest)

    def _load_manifest(self, manifest_path: Optional[Path]) -> List[Dict[str, Any]]:
        if manifest_path and manifest_path.is_file():
            with open(manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)

        # Fallback manifest built from demo assets.
        demo_dir = Path("/scratch/alasfour/llava-3d/LLaVA-3D/demo/my_images")
        questions = [
            "What objects are visible in this scene?",
            "Describe the layout of this environment.",
            "What should I be cautious about here?",
        ]
        samples: List[Dict[str, Any]] = []
        for image_path in sorted(demo_dir.glob("*.png")):
            for question in questions:
                samples.append(
                    {
                        "image_path": str(image_path),
                        "question": question,
                        "depth_path": None,
                        "meta": {"source": "demo"},
                    }
                )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        image_path = Path(sample["image_path"])
        if self.root_dir and not image_path.is_absolute():
            image_path = self.root_dir / image_path

        image = Image.open(image_path).convert("RGB")

        depth_path = sample.get("depth_path")
        depth = None
        if depth_path:
            depth_path = Path(depth_path)
            if self.root_dir and not depth_path.is_absolute():
                depth_path = self.root_dir / depth_path
            if depth_path.is_file():
                depth = Image.open(depth_path)

        return {
            "image": image,
            "depth": depth,
            "question": sample["question"],
            "answer": sample.get("answer"),
            "meta": sample.get("meta", {}),
        }


def _collate_samples(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "images": [item["image"] for item in batch],
        "depths": [item["depth"] for item in batch],
        "questions": [item["question"] for item in batch],
        "answers": [item.get("answer") for item in batch],
        "metas": [item.get("meta", {}) for item in batch],
    }


def create_dataloader(
    data_dir: Optional[str],
    tokenizer: Any,
    processor: Any,
    batch_size: int = 4,
    num_workers: int = 0,
    manifest: Optional[str] = None,
):
    """Create DataLoader that yields samples requiring further processing."""

    dataset = LLaVA3DDataset(
        data_manifest=Path(manifest) if manifest else None,
        tokenizer=tokenizer,
        processor=processor,
        root_dir=Path(data_dir) if data_dir else None,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_collate_samples,
    )

if __name__ == "__main__":
    print("Dataset loader created successfully!")
