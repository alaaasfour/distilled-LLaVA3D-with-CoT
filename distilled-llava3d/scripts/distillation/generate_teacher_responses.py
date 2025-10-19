#!/usr/bin/env python3
"""Generate teacher responses for RGB-D question prompts.

Reads a manifest of samples containing at least `image_path` and `question`,
optionally `depth_path`, runs the LLaVA-3D teacher to obtain textual answers,
and writes out a new manifest that includes the teacher's response.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from PIL import Image

LLAVA3D_REPO = Path("/scratch/alasfour/llava-3d/LLaVA-3D")
if LLAVA3D_REPO.is_dir():
    repo_path = str(LLAVA3D_REPO)
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

from llava.conversation import conv_templates
from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.mm_utils import process_images, tokenizer_special_token

from load_teacher import load_llava3d_teacher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate teacher pseudo-labels")
    parser.add_argument(
        "--input-manifest",
        type=Path,
        required=True,
        help="JSON manifest with samples (image_path, question, optional depth_path).",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        required=True,
        help="Where to write the manifest with teacher answers.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Optional root directory that input paths are relative to.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for teacher inference (cuda or cpu).",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp16",
        choices=["fp16", "bf16", "fp32"],
        help="Teacher precision.",
    )
    parser.add_argument(
        "--quant",
        type=str,
        default="4bit",
        choices=["4bit", "8bit", "none"],
        help="Quantization mode for teacher weights.",
    )
    parser.add_argument(
        "--conversation-template",
        type=str,
        default="llava_v1",
        help="Conversation template key (e.g., llava_v1).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for teacher inference (images are processed sequentially per batch).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum tokens to generate per response.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 for greedy).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Nucleus sampling top-p (None for greedy).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="ChaimZhu/LLaVA-3D-7B",
        help="Teacher model identifier (HF repo or local path).",
    )
    return parser.parse_args()


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def generate_batch(
    teacher_model,
    tokenizer,
    processor,
    device: torch.device,
    images: List[Image.Image],
    prompts: List[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float | None,
) -> List[str]:
    if isinstance(processor, dict):
        image_processor = processor.get("image")
    else:
        image_processor = getattr(processor, "image_processor", None)

    if image_processor is None:
        vision_tower = getattr(teacher_model.get_model(), "vision_tower", None)
        image_processor = getattr(vision_tower, "image_processor", None)

    pixel_values = process_images(images, image_processor, teacher_model.config)
    if isinstance(pixel_values, list):
        pixel_values = torch.stack(pixel_values, dim=0)
    if hasattr(pixel_values, "pixel_values"):
        pixel_values = pixel_values.pixel_values

    model_dtype = next(teacher_model.parameters()).dtype
    pixel_values = pixel_values.to(device=device, dtype=model_dtype)

    token_ids = [tokenizer_special_token(p, tokenizer, return_tensors="pt") for p in prompts]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        token_ids,
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    ).to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)

    with torch.inference_mode():
        output_ids = teacher_model.generate(
            inputs=input_ids,
            attention_mask=attention_mask,
            images=pixel_values,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )

    answers = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return [ans.strip() for ans in answers]


def main() -> None:
    args = parse_args()
    use_cuda = torch.cuda.is_available() and args.device.startswith("cuda")
    device = torch.device("cuda" if use_cuda else "cpu")

    quant = None if args.quant.lower() == "none" else args.quant
    precision = args.precision
    if device.type == "cpu":
        quant = None
        precision = "fp32"

    tokenizer, model, processor, _ = load_llava3d_teacher(
        model_path=args.model_path,
        device=str(device),
        precision=precision,
        quant=quant,
    )

    if device.type == "cpu":
        model = model.float()
    model.eval()

    
    input_manifest = json.loads(args.input_manifest.read_text("utf-8"))
    root = args.data_root if args.data_root else Path(".")
    root = Path(root)

    image_token = DEFAULT_IMAGE_TOKEN
    if model.config.mm_use_im_start_end:
        image_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN

    conv_template = conv_templates[args.conversation_template]

    augmented_samples: List[Dict[str, Any]] = []
    batch_images: List[Image.Image] = []
    batch_prompts: List[str] = []
    batch_indices: List[int] = []

    for idx, sample in enumerate(input_manifest):
        image_path = Path(sample["image_path"])
        if not image_path.is_absolute():
            image_path = root / image_path

        image = load_image(image_path)
        question = sample["question"]
        prompt = question
        if image_token not in prompt:
            prompt = f"{image_token}\n{prompt}"

        conv = conv_template.copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        full_prompt = conv.get_prompt()

        batch_images.append(image)
        batch_prompts.append(full_prompt)
        batch_indices.append(idx)

        if len(batch_images) == args.batch_size:
            responses = generate_batch(
                model,
                tokenizer,
                processor,
                device,
                batch_images,
                batch_prompts,
                args.max_new_tokens,
                args.temperature,
                args.top_p,
            )
            for sample_idx, answer in zip(batch_indices, responses):
                augmented = dict(input_manifest[sample_idx])
                augmented["answer"] = answer
                augmented_samples.append(augmented)
            batch_images.clear()
            batch_prompts.clear()
            batch_indices.clear()

    if batch_images:
        responses = generate_batch(
            model,
            tokenizer,
            processor,
            device,
            batch_images,
            batch_prompts,
            args.max_new_tokens,
            args.temperature,
            args.top_p,
        )
        for sample_idx, answer in zip(batch_indices, responses):
            augmented = dict(input_manifest[sample_idx])
            augmented["answer"] = answer
            augmented_samples.append(augmented)

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(augmented_samples, indent=2), encoding="utf-8")
    print(f"Wrote {len(augmented_samples)} samples to {args.output_manifest}")


if __name__ == "__main__":
    main()
