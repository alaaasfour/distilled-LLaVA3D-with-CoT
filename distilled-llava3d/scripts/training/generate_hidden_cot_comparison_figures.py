#!/usr/bin/env python3
"""
Generate a figure: comparison of output *without* Hidden CoT (final answer only)
vs *with* Hidden CoT in Diagnostic Mode (decoded thinking tokens T1..T8 + final answer).
Uses an image from data/3d_front_real or a generated placeholder.

"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def find_sample_image():
    """Find first image under data/3d_front_real."""
    base = ROOT / "data" / "scannet_real"
    if not base.exists():
        return None
    for ext in ("*.jpg", "*.png", "*.jpeg"):
        for p in base.rglob(ext):
            if p.is_file():
                return str(p)
    return None


def create_placeholder_image(size=(400, 300)):
    """Create a simple placeholder that looks like a 3D scene (room/floor)."""
    h, w = size
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            arr[y, x, 0] = int(80 + 40 * (y / h))
            arr[y, x, 1] = int(70 + 35 * (y / h))
            arr[y, x, 2] = int(60 + 30 * (y / h))
    arr[: h // 4, :] = [180, 175, 170]
    y1, y2 = int(h * 0.5), int(h * 0.85)
    x1, x2 = int(w * 0.3), int(w * 0.7)
    arr[y1:y2, x1:x2] = [120, 100, 90]
    return Image.fromarray(arr)


def get_example_data(question):
    """Example content for the figure (no model run)."""
    return {
        "answer_only": "This appears to be an indoor scene with various elements arranged in space. I can identify furniture and room features.",
        "thinking_lines": [
            "T1: scene",
            "T2: indoor",
            "T3: furniture",
            "T4: spatial",
            "T5: layout",
            "T6: objects",
            "T7: room",
            "T8: elements",
        ],
        "answer_with_cot": "This appears to be an indoor scene with various elements arranged in space. I can identify furniture and room features.",
        "question": question,
    }


def run_model_and_get_data(image_path, checkpoint_path, device, question):
    """Run model to get real answers and decoded thinking tokens."""
    import torch
    import torchvision.transforms as transforms
    from scripts.distillation.student_model import DistilledLLaVA3D, DistilledLLaVA3DConfig

    config = DistilledLLaVA3DConfig()
    config.num_thinking_tokens = 8
    model = DistilledLLaVA3D(config)
    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    tokenizer = load_tokenizer()
    pil_image = Image.open(image_path).convert("RGB")
    image_tensor = transforms.ToTensor()(pil_image).unsqueeze(0).float()
    if device != "cpu":
        image_tensor = image_tensor.to(device)

    with torch.no_grad():
        answer_only = model.generate_response(question, image_tensor)

    q_enc = tokenizer(question, return_tensors="pt", truncation=True, max_length=64, add_special_tokens=True)
    question_ids = q_enc["input_ids"].to(device)
    answer_start_index = question_ids.size(1)
    input_ids = question_ids
    with torch.no_grad():
        vision_outputs = model.vision_encoder(image_tensor)
        vision_features = vision_outputs.last_hidden_state
        decoded_tokens = model.decode_thinking_tokens(
            input_ids,
            answer_start_index=answer_start_index,
            tokenizer=tokenizer,
            vision_features_precomputed=vision_features,
            skip_special_tokens=True,
        )
    thinking_lines = [f"T{k+1}: {decoded_tokens[k] if k < len(decoded_tokens) else ''}" for k in range(8)]

    return {
        "answer_only": answer_only,
        "thinking_lines": thinking_lines,
        "answer_with_cot": answer_only,
        "question": question,
    }


def load_tokenizer():
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=False)
    except Exception:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("bert-base-uncased")


def build_figure(pil_image, data, output_path):
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    rcParams["font.size"] = 10
    fig = plt.figure(figsize=(12, 8))
    fig.patch.set_facecolor("white")

    ax_img = fig.add_axes([0.02, 0.15, 0.35, 0.7])
    ax_without = fig.add_axes([0.42, 0.45, 0.55, 0.5])
    ax_with = fig.add_axes([0.42, 0.05, 0.55, 0.38])

    ax_img.imshow(pil_image)
    ax_img.set_title("Input: 3D scene (SCANNET)", fontsize=11)
    ax_img.axis("off")

    ax_without.set_title("Without Hidden CoT (production)", fontsize=11, fontweight="bold")
    ax_without.text(0.02, 0.95, "Only the final answer is returned:", transform=ax_without.transAxes, fontsize=10, verticalalignment="top", wrap=True)
    ax_without.text(0.02, 0.72, data["answer_only"], transform=ax_without.transAxes, fontsize=9, verticalalignment="top", wrap=True, style="italic")
    ax_without.set_xlim(0, 1)
    ax_without.set_ylim(0, 1)
    ax_without.axis("off")

    ax_with.set_title("With Hidden CoT + Diagnostic Mode", fontsize=11, fontweight="bold")
    thinking_text = "\n".join(data["thinking_lines"])
    ax_with.text(0.02, 0.95, "Decoded thinking tokens (T1..T8):", transform=ax_with.transAxes, fontsize=9, verticalalignment="top")
    ax_with.text(0.02, 0.68, thinking_text, transform=ax_with.transAxes, fontsize=8, verticalalignment="top", family="monospace")
    ax_with.text(0.02, 0.32, "Final answer:", transform=ax_with.transAxes, fontsize=9, verticalalignment="top")
    ax_with.text(0.02, 0.08, data["answer_with_cot"], transform=ax_with.transAxes, fontsize=9, verticalalignment="top", wrap=True, style="italic")
    ax_with.set_xlim(0, 1)
    ax_with.set_ylim(0, 1)
    ax_with.axis("off")

    fig.suptitle(f'Question: "{data["question"]}"', fontsize=12, y=0.98)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Hidden CoT vs Diagnostic comparison figure.")
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--question", type=str, default="Describe this 3D scene and identify objects.")
    parser.add_argument("--run_model", action="store_true", help="Run model to fill text (slow). Default: use example text.")
    parser.add_argument("--device", type=str, default="cuda:0" if __import__("torch").cuda.is_available() else "cpu")
    args = parser.parse_args()

    image_path = args.image
    if not image_path or not Path(image_path).exists():
        candidate = find_sample_image()
        if candidate:
            image_path = candidate
            print(f"Using image from SCANNET: {image_path}")
        else:
            out_dir = ROOT / "results" / "figures"
            out_dir.mkdir(parents=True, exist_ok=True)
            placeholder_path = out_dir / "sample_3d_scene_placeholder.png"
            create_placeholder_image().save(placeholder_path)
            image_path = str(placeholder_path)
            print(f"No image in data/scannet; using placeholder: {image_path}")

    pil_image = Image.open(image_path).convert("RGB")

    if args.run_model:
        data = run_model_and_get_data(image_path, args.checkpoint, args.device, args.question)
    else:
        data = get_example_data(args.question)

    output_path = args.output or str(ROOT / "results" / "figures" / "hidden_cot_comparison.png")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    build_figure(pil_image, data, output_path)


if __name__ == "__main__":
    main()
