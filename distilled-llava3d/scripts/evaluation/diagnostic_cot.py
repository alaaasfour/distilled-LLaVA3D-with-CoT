#!/usr/bin/env python3
"""
Diagnostic mode for Hidden CoT: optionally decode thinking tokens to human-readable text.
Use this to verify whether the model is reasoning spatially or exploiting statistical biases.
Run only in development/debug; not for deployment.

"""
import argparse
import sys
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image

# Project root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.distillation.student_model import DistilledLLaVA3D, DistilledLLaVA3DConfig


def load_tokenizer():
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=False)
    except Exception:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("bert-base-uncased")


def main():
    parser = argparse.ArgumentParser(description="Decode Hidden CoT thinking tokens (diagnostic mode).")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to CoT checkpoint (e.g. cot_model_best.pt)")
    parser.add_argument("--image", type=str, default=None, help="Path to image (optional; if omitted uses dummy input)")
    parser.add_argument("--question", type=str, default="Describe this 3D scene and identify objects.", help="Question text")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_question_len", type=int, default=64)
    args = parser.parse_args()

    device = args.device
    if args.checkpoint and not Path(args.checkpoint).exists():
        print(f"Checkpoint not found: {args.checkpoint}")
        print("Run without --checkpoint to test with randomly initialized model (decoded tokens will be meaningless).")
        sys.exit(1)

    config = DistilledLLaVA3DConfig()
    config.num_thinking_tokens = getattr(config, "num_thinking_tokens", 8)
    model = DistilledLLaVA3D(config)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)
        print(f"Loaded checkpoint: {args.checkpoint}")
    model.to(device)
    model.eval()

    tokenizer = load_tokenizer()
    q_enc = tokenizer(args.question, return_tensors="pt", truncation=True, max_length=args.max_question_len, add_special_tokens=True)
    question_ids = q_enc["input_ids"].to(device)
    answer_start_index = question_ids.size(1)
    input_ids = question_ids

    if args.image and Path(args.image).exists():
        image = Image.open(args.image).convert("RGB")
        image_tensor = transforms.ToTensor()(image).unsqueeze(0).float().to(device)
        vision_outputs = model.vision_encoder(image_tensor)
        vision_features = vision_outputs.last_hidden_state
    else:
        batch_size = 1
        vision_features = torch.randn(batch_size, 1, model.config.vision_hidden_size, device=device) * 0.02
        if args.image:
            print(f"Warning: image not found {args.image}, using random features.")

    with torch.no_grad():
        decoded = model.decode_thinking_tokens(
            input_ids,
            answer_start_index=answer_start_index,
            tokenizer=tokenizer,
            vision_features_precomputed=vision_features,
            skip_special_tokens=True,
        )

    K = model.num_thinking_tokens
    print("\n--- Hidden CoT diagnostic mode: decoded thinking tokens ---")
    print(f"Question: {args.question}")
    print(f"Number of thinking tokens (K): {K}")
    print("Decoded text at each position (argmax over vocab):")
    for k in range(K):
        text = decoded[k] if k < len(decoded) else ""
        print(f"  T{k+1}: {repr(text)}")
    print("(Interpretation: these are the model's argmax token at each thinking position; may be subword or special.)")
    print("--- end diagnostic ---\n")


if __name__ == "__main__":
    main()
