#!/usr/bin/env python3
"""
Ablation over number of Hidden CoT thinking tokens K.
Runs training with K in {2, 4, 8, 16} (or custom list), each in a separate checkpoint dir,
then prints a table: K | Best Val Loss | Notes.

Usage:
  python scripts/training/run_cot_ablation_k.py [--k 2 4 8 16] [--epochs 3] [--data_root ...] [--base_dir ...]
  Or run each K manually: python train_cot.py --num_thinking_tokens 4 --checkpoint_dir checkpoints/cot_k4
  Then: python scripts/training/run_cot_ablation_k.py --collect_only --base_dir checkpoints
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run_one_k(k: int, max_epochs: int, data_root: str, base_dir: Path) -> dict:
    ckpt_dir = base_dir / f"cot_k{k}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT / "train_cot.py"),
        "--num_thinking_tokens", str(k),
        "--checkpoint_dir", str(ckpt_dir),
        "--max_epochs", str(max_epochs),
    ]
    if data_root:
        cmd += ["--data_root", data_root]
    try:
        subprocess.run(cmd, cwd=str(ROOT), check=False, timeout=3600 * 2)
    except subprocess.TimeoutExpired:
        pass
    result_path = ckpt_dir / "cot_training_results.json"
    if result_path.exists():
        data = json.loads(result_path.read_text())
        return {"K": k, "best_val_loss": data.get("best_val_loss", float("nan")), "path": str(ckpt_dir)}
    return {"K": k, "best_val_loss": float("nan"), "path": str(ckpt_dir)}


def collect_only(base_dir: Path) -> list:
    results = []
    for d in sorted(base_dir.iterdir()):
        if not d.is_dir() or not d.name.startswith("cot_k"):
            continue
        try:
            k = int(d.name.replace("cot_k", ""))
        except ValueError:
            continue
        result_path = d / "cot_training_results.json"
        if result_path.exists():
            data = json.loads(result_path.read_text())
            results.append({"K": k, "best_val_loss": data.get("best_val_loss", float("nan")), "path": str(d)})
    return sorted(results, key=lambda x: x["K"])


def main():
    ap = argparse.ArgumentParser(description="Hidden CoT K ablation: train with K=2,4,8,16 and collect best val loss.")
    ap.add_argument("--k", type=int, nargs="+", default=[2, 4, 8, 16], help="List of K values")
    ap.add_argument("--max_epochs", type=int, default=3, help="Max epochs per K (for quick ablation).")
    ap.add_argument("--data_root", type=str, default=None)
    ap.add_argument("--base_dir", type=str, default=None, help="Base checkpoint dir; each K gets base_dir/cot_k{K}")
    ap.add_argument("--collect_only", action="store_true", help="Only collect existing results from base_dir/cot_k*")
    args = ap.parse_args()

    base_dir = Path(args.base_dir or ROOT / "checkpoints" / "cot_ablation_k")
    base_dir.mkdir(parents=True, exist_ok=True)

    if args.collect_only:
        results = collect_only(base_dir)
    else:
        results = []
        for k in args.k:
            r = run_one_k(k, args.max_epochs, args.data_root, base_dir)
            results.append(r)

    print("\n--- Ablation: Number of Thinking Tokens (K) vs Best Validation Loss ---")
    print("| K | Best Val Loss | Checkpoint Dir |")
    print("|---|---------------|----------------|")
    for r in results:
        bvl = r.get("best_val_loss", float("nan"))
        bvl_str = f"{bvl:.6f}" if isinstance(bvl, (int, float)) and bvl == bvl else "N/A"
        print(f"| {r['K']} | {bvl_str} | {r.get('path', '')} |")
    print("---")
    return results


if __name__ == "__main__":
    main()
