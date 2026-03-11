#!/usr/bin/env python3
"""
Generate training loss convergence chart for Hidden CoT training (5 epochs).
Shows training loss and validation loss over epochs, with best validation point marked.

"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="Generate training loss convergence chart.")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    # Data from cot_train_8054322.err (5 epochs completed)
    epochs = np.array([1, 2, 3, 4, 5])
    train_loss = np.array([7.925547, 7.486521, 7.448448, 7.426996, 7.389563])
    val_loss = np.array([4.930106, 4.948332, 4.914103, 4.789919, 5.054531])
    best_val_epoch = 4
    best_val_loss = 4.789919

    rcParams["font.size"] = 11
    rcParams["font.family"] = "serif"
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")

    # Plot lines
    ax.plot(epochs, train_loss, marker="o", linewidth=2, markersize=8, label="Training Loss", color="#2E86AB")
    ax.plot(epochs, val_loss, marker="s", linewidth=2, markersize=8, label="Validation Loss", color="#A23B72")

    # Mark best validation point
    ax.plot(
        best_val_epoch,
        best_val_loss,
        marker="*",
        markersize=16,
        color="#F18F01",
        markeredgecolor="black",
        markeredgewidth=1.5,
        label=f"Best Val Loss: {best_val_loss:.4f} (Epoch {best_val_epoch})",
        zorder=10,
    )

    # Styling
    ax.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax.set_ylabel("Loss", fontsize=12, fontweight="bold")
    ax.set_title("Hidden CoT Training Loss Convergence (5 Epochs)", fontsize=13, fontweight="bold", pad=10)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper right", frameon=True, fancybox=True, shadow=True, fontsize=10)
    ax.set_xticks(epochs)
    ax.set_xlim(0.5, 5.5)

    # Add text annotation for best point
    ax.annotate(
        f"Best: {best_val_loss:.4f}",
        xy=(best_val_epoch, best_val_loss),
        xytext=(best_val_epoch + 0.3, best_val_loss + 0.15),
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", color="black", lw=1.5),
    )

    plt.tight_layout()
    output_path = args.output or str(ROOT / "results" / "figures" / "training_loss_convergence.png")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")
    print(f"Training loss: {train_loss[0]:.4f} → {train_loss[-1]:.4f} ({((train_loss[0] - train_loss[-1]) / train_loss[0] * 100):.1f}% reduction)")
    print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_val_epoch}")


if __name__ == "__main__":
    main()
