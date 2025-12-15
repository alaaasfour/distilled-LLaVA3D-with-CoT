#!/usr/bin/env python3
"""
Analyze training loss curves from training logs.
Extracts and visualizes training/validation loss over epochs.
"""

import re
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_training_log(log_file: Path) -> Dict:
    """Parse training log file to extract loss curves."""
    train_losses = []
    val_losses = []
    epochs = []
    batch_losses = []  # For detailed batch-level analysis
    
    # Try .err file if .out doesn't have data
    err_file = log_file.parent / (log_file.stem + '.err')
    if not log_file.exists() and err_file.exists():
        log_file = err_file
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Extract epoch-level losses - look for pattern like "📅 Epoch 1/50"
    epoch_pattern = r'📅 Epoch (\d+)/\d+'
    train_loss_pattern = r'Train Loss: ([\d.]+)'
    val_loss_pattern = r'Val Loss: ([\d.]+)'
    
    # Find all epochs
    epoch_matches = list(re.finditer(epoch_pattern, content))
    
    for epoch_match in epoch_matches:
        epoch_num = int(epoch_match.group(1))
        start_pos = epoch_match.end()
        
        # Find train and val loss for this epoch (look ahead)
        section = content[start_pos:start_pos+10000]  # Look ahead 10000 chars
        
        train_match = re.search(train_loss_pattern, section)
        val_match = re.search(val_loss_pattern, section)
        
        if train_match:
            train_loss = float(train_match.group(1))
            train_losses.append(train_loss)
            epochs.append(epoch_num)
        
        if val_match:
            val_loss = float(val_match.group(1))
            val_losses.append(val_loss)
    
    # Extract batch-level losses for detailed analysis
    batch_pattern = r'Batch \d+: \d+/\d+.*?Avg Loss: ([\d.]+)'
    batch_matches = re.findall(batch_pattern, content)
    batch_losses = [float(loss) for loss in batch_matches]
    
    return {
        'epochs': epochs,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'batch_losses': batch_losses
    }


def plot_loss_curves(data: Dict, output_dir: Path):
    """Plot training and validation loss curves."""
    epochs = data['epochs']
    train_losses = data['train_losses']
    val_losses = data['val_losses']
    batch_losses = data['batch_losses']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Epoch-level losses
    ax1 = axes[0]
    if epochs and train_losses:
        ax1.plot(epochs, train_losses, 'b-o', label='Training Loss', linewidth=2, markersize=8)
    if epochs and val_losses:
        ax1.plot(epochs, val_losses, 'r-s', label='Validation Loss', linewidth=2, markersize=8)
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss Curves', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)
    
    # Plot 2: Batch-level training loss (smoothed)
    ax2 = axes[1]
    if batch_losses:
        # Smooth the batch losses for better visualization
        window_size = min(50, len(batch_losses) // 10)
        if window_size > 1:
            smoothed = np.convolve(batch_losses, np.ones(window_size)/window_size, mode='valid')
            batch_indices = np.arange(len(smoothed))
            ax2.plot(batch_indices, smoothed, 'g-', label=f'Smoothed Training Loss (window={window_size})', linewidth=2, alpha=0.7)
        else:
            ax2.plot(batch_losses, 'g-', label='Training Loss (per batch)', linewidth=1, alpha=0.5)
    
    ax2.set_xlabel('Batch Index', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Batch-Level Training Loss', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_file = output_dir / 'training_loss_curves.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"✅ Saved loss curves to {output_file}")
    
    plt.close()


def generate_analysis_report(data: Dict, output_file: Path):
    """Generate text analysis report."""
    epochs = data['epochs']
    train_losses = data['train_losses']
    val_losses = data['val_losses']
    batch_losses = data['batch_losses']
    
    report = []
    report.append("=" * 60)
    report.append("TRAINING LOSS CURVE ANALYSIS")
    report.append("=" * 60)
    report.append("")
    
    # Epoch-level statistics
    if epochs and train_losses:
        report.append("📊 Epoch-Level Statistics:")
        report.append(f"  Total Epochs: {len(epochs)}")
        report.append(f"  Initial Training Loss: {train_losses[0]:.6f}")
        report.append(f"  Final Training Loss: {train_losses[-1]:.6f}")
        report.append(f"  Training Loss Reduction: {((train_losses[0] - train_losses[-1]) / train_losses[0] * 100):.2f}%")
        report.append(f"  Average Training Loss: {np.mean(train_losses):.6f}")
        report.append(f"  Min Training Loss: {np.min(train_losses):.6f}")
        report.append("")
    
    if epochs and val_losses:
        report.append("📊 Validation Statistics:")
        report.append(f"  Initial Validation Loss: {val_losses[0]:.6f}")
        report.append(f"  Final Validation Loss: {val_losses[-1]:.6f}")
        if len(val_losses) > 1:
            val_change = val_losses[-1] - val_losses[0]
            val_change_pct = (val_change / val_losses[0] * 100) if val_losses[0] > 0 else 0
            report.append(f"  Validation Loss Change: {val_change:+.6f} ({val_change_pct:+.2f}%)")
        report.append(f"  Average Validation Loss: {np.mean(val_losses):.6f}")
        report.append(f"  Min Validation Loss: {np.min(val_losses):.6f}")
        report.append("")
    
    # Overfitting analysis
    if train_losses and val_losses and len(train_losses) == len(val_losses):
        report.append("🔍 Overfitting Analysis:")
        gaps = [val - train for train, val in zip(train_losses, val_losses)]
        report.append(f"  Average Train-Val Gap: {np.mean(gaps):.6f}")
        report.append(f"  Final Train-Val Gap: {gaps[-1]:.6f}")
        if len(gaps) > 1:
            gap_trend = gaps[-1] - gaps[0]
            report.append(f"  Gap Trend: {gap_trend:+.6f} (increasing = overfitting)")
        report.append("")
    
    # Batch-level statistics
    if batch_losses:
        report.append("📊 Batch-Level Statistics:")
        report.append(f"  Total Batches: {len(batch_losses)}")
        report.append(f"  Initial Batch Loss: {batch_losses[0]:.6f}")
        report.append(f"  Final Batch Loss: {batch_losses[-1]:.6f}")
        report.append(f"  Average Batch Loss: {np.mean(batch_losses):.6f}")
        report.append(f"  Loss Std Dev: {np.std(batch_losses):.6f}")
        report.append("")
    
    # Recommendations
    report.append("💡 Recommendations:")
    if train_losses and val_losses:
        if len(val_losses) > 1 and val_losses[-1] > val_losses[0]:
            report.append("  ⚠️  Validation loss is increasing - consider early stopping or regularization")
        if train_losses[-1] > 0.1:
            report.append("  📈 Training loss is still high - model may benefit from more training")
        if train_losses and val_losses and (val_losses[-1] - train_losses[-1]) > 0.5:
            report.append("  ⚠️  Large gap between train and validation loss - possible overfitting")
    
    report.append("")
    report.append("=" * 60)
    
    # Save report
    with open(output_file, 'w') as f:
        f.write('\n'.join(report))
    
    logger.info(f"✅ Saved analysis report to {output_file}")
    
    # Print to console
    print('\n'.join(report))


def main():
    parser = argparse.ArgumentParser(description='Analyze training loss curves')
    parser.add_argument('--log_file', type=str, required=True,
                       help='Path to training log file')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for plots and reports')
    
    args = parser.parse_args()
    
    log_file = Path(args.log_file)
    if not log_file.exists():
        logger.error(f"Log file not found: {log_file}")
        return
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse log file
    logger.info(f"Parsing training log: {log_file}")
    data = parse_training_log(log_file)
    
    # Generate plots
    logger.info("Generating loss curves...")
    plot_loss_curves(data, output_dir)
    
    # Generate analysis report
    logger.info("Generating analysis report...")
    report_file = output_dir / 'training_analysis_report.txt'
    generate_analysis_report(data, report_file)
    
    # Save data as JSON
    json_file = output_dir / 'training_curves_data.json'
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=2)
    logger.info(f"✅ Saved data to {json_file}")


if __name__ == "__main__":
    main()

