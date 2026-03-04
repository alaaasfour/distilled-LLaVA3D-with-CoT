# Distilled LLaVA-3D: Efficient 3D Vision-Language Models with VGGT and Hidden CoT

Knowledge distillation framework for 3D vision-language models (VLMs): transfer 3D spatial understanding from a 7B-parameter teacher (LLaVA-3D) to a compact 2.29B-parameter student with **8.7× inference speedup** and **3× model compression** while retaining **54–72%** of teacher performance on specialized spatial reasoning. Features **VGGT** (Visual Geometry Grounded Transformer) as the vision encoder, **multi-task distillation** with uncertainty-based loss weighting, and **Hidden Chain-of-Thought (Hidden CoT)**—a latent scratchpad for improved reasoning without chain-of-thought data or interface changes.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

## 🎯 Overview

Large 3D VLMs like LLaVA-3D excel at spatial reasoning but are too heavy for edge and real-time use. This project:

1. **Distills** LLaVA-3D-7B into a 2.29B student with a custom transformer and **VGGT** vision encoder.
2. **Trains** with multi-task losses (text, depth, detection, spatial alignment) and **uncertainty-based** adaptive weighting.
3. **Adds Hidden CoT**: K learnable “thinking” tokens between vision and Q&A; trained and evaluated only on the final answer, with optional **diagnostic mode** to decode the scratchpad for research.

**Outcomes:** 8.7× faster inference, 3× smaller model, 54–72% of teacher performance on spatial tasks (68–72% on proximity/contact). Supports ScanNet and 3D-FRONT.

### Key Components

1. **Student Model**: Distilled LLaVA-3D with VGGT vision encoder (~3B parameters)
2. **Teacher Model**: Real LLaVA-3D-7B from HuggingFace (`ChaimZhu/LLaVA-3D-7B`)
3. **Vision Encoder**: VGGT-1B from HuggingFace (`facebook/VGGT-1B`)
4. **Auxiliary Teachers**: DPT-Large for depth estimation, YOLO for object detection

## ✨ Features

- **Efficient architecture**: 2.29B parameters (8.7 GB), 3× compression vs 7B teacher.
- **VGGT vision encoder**: Geometry-aware 3D features (e.g. camera/depth); ~1B params, full student stays 2.29B.
- **Multi-task distillation**: Text generation, depth (CE + regression + KL), object detection, spatial corresponding loss, multi-view and feature distillation.
- **Uncertainty-based loss weighting**: No manual loss weights; task weights adapt during training.
- **Hidden CoT**: Latent scratchpad (default K=8) for better reasoning; no CoT data or CoT teacher; deployment interface unchanged.
- **Diagnostic mode**: Optional decoding of thinking tokens to text (`scripts/diagnostic_cot.py`) for interpretability.
- **Reproducibility**: Cross-platform latency/RAM benchmark, K-ablation runner, training loss and comparison figure scripts.
- **Memory-friendly training**: CPU offloading for teacher (and optionally VGGT), 2-GPU split (vision vs transformer), gradient checkpointing option.

## 📦 Requirements

### System Requirements

- **GPU**: NVIDIA GPU with at least 10GB VRAM (20 GB+ recommended; 2× GPUs for CoT training).
- **CPU**: 8+ cores recommended
- **RAM**: 64GB+ recommended
- **OS**: Linux (tested on Ubuntu 20.04+)

### Software Requirements

- Python 3.10+
- CUDA 11.8+ or 12.1+
- PyTorch 2.0+
- SLURM (for cluster environments)

### Main dependencies

- `torch`, `torchvision`
- `transformers` (Hugging Face)
- `accelerate`, `einops`, `pillow`, `numpy`, `scipy`, `scikit-learn`, `tqdm`, `pyyaml`
- Optional: `ultralytics` (YOLO), `bitsandbytes` (8-bit optimizer)

## 🔧 Installation

### Step 1: Clone the Repository

```bash
cd ~/scratch
git clone <your-repo-url> distilled-llava3d
cd distilled-llava3d
```

### Step 2: Create Python Virtual Environment

```bash
# Create virtual environment
python3.11 -m venv distilled-llava3d-env

# Activate environment
source distilled-llava3d-env/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### Step 3: Install Core Dependencies

```bash
# Install PyTorch (adjust CUDA version as needed)
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core ML libraries
pip install transformers>=4.35.0
pip install accelerate>=0.24.0
pip install einops
pip install pillow
pip install opencv-python
pip install numpy
pip install scipy
pip install scikit-learn
pip install tqdm
pip install pyyaml
pip install wandb  # Optional: for experiment tracking
```

### Step 4: Install VGGT Vision Encoder

VGGT can be installed in two ways:

#### Option A: From HuggingFace (Recommended - Automatic)

The code will automatically download VGGT from HuggingFace when first used:

```bash
# No additional installation needed - transformers library handles it
# Model will be downloaded from: facebook/VGGT-1B
```

#### Option B: From GitHub (Manual Installation)

If you prefer to install from source:

```bash
# Clone VGGT repository
cd ~/scratch
git clone https://github.com/facebookresearch/vggt.git
cd vggt

# Install VGGT
pip install -e .

# Install additional dependencies
pip install einops

# Return to project directory
cd ~/scratch/distilled-llava3d
```

Or use the provided installation script:

```bash
bash install_vggt.sh
```

**Note**: The student model will automatically detect VGGT from either HuggingFace or the local installation.

### Step 5: Install LLaVA-3D Teacher Model

The teacher model is automatically downloaded from HuggingFace when first used. However, you need to set up the LLaVA-3D source code for proper integration:

```bash
# Clone LLaVA-3D repository (for utilities and model loading)
cd ~/scratch
git clone https://github.com/ChaimZhu/LLaVA-3D.git llava-3d
cd llava-3d/LLaVA-3D

# Install LLaVA-3D dependencies
pip install -e .

# Install additional requirements
pip install -r requirements.txt

# Return to project directory
cd ~/scratch/distilled-llava3d
```

**Note**: The teacher model (`ChaimZhu/LLaVA-3D-7B`) will be automatically downloaded from HuggingFace (~14GB) on first use. Ensure you have:
- HuggingFace account and authentication token (if model is gated)
- Sufficient disk space (~20GB for model files)

### Step 6: Install Additional Dependencies

```bash
# For depth estimation (DPT)
pip install transformers[vision]

# For object detection (YOLO)
pip install ultralytics

# For 3D data processing
pip install trimesh
pip install open3d  # Optional: for advanced 3D processing
```

### Step 7: Verify Installation

Test that all components are properly installed:

```bash
# Test VGGT integration
python test_vggt_integration.py

# Test teacher model loading (this will download the model on first run)
python -c "from real_llava3d_teacher import RealLLaVA3DTeacher; teacher = RealLLaVA3DTeacher(device='cpu'); print('✅ Teacher model loaded successfully')"
```

## 📊 Data Preparation

### Layout

Training expects a `data/` directory under the project root with one or more of:

```
data/
├── scannet/           # or scannet_real
│   └── <scene_id>/    # e.g. scene0000_00
│       ├── *.jpg      # and/or *.png, *.jpeg
│       └── images/    # optional subfolder
├── 3d_front/          # or 3d_front_real
│   └── <scene_id>/
│       └── *.jpg
```
- Each **scene** is a directory.
- **Images**: `.jpg`, `.png`, or `.jpeg` directly in the scene dir or in `images/`.
- No strict naming; the loader discovers scenes and samples up to 30 scenes per dataset (50 for `*_real`) and up to 10 images per scene.

### Datasets

- **ScanNet:** [ScanNet](https://github.com/ScanNet/ScanNet) - place extracted scenes under `data/scannet/`
- **3D-FRONT:** [3D-FRONT](https://huggingface.co/datasets/huanngzh/3D-Front) - extract to `data/3d_front/`

## 🚀 Training
### Hidden CoT training (recommended)

Train the student with Hidden CoT (K=8 by default; answer-only loss):

```bash
source distilled-llava3d-env/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Local
python train_cot.py --data_root data --checkpoint_dir checkpoints

# With options
python train_cot.py --data_root data --checkpoint_dir checkpoints \
  --num_thinking_tokens 8 --max_epochs 50
```

**Cluster (SLURM):**

```bash
# Edit scripts/training/run_cot_train.sbatch: set account, paths, and optionally data_root/checkpoint_dir via env or script.
sbatch scripts/training/run_cot_train.sbatch
```

Checkpoints: `checkpoints/cot_model_best.pt`, `checkpoints/cot_model_epoch_*.pt`. Results summary: `checkpoints/cot_training_results.json`.

### Ablation over number of thinking tokens (K)

```bash
# Train each K in {2,4,8,16} for 3 epochs (quick sweep)
python scripts/training/run_cot_ablation_k.py --k 2 4 8 16 --max_epochs 3 --base_dir checkpoints/cot_ablation_k

# Collect results only (after some runs)
python scripts/training/run_cot_ablation_k.py --collect_only --base_dir checkpoints/cot_ablation_k
```

### Other training scripts

- `fixed_training_pipeline.py`, `improved_training_pipeline.py`: Baseline (non-CoT) distillation.
- `scripts/training/run_train.sbatch`, `run_train_with_validation.sbatch`: SLURM for non-CoT runs.

Training depends on **real LLaVA-3D teacher** (`real_llava3d_teacher`). If that module is not available, CoT training will fail at import; ensure the teacher is set up as in your environment (e.g. LLaVA-3D repo or a custom wrapper).


## Evaluation and figures

### Diagnostic mode (decode thinking tokens)

```bash
python scripts/diagnostic_cot.py --checkpoint checkpoints/cot_model_best.pt \
  --image data/3d_front_real/bedroom_000/view_000.jpg \
  --question "Describe this 3D scene and identify objects."
```

### Cross-platform latency / RAM

```bash
python scripts/benchmark_cross_platform.py --checkpoint checkpoints/cot_model_best.pt \
  --warmup 3 --iters 20 --output results/cross_platform_results.csv
```

### Paper figures

- **Hidden CoT vs diagnostic comparison:**  
  `python scripts/generate_hidden_cot_comparison_figure.py --image data/3d_front_real/.../view_000.jpg --output results/figures/hidden_cot_comparison.png`
- **Training loss convergence:**  
  `python scripts/generate_training_loss_chart.py --output results/figures/training_loss_convergence.png`  
  (Edit script to point at your training log or use built-in example data.)

## 📁 Project Structure

```
distilled-llava3d/
├── README.md
│
├── train_cot.py                     # Hidden CoT training entry
├── real_llava3d_teacher.py          # LLaVA-3D teacher wrapper
├── real_depth_teacher.py            # Depth teacher (DPT)
├── object_detection_integration.py  # Detection (YOLO)
├── install_vggt.sh                  # VGGT install helper
│
├── scripts/
│   ├── distillation/
│   │   ├── student_model.py         # Student + VGGT + Hidden CoT
│   │   ├── uncertainty_loss.py     # Uncertainty-based loss
│   │   ├── dataset_loader.py
│   │   └── ...
│   ├── training/
│   │   ├── run_cot_train.sbatch     # SLURM CoT job
│   │   ├── run_cot_ablation_k.py     # K ablation runner
│   │   ├── run_train.sbatch
│   │   └── logs/
│   ├── evaluation/
│   │   └── spatial_benchmark_eval.py
│   ├── ablation/
│   ├── diagnostic_cot.py            # Decode thinking tokens
│   ├── benchmark_cross_platform.py  # Latency/RAM benchmark
│   ├── generate_hidden_cot_comparison_figure.py
│   └── generate_training_loss_chart.py
│
├── data/                            # Datasets (see Data Preparation)
├── checkpoints/                     # Saved models
├── results/
│   └── figures/                    # Generated figures
└── configs/
```

## 🔍 Key Implementation Details

### VGGT Vision Encoder Integration

**Location**: `scripts/distillation/student_model.py` - `VGGTVisionEncoder` class

**Key Features**:
- **Automatic Detection**: Tries HuggingFace first (`facebook/VGGT-1B`), then local installation
- **CPU Offloading**: VGGT runs on CPU by default to save GPU memory
- **Feature Extraction**: Uses VGGT's aggregator output (2048-dim features)
- **Image Preprocessing**: Automatically resizes images to 518x518 (VGGT's expected size)
- **Device Management**: Custom `_apply` method keeps VGGT on designated device even when parent model moves to GPU


### Memory Optimization

To fit training on smaller GPUs:

1. **VGGT on CPU**: Vision encoder runs on CPU (slower but memory-efficient)
2. **Teacher on CPU**: LLaVA-3D teacher runs on CPU with CPU offloading
3. **Batch Size = 1**: Reduced batch size to minimize memory usage
4. **Gradient Checkpointing**: Enabled for student model
5. **Periodic Cache Clearing**: `torch.cuda.empty_cache()` called every 5 batches

### Teacher Model Loading

The real LLaVA-3D teacher:
- Downloads from HuggingFace: `ChaimZhu/LLaVA-3D-7B`
- Loads on CPU to avoid OOM errors
- Uses `device_map={"": device}` for compatibility
- Changes working directory to LLaVA-3D root for proper initialization

## 🐛 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory (OOM)

**Symptoms**: `RuntimeError: CUDA out of memory`

**Solutions**:
- Reduce batch size to 1
- Ensure VGGT is on CPU: `vggt_device = 'cpu'`
- Ensure teacher is on CPU: `device="cpu"` in `RealLLaVA3DTeacher`
- Add more `torch.cuda.empty_cache()` calls
- Use gradient accumulation instead of larger batches

#### 2. VGGT Not Found

**Symptoms**: `ModuleNotFoundError: No module named 'vggt'`

**Solutions**:
- Install from HuggingFace: `pip install transformers`
- Or install from GitHub: `bash install_vggt.sh`
- Check that VGGT path is in `sys.path`

#### 3. Teacher Model Download Fails

**Symptoms**: `OSError: Can't load tokenizer` or download errors

**Solutions**:
- Check HuggingFace authentication: `huggingface-cli login`
- Ensure sufficient disk space (~20GB)
- Check internet connection
- Try downloading manually: `huggingface-cli download ChaimZhu/LLaVA-3D-7B`

#### 4. Image Size Errors

**Symptoms**: `AssertionError: Input image height X is not a multiple of patch height 14`

**Solutions**:
- Images are automatically resized to 518x518 in the code
- Check that `F.interpolate` is being called in `_extract_vggt_features`

#### 5. Dtype Mismatch Errors

**Symptoms**: `RuntimeError: Found dtype Double but expected Float`

**Solutions**:
- Ensure all tensors are `float32`: `.float()` or `dtype=torch.float32`
- Check loss computation uses correct dtypes
- Ensure cross-entropy targets are `long`: `.long()` or `dtype=torch.long`

#### 6. Slow Training

**Symptoms**: Training is very slow

**Solutions**:
- VGGT on CPU is slower but more memory-efficient
- Teacher on CPU is slower but necessary for memory constraints
- Consider using larger GPU if available
- Reduce number of training samples for faster iteration


## 📚 Citation

```bibtex
@misc{distilled-llava3d,
  title={Distilling 3D Spatial Reasoning into a Lightweight Vision-Language Model with CoT},
  author={Alaa Asfour},
  year={2026},
  url={https://github.com/alaaasfour/distilled-LLaVA3D-with-CoT}
}
```

### References

- **LLaVA-3D**: [ChaimZhu/LLaVA-3D](https://github.com/ChaimZhu/LLaVA-3D)
- **VGGT**: [VGGT Project Page](https://vgg-t.github.io/)
- **DPT**: [Intel DPT](https://github.com/isl-org/DPT)
- **YOLO**: [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- **ScanNet, 3D-FRONT:** See Data Preparation links above.

## 📝 License

## 🤝 Contributing

[Contributing guidelines]

## 👤 Author

Alaa Asfour (alaa.asfour@torontomu.ca)

## 🙏 Acknowledgments

- LLaVA-3D team for the teacher model
- VGGT team for the vision encoder
- All open-source contributors
- Digital Research Alliance of Canada
- Contributors to ScanNet, 3D-FRONT

---