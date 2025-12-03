# Distilled LLaVA-3D: Knowledge Distillation for Efficient 3D Vision-Language Understanding

A lightweight student model trained via knowledge distillation from the large LLaVA-3D teacher model, featuring VGGT (Visual Geometry Grounded Transformer) as the vision encoder for state-of-the-art 3D scene understanding.

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

This project implements knowledge distillation to create a smaller, more efficient student model that learns from the large LLaVA-3D-7B teacher model. The student model uses:

- **VGGT (Visual Geometry Grounded Transformer)** as the vision encoder - a state-of-the-art feed-forward network for 3D scene understanding (CVPR 2025 Best Paper Award)
- **Real LLaVA-3D-7B** as the teacher model for supervision
- **Multi-task learning** with depth estimation, object detection, and spatial reasoning
- **CPU offloading** for memory-efficient training

### Key Components

1. **Student Model**: Distilled LLaVA-3D with VGGT vision encoder (~3B parameters)
2. **Teacher Model**: Real LLaVA-3D-7B from HuggingFace (`ChaimZhu/LLaVA-3D-7B`)
3. **Vision Encoder**: VGGT-1B from HuggingFace (`facebook/VGGT-1B`)
4. **Auxiliary Teachers**: DPT-Large for depth estimation, YOLO for object detection

## ✨ Features

- **Efficient Architecture**: Reduced from 7B to ~3B parameters while maintaining performance
- **VGGT Integration**: State-of-the-art 3D vision encoder with pre-trained features
- **Multi-task Learning**: Simultaneous learning of text generation, depth estimation, object detection, and spatial reasoning
- **Memory Efficient**: CPU offloading for teacher model and VGGT to fit training on smaller GPUs
- **Real Teacher Supervision**: Uses actual LLaVA-3D-7B model (not mock) for high-quality supervision
- **Comprehensive Training**: Supports multiple 3D datasets (ScanNet, 3D-FRONT, Matterport3D)

## 📦 Requirements

### System Requirements

- **GPU**: NVIDIA GPU with at least 10GB VRAM (tested on H100 40GB MIG slice)
- **CPU**: 8+ cores recommended
- **RAM**: 64GB+ recommended
- **OS**: Linux (tested on Ubuntu 20.04+)

### Software Requirements

- Python 3.11+
- CUDA 11.8+ or 12.1+
- PyTorch 2.0+
- SLURM (for cluster environments)

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

## 🚀 Training

### Quick Start

#### Local Training (Single GPU)

```bash
# Activate environment
source distilled-llava3d-env/bin/activate

# Load CUDA module (if on cluster)
module load cuda

# Run training
python fixed_training_pipeline.py
```

### Training Configuration

Key training parameters in `fixed_training_pipeline.py`:

```python
# Training settings
epochs = 50
batch_size = 1  # Reduced for memory efficiency
learning_rate = 1e-4
device = "cuda"

# VGGT device (CPU offloading for memory efficiency)
vggt_device = 'cpu'  # or 'cuda' if GPU memory allows

# Loss weights
lambda_det = 0.35          # Object detection
lambda_depth_ce = 0.25     # Depth classification
lambda_depth_reg = 0.15    # Depth regression
lambda_depth_kl = 0.0125   # Depth KL divergence
lambda_spatial = 0.25      # Spatial reasoning
lambda_mv = 0.1           # Multi-view consistency
lambda_feat = 0.3         # Feature distillation
```

### Training Process

The training pipeline:

1. **Initialization**:
   - Loads student model with VGGT vision encoder
   - Loads real LLaVA-3D teacher model (on CPU for memory efficiency)
   - Loads depth teacher (DPT-Large) and object detection (YOLO)
   - Prepares training datasets

2. **Training Loop**:
   - For each batch:
     - Extract visual features using VGGT
     - Generate teacher responses (on CPU)
     - Compute multi-task losses (text, depth, detection, spatial)
     - Backpropagate and update student model

3. **Checkpointing**:
   - Saves checkpoints to `checkpoints/` directory
   - Logs training statistics

## 📁 Project Structure

```
distilled-llava3d/
├── README.md                          # This file
├── fixed_training_pipeline.py         # Main training script
├── real_llava3d_teacher.py            # Teacher model wrapper
├── real_depth_teacher.py              # Depth teacher (DPT)
├── object_detection_integration.py    # Object detection (YOLO)
├── spatial_reasoning_augmentation.py  # Spatial reasoning
├── install_vggt.sh                    # VGGT installation script
│
├── scripts/
│   ├── distillation/
│   │   ├── student_model.py           # Student model with VGGT
│   │   ├── distillation_loss.py       # Multi-task loss functions
│   │   └── dataset_loader.py          # Data loading utilities
│   │
│   ├── training/
│   │   └── run_train.sbatch          # SLURM training script
│   │
│   └── evaluation/
│       └── evaluate_3d_tasks.py      # Evaluation scripts
│
├── data/                              # Training datasets
├── checkpoints/                        # Model checkpoints
├── logs/                              # Training logs
└── configs/                           # Configuration files
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
  title={Distilled LLaVA-3D: Knowledge Distillation for Efficient 3D Vision-Language Understanding},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/distilled-llava3d}
}
```

### References

- **LLaVA-3D**: [ChaimZhu/LLaVA-3D](https://github.com/ChaimZhu/LLaVA-3D)
- **VGGT**: [VGGT Project Page](https://vgg-t.github.io/)
- **DPT**: [Intel DPT](https://github.com/isl-org/DPT)
- **YOLO**: [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)

## 📝 License

## 🤝 Contributing

[Contributing guidelines]

## 👤 Author

[Your name and contact information]

## 🙏 Acknowledgments

- LLaVA-3D team for the teacher model
- VGGT team for the vision encoder
- All open-source contributors
- Digital Research Alliance of Canada

---