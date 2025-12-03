#!/bin/bash
# Script to install VGGT for the distilled LLaVA-3D project

set -e

echo "=========================================="
echo "Installing VGGT for Distilled LLaVA-3D"
echo "=========================================="

# Check if VGGT is already cloned
VGGT_PATH="/home/alasfour/scratch/vggt"
if [ ! -d "$VGGT_PATH" ]; then
    echo "📦 Cloning VGGT repository..."
    cd /home/alasfour/scratch
    git clone https://github.com/facebookresearch/vggt.git
    echo "✅ VGGT cloned successfully"
else
    echo "✅ VGGT repository already exists at $VGGT_PATH"
fi

# Install dependencies (skip opencv-python since it's already available)
echo ""
echo "📦 Installing VGGT dependencies..."
cd "$VGGT_PATH"

# Install dependencies one by one (skip opencv-python)
pip install einops --quiet || echo "⚠️  einops installation had issues (may already be installed)"

echo ""
echo "✅ VGGT installation complete!"
echo ""
echo "📝 Next steps:"
echo "   1. VGGT is now available at: $VGGT_PATH"
echo "   2. The integration code will automatically detect it"
echo "   3. Run: python test_vggt_integration.py"
echo "   4. If VGGT loads successfully, you can proceed with training"
echo ""
echo "ℹ️  Note: VGGT will download pretrained weights from HuggingFace"
echo "   on first use (facebook/VGGT-1B)"


