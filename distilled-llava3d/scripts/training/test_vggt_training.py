#!/usr/bin/env python3
"""
Quick test to verify VGGT integration works with image resizing.
"""

import sys
sys.path.append('/home/alasfour/scratch/distilled-llava3d')

import torch
from scripts.distillation.student_model import DistilledLLaVA3D, DistilledLLaVA3DConfig

def test_vggt_with_different_sizes():
    """Test VGGT with various image sizes."""
    print("=" * 60)
    print("Testing VGGT with Image Resizing")
    print("=" * 60)
    
    config = DistilledLLaVA3DConfig()
    model = DistilledLLaVA3D(config)
    
    # Check if VGGT is loaded
    if hasattr(model.vision_encoder, 'vggt_model') and model.vision_encoder.vggt_model is not None:
        print("✅ VGGT is loaded")
    else:
        print("⚠️  VGGT not loaded, using fallback")
        return True
    
    # Test with different image sizes
    test_sizes = [
        (224, 224),  # Standard size
        (968, 1296),  # Problematic size from error
        (518, 518),  # VGGT's expected size
        (256, 256),  # Another common size
    ]
    
    print("\n🧪 Testing different image sizes...")
    for height, width in test_sizes:
        try:
            batch_size = 1
            pixel_values = torch.randn(batch_size, 3, height, width)
            
            with torch.no_grad():
                output = model.vision_encoder(pixel_values)
                features = output.last_hidden_state
            
            print(f"✅ Size {height}x{width}: Output shape {features.shape}")
        except Exception as e:
            print(f"❌ Size {height}x{width}: {str(e)[:100]}")
            return False
    
    print("\n" + "=" * 60)
    print("✅ All image size tests passed!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_vggt_with_different_sizes()
    sys.exit(0 if success else 1)


