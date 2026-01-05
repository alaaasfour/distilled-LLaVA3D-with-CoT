#!/usr/bin/env python3
"""
Visualization Script for Attention Maps and Depth Predictions
Generates visualizations showing:
1. Attention maps from the vision encoder
2. Depth predictions
3. Overlay of attention on depth maps
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image
import torchvision.transforms as transforms
import logging
import json

from scripts.distillation.student_model import DistilledLLaVA3D, DistilledLLaVA3DConfig, MockVisionEncoder
from real_depth_teacher import RealDepthTeacher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AttentionDepthVisualizer:
    """
    Visualizes attention maps and depth predictions from the student model.
    """
    
    def __init__(self,
                 student_checkpoint: str,
                 device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # Load student model
        logger.info(f"Loading student model from {student_checkpoint}...")
        self.student_model = self._load_student_model(student_checkpoint)
        self.student_model.eval()
        
        # Load depth teacher for comparison
        try:
            self.depth_teacher = RealDepthTeacher(device=self.device)
            logger.info("✅ Depth teacher loaded for comparison")
        except Exception as e:
            logger.warning(f"⚠️  Could not load depth teacher: {e}")
            self.depth_teacher = None
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    
    def _load_student_model(self, checkpoint_path: str) -> DistilledLLaVA3D:
        """Load student model with architecture compatibility."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        has_old_arch = 'vision_encoder.conv_layers.0.weight' in state_dict
        has_new_arch = 'vision_encoder.vggt_model' in state_dict
        
        config = DistilledLLaVA3DConfig()
        
        if has_old_arch and not has_new_arch:
            logger.info("⚠️  Detected old architecture (MockVisionEncoder)")
            model = DistilledLLaVA3D(config)
            model.vision_encoder = MockVisionEncoder(config)
            model.load_state_dict(state_dict, strict=False)
        else:
            logger.info("✅ Detected new architecture (VGGTVisionEncoder)")
            model = DistilledLLaVA3D(config)
            model.load_state_dict(state_dict, strict=False)
        
        model.to(self.device)
        return model
    
    def extract_attention_maps(self, image_tensor: torch.Tensor) -> Optional[np.ndarray]:
        """
        Extract attention maps from the vision encoder.
        Returns attention map as numpy array.
        """
        try:
            with torch.no_grad():
                # Forward pass through vision encoder
                outputs = self.student_model.vision_encoder(image_tensor)
                
                # Try to extract attention weights
                # This depends on the architecture - VGGT may have attention
                if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                    # Use last layer attention
                    attention = outputs.attentions[-1]  # Shape: (batch, heads, seq_len, seq_len)
                    # Average over heads
                    attention = attention.mean(dim=1)  # (batch, seq_len, seq_len)
                    # Get attention to CLS token or average
                    attention_map = attention[0, 0, 1:].cpu().numpy()  # Skip CLS token
                    return attention_map
                elif hasattr(self.student_model.vision_encoder, 'vggt_model'):
                    # For VGGT, try to get feature maps
                    # This is a simplified version - actual implementation depends on VGGT architecture
                    features = outputs.last_hidden_state  # (batch, seq_len, dim)
                    # Compute attention as feature magnitude
                    attention_map = features[0].norm(dim=-1).cpu().numpy()
                    return attention_map
                else:
                    # Fallback: use feature magnitude as attention proxy
                    features = outputs.last_hidden_state
                    attention_map = features[0].norm(dim=-1).cpu().numpy()
                    return attention_map
        except Exception as e:
            logger.warning(f"Could not extract attention maps: {e}")
            return None
    
    def extract_depth_prediction(self, image_tensor: torch.Tensor) -> Optional[np.ndarray]:
        """
        Extract depth prediction from the student model.
        Returns depth map as numpy array.
        """
        try:
            with torch.no_grad():
                # Forward pass
                outputs = self.student_model.vision_encoder(image_tensor)
                vision_features = outputs.last_hidden_state.squeeze(1)
                
                # Get depth logits
                depth_logits = self.student_model.depth_head(vision_features)
                depth_probs = F.softmax(depth_logits, dim=-1)
                
                # Convert to depth values (using bin centers)
                bin_centers = np.array([0.2, 0.5, 0.8])
                depth_value = (depth_probs[0].cpu().numpy() * bin_centers).sum()
                
                # For visualization, create a simple depth map
                # In practice, you'd want to reshape this to image dimensions
                # For now, return the depth value and probabilities
                return {
                    'depth_value': depth_value,
                    'depth_probs': depth_probs[0].cpu().numpy(),
                    'depth_logits': depth_logits[0].cpu().numpy()
                }
        except Exception as e:
            logger.warning(f"Could not extract depth prediction: {e}")
            return None
    
    def visualize_sample(self, 
                        image_path: str,
                        output_dir: str = "visualizations",
                        question: str = "Describe this 3D scene.") -> Dict:
        """
        Visualize attention and depth for a single sample.
        
        Returns:
            Dictionary with visualization paths and metadata
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device).float()
        image_np = np.array(image.resize((224, 224)))
        
        # Extract attention and depth
        attention_map = self.extract_attention_maps(image_tensor)
        depth_pred = self.extract_depth_prediction(image_tensor)
        
        # Get student response
        with torch.no_grad():
            student_response = self.student_model.generate_response(question, image_tensor)
            if isinstance(student_response, dict):
                student_text = student_response.get('response', str(student_response))
            else:
                student_text = str(student_response)
        
        # Get ground truth depth (if available)
        gt_depth = None
        if self.depth_teacher is not None:
            try:
                depth_continuous, _ = self.depth_teacher.get_depth_labels(image_np, num_bins=3)
                gt_depth = depth_continuous
            except:
                pass
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        # Original image
        axes[0, 0].imshow(image_np)
        axes[0, 0].set_title('Original Image', fontsize=12)
        axes[0, 0].axis('off')
        
        # Attention map overlay
        if attention_map is not None:
            # Reshape attention to image dimensions (simplified)
            # In practice, you'd need to properly map attention tokens to spatial locations
            attention_2d = attention_map[:image_np.shape[0] * image_np.shape[1]].reshape(
                image_np.shape[0], image_np.shape[1]
            ) if len(attention_map) >= image_np.shape[0] * image_np.shape[1] else \
            np.ones((image_np.shape[0], image_np.shape[1])) * attention_map.mean()
            
            axes[0, 1].imshow(image_np)
            attention_overlay = axes[0, 1].imshow(attention_2d, alpha=0.5, cmap='hot')
            axes[0, 1].set_title('Attention Map Overlay', fontsize=12)
            axes[0, 1].axis('off')
            plt.colorbar(attention_overlay, ax=axes[0, 1])
        else:
            axes[0, 1].text(0.5, 0.5, 'Attention map\nnot available', 
                          ha='center', va='center', fontsize=12)
            axes[0, 1].axis('off')
        
        # Depth prediction
        if depth_pred is not None:
            depth_value = depth_pred['depth_value']
            depth_probs = depth_pred['depth_probs']
            
            # Create depth visualization
            depth_vis = np.ones((image_np.shape[0], image_np.shape[1])) * depth_value
            axes[1, 0].imshow(depth_vis, cmap='viridis')
            axes[1, 0].set_title(f'Predicted Depth (value: {depth_value:.3f})', fontsize=12)
            axes[1, 0].axis('off')
            
            # Depth probability distribution
            axes[1, 1].bar(['Near', 'Mid', 'Far'], depth_probs)
            axes[1, 1].set_title('Depth Probability Distribution', fontsize=12)
            axes[1, 1].set_ylabel('Probability')
            axes[1, 1].set_ylim([0, 1])
        else:
            axes[1, 0].text(0.5, 0.5, 'Depth prediction\nnot available', 
                          ha='center', va='center', fontsize=12)
            axes[1, 0].axis('off')
            axes[1, 1].text(0.5, 0.5, 'Depth probabilities\nnot available', 
                          ha='center', va='center', fontsize=12)
            axes[1, 1].axis('off')
        
        # Add student response as text
        fig.suptitle(f'Student Response: {student_text[:100]}...', fontsize=10, y=0.02)
        
        # Save visualization
        image_name = Path(image_path).stem
        viz_path = output_path / f"{image_name}_attention_depth.png"
        plt.tight_layout()
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Visualization saved to {viz_path}")
        
        return {
            'image_path': image_path,
            'visualization_path': str(viz_path),
            'student_response': student_text,
            'attention_available': attention_map is not None,
            'depth_available': depth_pred is not None,
            'depth_value': depth_pred['depth_value'] if depth_pred else None
        }
    
    def visualize_batch(self, 
                       image_paths: List[str],
                       output_dir: str = "visualizations",
                       question: str = "Describe this 3D scene.") -> List[Dict]:
        """
        Visualize multiple samples.
        
        Returns:
            List of visualization results
        """
        results = []
        for i, image_path in enumerate(image_paths):
            logger.info(f"Visualizing {i+1}/{len(image_paths)}: {Path(image_path).name}")
            try:
                result = self.visualize_sample(image_path, output_dir, question)
                results.append(result)
            except Exception as e:
                logger.warning(f"Error visualizing {image_path}: {e}")
                continue
        
        # Save summary
        summary_path = Path(output_dir) / "visualization_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"✅ Visualized {len(results)}/{len(image_paths)} samples")
        logger.info(f"💾 Summary saved to {summary_path}")
        
        return results


def main():
    """Main visualization function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize Attention Maps and Depth Predictions')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to student model checkpoint')
    parser.add_argument('--image_path', type=str, default=None,
                       help='Path to single image (or use --data_root for batch)')
    parser.add_argument('--data_root', type=str, default='/home/alasfour/scratch/distilled-llava3d/data',
                       help='Root directory for images (for batch visualization)')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of samples to visualize (if using data_root)')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--question', type=str, default='Describe this 3D scene and identify objects.',
                       help='Question to ask the model')
    
    args = parser.parse_args()
    
    # Initialize visualizer
    visualizer = AttentionDepthVisualizer(
        student_checkpoint=args.checkpoint,
        device='cuda'
    )
    
    # Get image paths
    if args.image_path:
        image_paths = [args.image_path]
    else:
        # Load from data directory
        image_paths = []
        data_root = Path(args.data_root)
        for dataset_dir in ['scannet', '3d_front', 'matterport3d']:
            dataset_path = data_root / dataset_dir
            if not dataset_path.exists():
                continue
            for scene_dir in dataset_path.iterdir():
                if not scene_dir.is_dir():
                    continue
                images_dir = scene_dir / 'images'
                if images_dir.exists():
                    image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
                else:
                    image_files = list(scene_dir.glob('*.jpg')) + list(scene_dir.glob('*.png'))
                for img_file in image_files[:2]:
                    if len(image_paths) >= args.num_samples:
                        break
                    image_paths.append(str(img_file))
                if len(image_paths) >= args.num_samples:
                    break
            if len(image_paths) >= args.num_samples:
                break
    
    logger.info(f"✅ Found {len(image_paths)} images to visualize")
    
    # Visualize
    results = visualizer.visualize_batch(
        image_paths,
        output_dir=args.output_dir,
        question=args.question
    )
    
    logger.info(f"✅ Visualization complete! Generated {len(results)} visualizations")


if __name__ == "__main__":
    main()




