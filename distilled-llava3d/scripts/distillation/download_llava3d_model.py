#!/usr/bin/env python3
"""
Download LLaVA-3D Pre-trained Model
===================================

This script downloads the actual LLaVA-3D pre-trained model from Hugging Face
for use in real teacher distillation.
"""

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from huggingface_hub import hf_hub_download, snapshot_download
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_llava3d_model(model_name: str = "llava-hf/LLaVA-3D-7B", 
                           save_dir: str = "/home/alasfour/scratch/llava-3d/models"):
    """
    Download LLaVA-3D model from Hugging Face.
    
    Args:
        model_name: Hugging Face model name
        save_dir: Directory to save the model
    """
    logger.info(f"📥 Downloading LLaVA-3D model: {model_name}")
    logger.info(f"   Save Directory: {save_dir}")
    
    try:
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Download model using snapshot_download
        logger.info("🔄 Downloading model files...")
        model_path = snapshot_download(
            repo_id=model_name,
            local_dir=save_dir,
            local_dir_use_symlinks=False
        )
        
        logger.info(f"✅ Model downloaded successfully!")
        logger.info(f"   Model Path: {model_path}")
        
        # Test loading the model
        logger.info("🧪 Testing model loading...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            logger.info(f"✅ Tokenizer loaded: {type(tokenizer).__name__}")
            
            # Try loading model with memory optimization
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                load_in_4bit=True
            )
            logger.info(f"✅ Model loaded: {type(model).__name__}")
            
            return model_path
            
        except Exception as e:
            logger.warning(f"⚠️ Model loading test failed: {e}")
            logger.info("   Model files downloaded but loading test failed")
            return model_path
            
    except Exception as e:
        logger.error(f"❌ Failed to download model: {e}")
        return None

def test_model_download():
    """Test downloading the LLaVA-3D model."""
    logger.info("🧪 Testing LLaVA-3D model download")
    
    # Try different model names
    model_names = [
        "llava-hf/LLaVA-3D-7B",
        "llava-hf/LLaVA-3D-13B", 
        "llava-hf/LLaVA-3D",
        "llava-hf/llava-3d-7b",
        "llava-hf/llava-3d-13b"
    ]
    
    for model_name in model_names:
        logger.info(f"🔄 Trying model: {model_name}")
        model_path = download_llava3d_model(model_name)
        
        if model_path:
            logger.info(f"✅ Successfully downloaded: {model_name}")
            return model_path
        else:
            logger.warning(f"⚠️ Failed to download: {model_name}")
    
    logger.error("❌ All model downloads failed")
    return None

if __name__ == "__main__":
    # Test downloading
    model_path = test_model_download()
    
    if model_path:
        logger.info(f"🎉 LLaVA-3D model ready at: {model_path}")
    else:
        logger.error("❌ Failed to download LLaVA-3D model")

