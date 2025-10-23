#!/usr/bin/env python3
"""
Real LLaVA-3D Teacher Integration for Distillation
=================================================

This module provides robust integration with the real LLaVA-3D teacher model
for knowledge distillation. It handles memory optimization, model loading,
and response generation for training the distilled student model.
"""

import os
import sys
import torch
import gc
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging

# Add LLaVA-3D to path
sys.path.append('/home/alasfour/scratch/llava-3d/LLaVA-3D')

try:
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from llava.conversation import conv_templates, SeparatorStyle
    from llava.utils import disable_torch_init
    from transformers import AutoTokenizer, AutoModelForCausalLM
    LLAVA_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ LLaVA-3D not available: {e}")
    LLAVA_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealTeacherIntegration:
    """
    Real LLaVA-3D Teacher Integration for Knowledge Distillation
    
    This class provides a robust interface to the real LLaVA-3D teacher model
    with memory optimization and efficient response generation.
    """
    
    def __init__(self, 
                 model_path: str = "/home/alasfour/scratch/llava-3d/LLaVA-3D",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 load_in_4bit: bool = True,
                 load_in_8bit: bool = False,
                 torch_dtype: torch.dtype = torch.float16):
        """
        Initialize the real teacher integration.
        
        Args:
            model_path: Path to LLaVA-3D model
            device: Device to run on
            load_in_4bit: Use 4-bit quantization
            load_in_8bit: Use 8-bit quantization
            torch_dtype: Data type for model weights
        """
        self.model_path = model_path
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.torch_dtype = torch_dtype
        
        self.teacher_model = None
        self.tokenizer = None
        self.image_processor = None
        self.conv_mode = None
        
        # Memory optimization
        self.max_memory = None
        self.device_map = "auto"
        
        logger.info(f"🚀 Initializing Real Teacher Integration")
        logger.info(f"   Model Path: {model_path}")
        logger.info(f"   Device: {device}")
        logger.info(f"   4-bit Quantization: {load_in_4bit}")
        logger.info(f"   8-bit Quantization: {load_in_8bit}")
    
    def load_teacher_model(self) -> bool:
        """
        Load the real LLaVA-3D teacher model with memory optimization.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not LLAVA_AVAILABLE:
            logger.error("❌ LLaVA-3D not available. Please check installation.")
            return False
        
        try:
            logger.info("🔄 Loading real LLaVA-3D teacher model...")
            
            # Disable torch init for memory efficiency
            disable_torch_init()
            
            # Get model name
            model_name = get_model_name_from_path(self.model_path)
            logger.info(f"   Model Name: {model_name}")
            
            # Configure quantization
            quantization_config = None
            if self.load_in_4bit:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=self.torch_dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            elif self.load_in_8bit:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            
            # Load model with memory optimization
            self.teacher_model, self.image_processor, self.tokenizer = load_pretrained_model(
                model_path=self.model_path,
                model_base=None,
                model_name=model_name,
                load_8bit=self.load_in_8bit,
                load_4bit=self.load_in_4bit,
                device_map=self.device_map,
                torch_dtype=self.torch_dtype,
                quantization_config=quantization_config
            )
            
            # Set conversation mode
            self.conv_mode = "llava_v1"
            
            logger.info("✅ Real LLaVA-3D teacher model loaded successfully!")
            logger.info(f"   Model: {type(self.teacher_model).__name__}")
            logger.info(f"   Device Map: {self.device_map}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load teacher model: {e}")
            return False
    
    def generate_teacher_response(self, 
                                image_path: Union[str, Path],
                                question: str,
                                max_new_tokens: int = 512,
                                temperature: float = 0.7,
                                top_p: float = 0.9) -> Dict[str, any]:
        """
        Generate teacher response for a given image and question.
        
        Args:
            image_path: Path to input image
            question: Question about the image
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            
        Returns:
            Dict containing response and metadata
        """
        if self.teacher_model is None:
            return {"error": "Teacher model not loaded"}
        
        try:
            # Load and process image
            if isinstance(image_path, str):
                image_path = Path(image_path)
            
            if not image_path.exists():
                return {"error": f"Image not found: {image_path}"}
            
            # Process image
            image = self.image_processor.process_images([str(image_path)], return_tensors='pt')
            if torch.cuda.is_available():
                image = {k: v.to(self.device) for k, v in image.items()}
            
            # Prepare conversation
            conv = conv_templates[self.conv_mode].copy()
            conv.append_message(conv.roles[0], f"{DEFAULT_IMAGE_TOKEN}\n{question}")
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            
            # Tokenize
            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
            if torch.cuda.is_available():
                input_ids = input_ids.to(self.device)
            
            # Generate response
            with torch.no_grad():
                output_ids = self.teacher_model.generate(
                    input_ids,
                    images=image['pixel_values'],
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                    use_cache=True
                )
            
            # Decode response
            response = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            
            # Extract only the assistant's response
            if conv.sep_style == SeparatorStyle.TWO:
                response = response.split(conv.sep2)[-1].strip()
            else:
                response = response.split(conv.roles[1] + ":")[-1].strip()
            
            return {
                "response": response,
                "question": question,
                "image_path": str(image_path),
                "model_name": "LLaVA-3D",
                "generation_params": {
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_p": top_p
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating teacher response: {e}")
            return {"error": str(e)}
    
    def generate_batch_responses(self, 
                               image_question_pairs: List[Tuple[str, str]],
                               batch_size: int = 4) -> List[Dict[str, any]]:
        """
        Generate responses for a batch of image-question pairs.
        
        Args:
            image_question_pairs: List of (image_path, question) tuples
            batch_size: Batch size for processing
            
        Returns:
            List of response dictionaries
        """
        responses = []
        
        for i in range(0, len(image_question_pairs), batch_size):
            batch = image_question_pairs[i:i + batch_size]
            logger.info(f"🔄 Processing batch {i//batch_size + 1}/{(len(image_question_pairs)-1)//batch_size + 1}")
            
            for image_path, question in batch:
                response = self.generate_teacher_response(image_path, question)
                responses.append(response)
                
                # Clear cache periodically
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
        
        return responses
    
    def get_teacher_features(self, image_path: Union[str, Path]) -> Dict[str, any]:
        """
        Extract teacher model features for distillation.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Dict containing extracted features
        """
        if self.teacher_model is None:
            return {"error": "Teacher model not loaded"}
        
        try:
            # Process image
            image = self.image_processor.process_images([str(image_path)], return_tensors='pt')
            if torch.cuda.is_available():
                image = {k: v.to(self.device) for k, v in image.items()}
            
            # Extract features from vision encoder
            with torch.no_grad():
                # Get vision features
                vision_features = self.teacher_model.get_model().get_vision_tower()(image['pixel_values'])
                
                # Get language model features
                if hasattr(self.teacher_model.get_model(), 'get_language_model'):
                    lang_features = self.teacher_model.get_model().get_language_model()
                else:
                    lang_features = None
                
                return {
                    "vision_features": vision_features.cpu(),
                    "language_features": lang_features,
                    "image_path": str(image_path),
                    "feature_shape": vision_features.shape
                }
                
        except Exception as e:
            logger.error(f"❌ Error extracting teacher features: {e}")
            return {"error": str(e)}
    
    def cleanup(self):
        """Clean up resources."""
        if self.teacher_model is not None:
            del self.teacher_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("🧹 Cleaned up teacher model resources")

def test_real_teacher_integration():
    """Test the real teacher integration."""
    logger.info("🧪 Testing Real Teacher Integration")
    
    # Initialize integration
    integration = RealTeacherIntegration()
    
    # Load teacher model
    if integration.load_teacher_model():
        logger.info("✅ Teacher model loaded successfully!")
        
        # Test with a sample image
        test_image = "/home/alasfour/scratch/distilled-llava3d/demo/scannet/posed_images/scene0356_00/00020.png"
        test_question = "What objects can you see in this 3D scene?"
        
        if os.path.exists(test_image):
            response = integration.generate_teacher_response(test_image, test_question)
            logger.info(f"📝 Teacher Response: {response.get('response', 'No response')}")
        else:
            logger.warning("⚠️ Test image not found, skipping response test")
        
        # Cleanup
        integration.cleanup()
    else:
        logger.error("❌ Failed to load teacher model")

if __name__ == "__main__":
    test_real_teacher_integration()

