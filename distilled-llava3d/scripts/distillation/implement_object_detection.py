#!/usr/bin/env python3
"""Implementation of object detection for the student model."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class LightweightObjectDetector(nn.Module):
    """Lightweight object detector for the student model."""
    
    def __init__(self, num_classes=20):
        super().__init__()
        self.num_classes = num_classes
        
        # Lightweight backbone (MobileNet-style)
        self.backbone = nn.Sequential(
            # Initial conv
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # MobileNet blocks
            self._make_mobile_block(32, 64, 1),
            self._make_mobile_block(64, 128, 2),
            self._make_mobile_block(128, 256, 2),
            self._make_mobile_block(256, 512, 2),
            
            # Global average pooling
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Detection head
        self.detection_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )
        
        # Object classes relevant to our use case
        self.class_names = [
            'person', 'car', 'truck', 'bus', 'motorcycle', 'bicycle',
            'building', 'house', 'tree', 'grass', 'water', 'sky',
            'road', 'sidewalk', 'fence', 'sign', 'bench', 'pole',
            'boat', 'airplane'
        ]
    
    def _make_mobile_block(self, in_channels, out_channels, stride):
        """Create a MobileNet-style block."""
        return nn.Sequential(
            # Depthwise conv
            nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            
            # Pointwise conv
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        logits = self.detection_head(features)
        return logits
    
    def detect_objects(self, image_tensor, confidence_threshold=0.3):
        """Detect objects in the image."""
        with torch.no_grad():
            logits = self.forward(image_tensor)
            probabilities = torch.softmax(logits, dim=1)
            
            # Get top predictions
            top_probs, top_indices = torch.topk(probabilities, k=5, dim=1)
            
            detected_objects = []
            for i in range(top_probs.size(1)):
                if top_probs[0, i] > confidence_threshold:
                    class_idx = top_indices[0, i].item()
                    confidence = top_probs[0, i].item()
                    class_name = self.class_names[class_idx]
                    detected_objects.append({
                        'class': class_name,
                        'confidence': confidence,
                        'class_id': class_idx
                    })
            
            return detected_objects

class EnhancedStudentModel:
    """Enhanced student model with object detection."""
    
    def __init__(self):
        # Import the existing model
        from scripts.distillation.student_model import DistilledLLaVA3D, DistilledLLaVA3DConfig
        
        self.config = DistilledLLaVA3DConfig()
        self.base_model = DistilledLLaVA3D(self.config)
        self.object_detector = LightweightObjectDetector()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    
    def analyze_image_with_objects(self, pixel_values):
        """Analyze image with both statistical features and object detection."""
        # Get base analysis
        base_features = self.base_model.analyze_image_content(pixel_values)
        
        # Get object detection
        detected_objects = self.object_detector.detect_objects(pixel_values)
        
        # Enhanced features
        enhanced_features = base_features.copy()
        
        # Add object-based features
        enhanced_features['detected_objects'] = detected_objects
        enhanced_features['num_objects'] = len(detected_objects)
        enhanced_features['object_classes'] = [obj['class'] for obj in detected_objects]
        enhanced_features['object_confidences'] = [obj['confidence'] for obj in detected_objects]
        
        # Specific object detection flags
        enhanced_features['has_person_detected'] = any(obj['class'] == 'person' for obj in detected_objects)
        enhanced_features['has_vehicle_detected'] = any(obj['class'] in ['car', 'truck', 'bus', 'motorcycle'] for obj in detected_objects)
        enhanced_features['has_building_detected'] = any(obj['class'] in ['building', 'house'] for obj in detected_objects)
        enhanced_features['has_nature_detected'] = any(obj['class'] in ['tree', 'grass', 'water'] for obj in detected_objects)
        enhanced_features['has_sky_detected'] = any(obj['class'] == 'sky' for obj in detected_objects)
        
        return enhanced_features
    
    def generate_enhanced_response(self, question, pixel_values):
        """Generate response using enhanced features."""
        # Get enhanced analysis
        features = self.analyze_image_with_objects(pixel_values)
        
        # Generate base response
        base_response = self.base_model.generate_response(question, pixel_values)
        
        # Enhance with object information
        question_lower = question.lower()
        
        if "what objects" in question_lower or "what do you see" in question_lower:
            if features['detected_objects']:
                object_list = [f"{obj['class']} ({obj['confidence']:.2f})" for obj in features['detected_objects']]
                enhanced_response = f"{base_response} Specifically, I can detect: {', '.join(object_list)}."
            else:
                enhanced_response = f"{base_response} No specific objects were detected with high confidence."
        else:
            enhanced_response = base_response
        
        return enhanced_response, features

def test_enhanced_model():
    """Test the enhanced model with object detection."""
    print("🧪 Testing Enhanced Model with Object Detection")
    print("=" * 60)
    
    # Initialize enhanced model
    enhanced_model = EnhancedStudentModel()
    enhanced_model.base_model.eval()
    enhanced_model.object_detector.eval()
    
    # Test images
    test_images = [
        "/scratch/alasfour/llava-3d/LLaVA-3D/demo/my_images/IMG_001.png",
        "/scratch/alasfour/llava-3d/LLaVA-3D/demo/my_images/LLaVA3D-view.jpg"
    ]
    
    questions = [
        "What objects can you see in this image?",
        "What are the things I should be cautious about when I visit here?"
    ]
    
    for image_path in test_images:
        print(f"\n🖼️  Testing: {image_path}")
        print("-" * 40)
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        pixel_values = enhanced_model.transform(image).unsqueeze(0)
        
        for question in questions:
            print(f"\n❓ Question: {question}")
            
            # Get enhanced response
            response, features = enhanced_model.generate_enhanced_response(question, pixel_values)
            
            print(f"🤖 Response: {response}")
            print(f"🔍 Detected Objects: {features['detected_objects']}")
            print(f"📊 Object Count: {features['num_objects']}")
            print(f"🎯 Specific Flags:")
            print(f"   - Person: {features['has_person_detected']}")
            print(f"   - Vehicle: {features['has_vehicle_detected']}")
            print(f"   - Building: {features['has_building_detected']}")
            print(f"   - Nature: {features['has_nature_detected']}")
            print(f"   - Sky: {features['has_sky_detected']}")

if __name__ == "__main__":
    test_enhanced_model()
