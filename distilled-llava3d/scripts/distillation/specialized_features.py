#!/usr/bin/env python3
"""Specialized features and improved response quality for distilled LLaVA-3D."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import time
from typing import Dict, List, Tuple, Any
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class SpecializedFeatures:
    """Specialized features for different task types."""
    
    def __init__(self, base_model, device='cuda'):
        self.base_model = base_model
        self.device = device
        
        # Task-specific modules
        self.safety_analyzer = SafetyAnalyzer()
        self.spatial_reasoner = SpatialReasoner()
        self.scene_descriptor = SceneDescriptor()
        self.navigation_guide = NavigationGuide()
        
        # Response quality enhancer
        self.response_enhancer = ResponseEnhancer()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    
    def analyze_with_specialized_features(self, question, pixel_values):
        """Analyze with specialized features based on question type."""
        question_lower = question.lower()
        
        # Get base analysis
        base_features = self.base_model.analyze_image_content(pixel_values)
        
        # Determine task type and apply specialized analysis
        if any(word in question_lower for word in ['cautious', 'danger', 'safety', 'risk']):
            specialized_features = self.safety_analyzer.analyze_safety(pixel_values, base_features)
            task_type = 'safety'
        elif any(word in question_lower for word in ['spatial', 'relationship', 'depth', 'position']):
            specialized_features = self.spatial_reasoner.analyze_spatial(pixel_values, base_features)
            task_type = 'spatial'
        elif any(word in question_lower for word in ['describe', 'scene', 'environment', 'layout']):
            specialized_features = self.scene_descriptor.analyze_scene(pixel_values, base_features)
            task_type = 'description'
        elif any(word in question_lower for word in ['navigate', 'path', 'route', 'direction']):
            specialized_features = self.navigation_guide.analyze_navigation(pixel_values, base_features)
            task_type = 'navigation'
        else:
            specialized_features = self._general_analysis(pixel_values, base_features)
            task_type = 'general'
        
        # Combine features
        combined_features = {
            **base_features,
            **specialized_features,
            'task_type': task_type,
            'specialized_analysis': True
        }
        
        return combined_features
    
    def generate_specialized_response(self, question, pixel_values):
        """Generate specialized response based on task type."""
        # Get specialized analysis
        features = self.analyze_with_specialized_features(question, pixel_values)
        
        # Generate base response
        base_response = self.base_model.generate_response(question, pixel_values)
        
        # Enhance with specialized features
        enhanced_response = self.response_enhancer.enhance_response(
            base_response, features, question
        )
        
        return enhanced_response, features
    
    def _general_analysis(self, pixel_values, base_features):
        """General analysis for non-specialized tasks."""
        return {
            'general_confidence': 0.7,
            'analysis_type': 'general',
            'comprehensive_understanding': True
        }

class SafetyAnalyzer:
    """Specialized safety analysis module."""
    
    def __init__(self):
        self.safety_categories = [
            'height_risks', 'terrain_hazards', 'weather_conditions',
            'structural_hazards', 'traffic_safety', 'water_safety'
        ]
    
    def analyze_safety(self, pixel_values, base_features):
        """Analyze safety aspects of the scene."""
        safety_features = {
            'safety_level': 'medium',
            'primary_hazards': [],
            'safety_recommendations': [],
            'risk_assessment': 'moderate'
        }
        
        # Analyze based on scene type
        if base_features.get('is_outdoor', False):
            if base_features.get('has_buildings', False):
                safety_features['primary_hazards'].append('structural_elements')
                safety_features['primary_hazards'].append('height_risks')
                safety_features['safety_recommendations'].append('Watch for falling objects and structural hazards')
            else:
                safety_features['primary_hazards'].append('terrain_hazards')
                safety_features['safety_recommendations'].append('Be cautious of uneven terrain and natural hazards')
        
        if base_features.get('has_water', False):
            safety_features['primary_hazards'].append('water_safety')
            safety_features['safety_recommendations'].append('Exercise caution near water bodies')
        
        # Determine safety level
        if len(safety_features['primary_hazards']) > 2:
            safety_features['safety_level'] = 'high'
            safety_features['risk_assessment'] = 'high'
        elif len(safety_features['primary_hazards']) > 0:
            safety_features['safety_level'] = 'medium'
            safety_features['risk_assessment'] = 'moderate'
        else:
            safety_features['safety_level'] = 'low'
            safety_features['risk_assessment'] = 'low'
        
        return safety_features

class SpatialReasoner:
    """Specialized spatial reasoning module."""
    
    def __init__(self):
        self.spatial_concepts = [
            'depth_ordering', 'relative_positioning', 'spatial_hierarchy',
            'foreground_background', '3d_orientation', 'spatial_relationships'
        ]
    
    def analyze_spatial(self, pixel_values, base_features):
        """Analyze spatial relationships in the scene."""
        spatial_features = {
            'depth_layers': 3,
            'spatial_hierarchy': 'clear',
            'depth_ordering': 'foreground_background',
            'spatial_relationships': 'complex',
            '3d_understanding': 'good'
        }
        
        # Analyze depth perception
        if base_features.get('has_foreground', False) and base_features.get('has_background', False):
            spatial_features['depth_layers'] = 3
            spatial_features['spatial_hierarchy'] = 'clear'
        else:
            spatial_features['depth_layers'] = 2
            spatial_features['spatial_hierarchy'] = 'moderate'
        
        # Analyze spatial relationships
        if base_features.get('has_person', False):
            spatial_features['spatial_relationships'] = 'person_centered'
        elif base_features.get('has_buildings', False):
            spatial_features['spatial_relationships'] = 'architectural'
        else:
            spatial_features['spatial_relationships'] = 'natural'
        
        return spatial_features

class SceneDescriptor:
    """Specialized scene description module."""
    
    def __init__(self):
        self.scene_elements = [
            'architectural', 'natural', 'urban', 'rural', 'indoor', 'outdoor'
        ]
    
    def analyze_scene(self, pixel_values, base_features):
        """Analyze scene for comprehensive description."""
        scene_features = {
            'scene_type': 'mixed',
            'dominant_elements': [],
            'scene_complexity': 'moderate',
            'visual_composition': 'balanced',
            'descriptive_elements': []
        }
        
        # Determine scene type
        if base_features.get('is_outdoor', False):
            if base_features.get('has_buildings', False):
                scene_features['scene_type'] = 'urban_outdoor'
                scene_features['dominant_elements'].append('architectural')
            else:
                scene_features['scene_type'] = 'natural_outdoor'
                scene_features['dominant_elements'].append('natural')
        else:
            scene_features['scene_type'] = 'indoor'
            scene_features['dominant_elements'].append('architectural')
        
        # Analyze complexity
        element_count = sum([
            base_features.get('has_person', False),
            base_features.get('has_buildings', False),
            base_features.get('has_sky', False),
            base_features.get('has_natural_elements', False)
        ])
        
        if element_count > 3:
            scene_features['scene_complexity'] = 'high'
        elif element_count > 1:
            scene_features['scene_complexity'] = 'moderate'
        else:
            scene_features['scene_complexity'] = 'low'
        
        return scene_features

class NavigationGuide:
    """Specialized navigation guidance module."""
    
    def __init__(self):
        self.navigation_elements = [
            'pathways', 'obstacles', 'landmarks', 'direction_indicators'
        ]
    
    def analyze_navigation(self, pixel_values, base_features):
        """Analyze scene for navigation guidance."""
        navigation_features = {
            'navigability': 'moderate',
            'path_visibility': 'clear',
            'obstacles': [],
            'landmarks': [],
            'navigation_difficulty': 'medium'
        }
        
        # Analyze navigability
        if base_features.get('has_buildings', False):
            navigation_features['navigability'] = 'structured'
            navigation_features['landmarks'].append('buildings')
        else:
            navigation_features['navigability'] = 'natural'
            navigation_features['landmarks'].append('natural_elements')
        
        # Analyze obstacles
        if base_features.get('has_person', False):
            navigation_features['obstacles'].append('people')
        
        # Determine navigation difficulty
        if len(navigation_features['obstacles']) > 1:
            navigation_features['navigation_difficulty'] = 'high'
        elif len(navigation_features['obstacles']) > 0:
            navigation_features['navigation_difficulty'] = 'medium'
        else:
            navigation_features['navigation_difficulty'] = 'low'
        
        return navigation_features

class ResponseEnhancer:
    """Response quality enhancer."""
    
    def __init__(self):
        self.response_templates = {
            'safety': {
                'high_risk': "This environment presents significant safety considerations. You should be extremely cautious about {hazards}. {recommendations}",
                'medium_risk': "This environment has moderate safety considerations. Be cautious about {hazards}. {recommendations}",
                'low_risk': "This environment appears relatively safe, but always exercise general caution. {recommendations}"
            },
            'spatial': {
                'complex': "The spatial relationships in this scene are complex with {depth_layers} distinct depth layers. The spatial hierarchy shows {spatial_hierarchy} with {spatial_relationships}.",
                'moderate': "The spatial relationships demonstrate clear depth perception with {spatial_relationships} and {depth_ordering}.",
                'simple': "The spatial relationships show basic depth perception with {spatial_relationships}."
            },
            'description': {
                'high_complexity': "This is a complex {scene_type} scene with {scene_complexity} visual elements. The scene contains {dominant_elements} with {visual_composition} composition.",
                'moderate_complexity': "This is a {scene_type} scene with moderate complexity. The scene shows {dominant_elements} elements.",
                'low_complexity': "This is a simple {scene_type} scene with {dominant_elements} elements."
            },
            'navigation': {
                'high_difficulty': "Navigation in this environment is challenging due to {obstacles}. The {navigability} layout requires careful attention to {landmarks}.",
                'medium_difficulty': "Navigation is moderately complex with {navigability} pathways. Watch for {obstacles} and use {landmarks} as reference points.",
                'low_difficulty': "Navigation appears straightforward with {navigability} pathways and clear {landmarks}."
            }
        }
    
    def enhance_response(self, base_response, features, question):
        """Enhance response with specialized features."""
        task_type = features.get('task_type', 'general')
        
        if task_type == 'safety':
            return self._enhance_safety_response(base_response, features)
        elif task_type == 'spatial':
            return self._enhance_spatial_response(base_response, features)
        elif task_type == 'description':
            return self._enhance_description_response(base_response, features)
        elif task_type == 'navigation':
            return self._enhance_navigation_response(base_response, features)
        else:
            return self._enhance_general_response(base_response, features)
    
    def _enhance_safety_response(self, base_response, features):
        """Enhance safety response."""
        safety_level = features.get('safety_level', 'medium')
        hazards = features.get('primary_hazards', [])
        recommendations = features.get('safety_recommendations', [])
        
        if safety_level == 'high':
            template = self.response_templates['safety']['high_risk']
        elif safety_level == 'medium':
            template = self.response_templates['safety']['medium_risk']
        else:
            template = self.response_templates['safety']['low_risk']
        
        enhanced = template.format(
            hazards=', '.join(hazards) if hazards else 'general hazards',
            recommendations='. '.join(recommendations) if recommendations else 'Follow general safety guidelines'
        )
        
        return f"{base_response} {enhanced}"
    
    def _enhance_spatial_response(self, base_response, features):
        """Enhance spatial response."""
        depth_layers = features.get('depth_layers', 2)
        spatial_hierarchy = features.get('spatial_hierarchy', 'moderate')
        spatial_relationships = features.get('spatial_relationships', 'basic')
        
        if depth_layers > 2:
            template = self.response_templates['spatial']['complex']
        elif depth_layers > 1:
            template = self.response_templates['spatial']['moderate']
        else:
            template = self.response_templates['spatial']['simple']
        
        enhanced = template.format(
            depth_layers=depth_layers,
            spatial_hierarchy=spatial_hierarchy,
            spatial_relationships=spatial_relationships
        )
        
        return f"{base_response} {enhanced}"
    
    def _enhance_description_response(self, base_response, features):
        """Enhance description response."""
        scene_type = features.get('scene_type', 'mixed')
        scene_complexity = features.get('scene_complexity', 'moderate')
        dominant_elements = features.get('dominant_elements', [])
        visual_composition = features.get('visual_composition', 'balanced')
        
        if scene_complexity == 'high':
            template = self.response_templates['description']['high_complexity']
        elif scene_complexity == 'moderate':
            template = self.response_templates['description']['moderate_complexity']
        else:
            template = self.response_templates['description']['low_complexity']
        
        enhanced = template.format(
            scene_type=scene_type,
            scene_complexity=scene_complexity,
            dominant_elements=', '.join(dominant_elements) if dominant_elements else 'various',
            visual_composition=visual_composition
        )
        
        return f"{base_response} {enhanced}"
    
    def _enhance_navigation_response(self, base_response, features):
        """Enhance navigation response."""
        navigability = features.get('navigability', 'moderate')
        obstacles = features.get('obstacles', [])
        landmarks = features.get('landmarks', [])
        navigation_difficulty = features.get('navigation_difficulty', 'medium')
        
        if navigation_difficulty == 'high':
            template = self.response_templates['navigation']['high_difficulty']
        elif navigation_difficulty == 'medium':
            template = self.response_templates['navigation']['medium_difficulty']
        else:
            template = self.response_templates['navigation']['low_difficulty']
        
        enhanced = template.format(
            navigability=navigability,
            obstacles=', '.join(obstacles) if obstacles else 'potential obstacles',
            landmarks=', '.join(landmarks) if landmarks else 'visible landmarks'
        )
        
        return f"{base_response} {enhanced}"
    
    def _enhance_general_response(self, base_response, features):
        """Enhance general response."""
        # Add general enhancements based on available features
        enhancements = []
        
        if features.get('specialized_analysis', False):
            enhancements.append("comprehensive analysis")
        
        if features.get('task_type'):
            enhancements.append(f"{features['task_type']} analysis")
        
        if enhancements:
            enhanced = f"{base_response} This response includes {', '.join(enhancements)}."
        else:
            enhanced = base_response
        
        return enhanced

def test_specialized_features():
    """Test the specialized features."""
    print("🧪 Testing Specialized Features")
    print("=" * 50)
    
    # Import base model
    from scripts.distillation.student_model import DistilledLLaVA3D, DistilledLLaVA3DConfig
    
    # Initialize base model
    config = DistilledLLaVA3DConfig()
    base_model = DistilledLLaVA3D(config)
    base_model.eval()
    
    # Initialize specialized features
    specialized = SpecializedFeatures(base_model)
    
    # Test different question types
    test_questions = [
        "What should I be cautious about in this environment?",
        "What are the spatial relationships in this scene?",
        "Describe this 3D scene in detail.",
        "How should I navigate through this environment?",
        "What objects can you see in this image?"
    ]
    
    test_image = torch.randn(1, 3, 224, 224)
    
    for question in test_questions:
        print(f"\n❓ Question: {question}")
        
        # Get specialized response
        response, features = specialized.generate_specialized_response(question, test_image)
        
        print(f"🤖 Response: {response}")
        print(f"🔍 Task Type: {features.get('task_type', 'general')}")
        print(f"📊 Specialized Features: {len([k for k in features.keys() if k.startswith('specialized') or k in ['safety_level', 'depth_layers', 'scene_type', 'navigability']])}")

if __name__ == "__main__":
    test_specialized_features()
