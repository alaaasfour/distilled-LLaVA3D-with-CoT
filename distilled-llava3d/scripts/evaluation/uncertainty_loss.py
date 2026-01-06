#!/usr/bin/env python3
"""
Uncertainty-Based Multi-Task Loss Weighting
Implements learnable uncertainty weights for multi-task learning in 3D VLM distillation.
Based on: "Multi-Task Learning Using Uncertainty to Weigh Losses" (Kendall et al., CVPR 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional


class UncertaintyWeightedLoss(nn.Module):
    """
    Uncertainty-weighted multi-task loss.
    Learns task-specific uncertainty parameters to automatically balance losses.
    """
    
    def __init__(self, num_tasks: int = 5, init_uncertainty: float = 0.0):
        """
        Initialize uncertainty-weighted loss.
        
        Args:
            num_tasks: Number of tasks (text, depth, detection, spatial, multiview)
            init_uncertainty: Initial log-uncertainty value (0.0 = weight 1.0)
        """
        super().__init__()
        # Learnable log-uncertainty parameters (one per task)
        # Lower uncertainty = higher weight for that task
        self.log_uncertainties = nn.Parameter(torch.ones(num_tasks) * init_uncertainty)
        
    def forward(self, task_losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute uncertainty-weighted total loss.
        
        Args:
            task_losses: Dictionary mapping task names to loss tensors
                Expected keys: 'text', 'depth', 'detection', 'spatial', 'multiview'
        
        Returns:
            Weighted total loss
        """
        total_loss = 0.0
        task_names = list(task_losses.keys())
        
        for i, task_name in enumerate(task_names):
            if task_name in task_losses and task_losses[task_name] is not None:
                loss = task_losses[task_name]
                
                # Weight = 1 / (2 * sigma^2), where sigma^2 = exp(log_uncertainty)
                # This gives: weight = 1 / (2 * exp(log_uncertainty)) = 0.5 * exp(-log_uncertainty)
                # For numerical stability, we use: weight = exp(-log_uncertainty) / 2
                log_sigma_sq = self.log_uncertainties[i]
                precision = torch.exp(-log_sigma_sq)  # 1 / sigma^2
                weighted_loss = precision * loss + log_sigma_sq
                
                total_loss += weighted_loss
        
        return total_loss
    
    def get_task_weights(self) -> Dict[str, float]:
        """Get current task weights (for logging/monitoring)."""
        weights = {}
        for i, log_sigma_sq in enumerate(self.log_uncertainties):
            weight = torch.exp(-log_sigma_sq).item()
            weights[f'task_{i}'] = weight
        return weights
    
    def get_uncertainties(self) -> Dict[str, float]:
        """Get current uncertainty values (for logging/monitoring)."""
        uncertainties = {}
        for i, log_sigma_sq in enumerate(self.log_uncertainties):
            sigma_sq = torch.exp(log_sigma_sq).item()
            uncertainties[f'task_{i}'] = sigma_sq
        return uncertainties


class AdaptiveUncertaintyLoss(nn.Module):
    """
    Adaptive uncertainty-weighted loss with task difficulty estimation.
    Automatically adjusts weights based on task difficulty and training progress.
    """
    
    def __init__(self, num_tasks: int = 5, 
                 adaptation_rate: float = 0.1,
                 min_weight: float = 0.01,
                 max_weight: float = 10.0):
        """
        Initialize adaptive uncertainty-weighted loss.
        
        Args:
            num_tasks: Number of tasks
            adaptation_rate: Rate at which weights adapt (0.0 = static, 1.0 = fully adaptive)
            min_weight: Minimum task weight
            max_weight: Maximum task weight
        """
        super().__init__()
        self.num_tasks = num_tasks
        self.adaptation_rate = adaptation_rate
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        # Learnable log-uncertainties (initialize to small positive values to prevent explosion)
        # Initialize to log(0.5) so initial precision is 0.5
        self.log_uncertainties = nn.Parameter(torch.ones(num_tasks) * np.log(2.0))
        
        # Track task difficulties (running averages)
        # Initialize to equal difficulty (1/num_tasks)
        self.register_buffer('task_difficulties', torch.ones(num_tasks) / num_tasks)
        self.register_buffer('task_losses_ema', torch.ones(num_tasks) * 0.1)  # Initialize to small non-zero value
        
    def forward(self, task_losses: Dict[str, torch.Tensor], 
                task_names: Optional[list] = None) -> torch.Tensor:
        """
        Compute adaptive uncertainty-weighted total loss.
        
        Args:
            task_losses: Dictionary mapping task names to loss tensors
            task_names: Optional list of task names in order (if None, uses dict keys)
        
        Returns:
            Weighted total loss
        """
        if task_names is None:
            task_names = list(task_losses.keys())
        
        total_loss = 0.0
        
        for i, task_name in enumerate(task_names):
            if i >= self.num_tasks:
                break
                
            if task_name in task_losses and task_losses[task_name] is not None:
                loss = task_losses[task_name]
                
                # Update task difficulty estimate (exponential moving average)
                with torch.no_grad():
                    loss_value = max(loss.item(), 1e-6)  # Prevent zero loss
                    self.task_losses_ema[i] = 0.9 * self.task_losses_ema[i] + 0.1 * loss_value
                    # Difficulty = normalized loss (relative to other tasks)
                    # Add small epsilon to prevent division issues
                    ema_sum = self.task_losses_ema.sum()
                    if ema_sum > 1e-6:
                        self.task_difficulties[i] = self.task_losses_ema[i] / (ema_sum + 1e-6)
                    else:
                        self.task_difficulties[i] = 1.0 / self.num_tasks  # Equal difficulty
                
                # Adaptive uncertainty: higher difficulty = lower uncertainty (higher weight)
                # But we also learn uncertainty, so combine both
                # Clamp difficulty to prevent extreme values
                clamped_difficulty = torch.clamp(self.task_difficulties[i], min=1e-4, max=0.99)
                difficulty_factor = 1.0 / (clamped_difficulty + 1e-6)
                
                # Clamp difficulty_factor to prevent explosion
                difficulty_factor = torch.clamp(difficulty_factor, min=0.1, max=100.0)
                
                learned_precision = torch.exp(-self.log_uncertainties[i])
                
                # Combine learned and adaptive weights
                adaptive_precision = learned_precision * (1.0 - self.adaptation_rate) + \
                                   difficulty_factor * self.adaptation_rate
                
                # Clamp precision to valid range (more conservative)
                adaptive_precision = torch.clamp(adaptive_precision, 
                                                self.min_weight / 2.0, 
                                                min(self.max_weight / 2.0, 50.0))  # Cap at 50
                
                # Weighted loss: precision * loss + regularization term
                weighted_loss = adaptive_precision * loss + self.log_uncertainties[i]
                
                total_loss += weighted_loss
        
        return total_loss
    
    def get_task_weights(self, task_names: Optional[list] = None) -> Dict[str, float]:
        """Get current task weights."""
        if task_names is None:
            task_names = [f'task_{i}' for i in range(self.num_tasks)]
        
        weights = {}
        for i, task_name in enumerate(task_names):
            if i < self.num_tasks:
                learned_precision = torch.exp(-self.log_uncertainties[i]).item()
                difficulty_factor = 1.0 / (self.task_difficulties[i].item() + 1e-8)
                adaptive_precision = learned_precision * (1.0 - self.adaptation_rate) + \
                                   difficulty_factor * self.adaptation_rate
                weights[task_name] = adaptive_precision * 2.0  # Convert to weight
        
        return weights


class MultiTaskUncertaintyLoss(nn.Module):
    """
    Complete multi-task uncertainty-weighted loss for 3D VLM distillation.
    Handles: text generation, depth estimation, object detection, spatial reasoning, multiview.
    """
    
    def __init__(self, 
                 use_uncertainty: bool = True,
                 adaptation_rate: float = 0.1,
                 init_weights: Optional[Dict[str, float]] = None):
        """
        Initialize multi-task uncertainty loss.
        
        Args:
            use_uncertainty: Whether to use learnable uncertainty weights
            adaptation_rate: Rate for adaptive weighting (if use_uncertainty=True)
            init_weights: Initial static weights (if use_uncertainty=False)
                Expected keys: 'text', 'depth', 'detection', 'spatial', 'multiview', 'feature'
        """
        super().__init__()
        self.use_uncertainty = use_uncertainty
        
        if use_uncertainty:
            # 6 tasks: text, depth_ce, depth_reg, depth_kl, detection, spatial, multiview, feature
            self.uncertainty_loss = AdaptiveUncertaintyLoss(
                num_tasks=8,
                adaptation_rate=adaptation_rate
            )
        else:
            # Static weights
            self.static_weights = init_weights or {
                'text': 1.0,
                'depth_ce': 0.25,
                'depth_reg': 0.15,
                'depth_kl': 0.0125,
                'detection': 0.35,
                'spatial': 0.25,
                'multiview': 0.1,
                'feature': 0.3
            }
    
    def forward(self, 
                text_loss: Optional[torch.Tensor] = None,
                depth_ce_loss: Optional[torch.Tensor] = None,
                depth_reg_loss: Optional[torch.Tensor] = None,
                depth_kl_loss: Optional[torch.Tensor] = None,
                detection_loss: Optional[torch.Tensor] = None,
                spatial_loss: Optional[torch.Tensor] = None,
                multiview_loss: Optional[torch.Tensor] = None,
                feature_loss: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute weighted multi-task loss.
        
        Returns:
            Total weighted loss
        """
        task_losses = {
            'text': text_loss,
            'depth_ce': depth_ce_loss,
            'depth_reg': depth_reg_loss,
            'depth_kl': depth_kl_loss,
            'detection': detection_loss,
            'spatial': spatial_loss,
            'multiview': multiview_loss,
            'feature': feature_loss
        }
        
        # Filter out None losses
        task_losses = {k: v for k, v in task_losses.items() if v is not None}
        
        if self.use_uncertainty:
            task_names = ['text', 'depth_ce', 'depth_reg', 'depth_kl', 
                         'detection', 'spatial', 'multiview', 'feature']
            return self.uncertainty_loss(task_losses, task_names)
        else:
            # Static weighting
            total_loss = 0.0
            for task_name, loss in task_losses.items():
                weight = self.static_weights.get(task_name, 1.0)
                total_loss += weight * loss
            return total_loss
    
    def get_weights(self) -> Dict[str, float]:
        """Get current task weights (for logging)."""
        if self.use_uncertainty:
            return self.uncertainty_loss.get_task_weights(
                ['text', 'depth_ce', 'depth_reg', 'depth_kl', 
                 'detection', 'spatial', 'multiview', 'feature']
            )
        else:
            return self.static_weights.copy()




