"""
Robust Fuzzy Neural Network Engine
Implements 10 fuzzy rules with membership functions and adaptive inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class RobustFuzzyInferenceEngine(nn.Module):
    """
    Robust Fuzzy Inference Engine with 10 fuzzy rules.
    Implements membership functions, normalized firing strength, and MLP-based defuzzification.
    """
    
    def __init__(self, input_dim: int, num_rules: int = 10, hidden_dim: int = 64):
        """
        Args:
            input_dim: Input feature dimension
            num_rules: Number of fuzzy rules (default: 10)
            hidden_dim: Hidden dimension for defuzzification MLP
        """
        super(RobustFuzzyInferenceEngine, self).__init__()
        
        self.input_dim = input_dim
        self.num_rules = num_rules
        
        # Membership function parameters (Gaussian)
        # Each rule has centers and widths for each input dimension
        self.rule_centers = nn.Parameter(torch.randn(num_rules, input_dim))
        self.rule_widths = nn.Parameter(torch.ones(num_rules, input_dim))
        
        # Rule weights (consequent parameters)
        self.rule_weights = nn.Parameter(torch.randn(num_rules, input_dim))
        
        # Defuzzification MLP
        self.defuzzifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Initialize parameters
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize fuzzy rule parameters."""
        # Initialize centers with small random values
        nn.init.normal_(self.rule_centers, mean=0.0, std=0.1)
        
        # Initialize widths to reasonable values
        nn.init.constant_(self.rule_widths, 1.0)
        
        # Initialize rule weights
        nn.init.normal_(self.rule_weights, mean=0.0, std=0.1)
    
    def compute_membership(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Gaussian membership values for each rule.
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            Membership values [batch_size, num_rules, input_dim]
        """
        # Expand dimensions for broadcasting
        # x: [batch_size, input_dim]
        # rule_centers: [num_rules, input_dim]
        # rule_widths: [num_rules, input_dim]
        
        x_expanded = x.unsqueeze(1)  # [batch_size, 1, input_dim]
        centers_expanded = self.rule_centers.unsqueeze(0)  # [1, num_rules, input_dim]
        widths_expanded = self.rule_widths.unsqueeze(0)  # [1, num_rules, input_dim]
        
        # Gaussian membership: exp(-0.5 * ((x - center) / width)^2)
        diff = x_expanded - centers_expanded
        membership = torch.exp(-0.5 * torch.pow(diff / (widths_expanded + 1e-8), 2))
        
        return membership  # [batch_size, num_rules, input_dim]
    
    def compute_firing_strength(self, membership: torch.Tensor) -> torch.Tensor:
        """
        Compute normalized firing strength for each rule.
        
        Args:
            membership: Membership values [batch_size, num_rules, input_dim]
            
        Returns:
            Normalized firing strength [batch_size, num_rules]
        """
        # Product inference: multiply membership values across dimensions
        # Then take minimum (T-norm) or product
        firing_strength = torch.prod(membership, dim=2)  # [batch_size, num_rules]
        
        # Normalize firing strength
        firing_sum = torch.sum(firing_strength, dim=1, keepdim=True) + 1e-8
        normalized_firing = firing_strength / firing_sum
        
        return normalized_firing
    
    def compute_rule_output(self, x: torch.Tensor, firing_strength: torch.Tensor) -> torch.Tensor:
        """
        Compute rule outputs using weighted sum.
        
        Args:
            x: Input features [batch_size, input_dim]
            firing_strength: Normalized firing strength [batch_size, num_rules]
            
        Returns:
            Rule outputs [batch_size, input_dim]
        """
        # Weighted combination of rule consequents
        # Each rule has a weight vector
        rule_outputs = self.rule_weights.unsqueeze(0)  # [1, num_rules, input_dim]
        
        # Weight by firing strength
        firing_expanded = firing_strength.unsqueeze(2)  # [batch_size, num_rules, 1]
        weighted_output = torch.sum(rule_outputs * firing_expanded, dim=1)  # [batch_size, input_dim]
        
        return weighted_output
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through fuzzy inference engine.
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            Defuzzified output [batch_size, input_dim]
        """
        # Compute membership
        membership = self.compute_membership(x)  # [batch_size, num_rules, input_dim]
        
        # Compute normalized firing strength φ_k
        firing_strength = self.compute_firing_strength(membership)  # [batch_size, num_rules]
        
        # Compute rule outputs
        rule_output = self.compute_rule_output(x, firing_strength)  # [batch_size, input_dim]
        
        # Combine with input (residual connection)
        combined = x + rule_output
        
        # Defuzzification through MLP
        output = self.defuzzifier(combined)
        
        return output

