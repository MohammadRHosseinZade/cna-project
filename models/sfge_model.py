"""
SFGE Model: Simplicial-Fuzzy-Graph-Enhanced Model
Combines Simplicial Neural Networks, GCN, and Fuzzy Inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .simplicial_layers import MultiSimplicialLayer
from .gcn_global_layer import GlobalGCNLayer
from .fuzzy_engine import RobustFuzzyInferenceEngine


class SFGEModel(nn.Module):
    """
    Complete SFGE Model architecture.
    
    Architecture:
    1. Simplicial Neural Network (SNN) layers
    2. Global GCN enhancement
    3. Multi-view linear fusion
    4. Robust Fuzzy Inference Engine
    5. Final MLP classifier
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 2,  # Binary classification
        k_snn: int = 2,
        d_snn: int = 64,
        num_fuzzy_rules: int = 10,
        dropout: float = 0.1
    ):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            output_dim: Output dimension (number of classes)
            k_snn: Number of SNN layers
            d_snn: SNN hidden dimension
            num_fuzzy_rules: Number of fuzzy rules
            dropout: Dropout rate
        """
        super(SFGEModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Simplicial Neural Network
        self.snn = MultiSimplicialLayer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            k_snn=k_snn,
            d_snn=d_snn,
            num_powers=2
        )
        
        # Global GCN Layer
        self.gcn = GlobalGCNLayer(in_dim=input_dim, out_dim=hidden_dim)
        
        # Multi-view linear fusion layer
        # Fuses SNN output and GCN output
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Robust Fuzzy Inference Engine
        self.fuzzy_engine = RobustFuzzyInferenceEngine(
            input_dim=hidden_dim,
            num_rules=num_fuzzy_rules,
            hidden_dim=hidden_dim
        )
        
        # Final MLP classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        L_d: torch.Tensor,
        L_u: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through SFGE model.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            L_d: Lower Laplacian matrix [num_nodes, num_nodes]
            L_u: Upper Laplacian matrix [num_nodes, num_nodes]
            
        Returns:
            Logits [num_nodes, output_dim]
        """
        # 1. Simplicial Neural Network processing
        x_snn = self.snn(x, L_d, L_u)  # [num_nodes, hidden_dim]
        
        # 2. Global GCN enhancement
        x_gcn = self.gcn(x, edge_index)  # [num_nodes, hidden_dim]
        
        # 3. Multi-view linear fusion
        x_fused = torch.cat([x_snn, x_gcn], dim=1)  # [num_nodes, hidden_dim * 2]
        x_fused = self.fusion(x_fused)  # [num_nodes, hidden_dim]
        
        # 4. Robust Fuzzy Inference
        x_fuzzy = self.fuzzy_engine(x_fused)  # [num_nodes, hidden_dim]
        
        # 5. Final classification
        logits = self.classifier(x_fuzzy)  # [num_nodes, output_dim]
        
        return logits
    
    def predict_proba(self, x: torch.Tensor, edge_index: torch.Tensor, 
                     L_d: torch.Tensor, L_u: torch.Tensor) -> torch.Tensor:
        """
        Predict class probabilities.
        
        Args:
            x: Node features
            edge_index: Edge indices
            L_d: Lower Laplacian
            L_u: Upper Laplacian
            
        Returns:
            Class probabilities [num_nodes, output_dim]
        """
        logits = self.forward(x, edge_index, L_d, L_u)
        return F.softmax(logits, dim=1)
    
    def predict(self, x: torch.Tensor, edge_index: torch.Tensor,
               L_d: torch.Tensor, L_u: torch.Tensor) -> torch.Tensor:
        """
        Predict class labels.
        
        Args:
            x: Node features
            edge_index: Edge indices
            L_d: Lower Laplacian
            L_u: Upper Laplacian
            
        Returns:
            Predicted class labels [num_nodes]
        """
        logits = self.forward(x, edge_index, L_d, L_u)
        return torch.argmax(logits, dim=1)

