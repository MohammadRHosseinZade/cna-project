"""
Simplicial Convolution Layers
Implements simplicial neural network layers for processing simplicial complexes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class SimplicialConvolutionLayer(nn.Module):
    """
    Simplicial Convolution Layer that processes features using lower and upper Laplacians.
    
    Implements: X' = P( Σ L_d^p X + Σ L_u^p X + X_H )
    where P is the aggregation-selection-reduction operation.
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        k_snn: int = 2,
        d_snn: int = 64,
        num_powers: int = 2
    ):
        """
        Args:
            in_dim: Input feature dimension
            out_dim: Output feature dimension
            k_snn: Number of SNN layers
            d_snn: Hidden dimension for SNN
            num_powers: Number of Laplacian powers to use
        """
        super(SimplicialConvolutionLayer, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.k_snn = k_snn
        self.d_snn = d_snn
        self.num_powers = num_powers
        
        # Power-specific transformation layers
        self.lower_transformers = nn.ModuleList([
            nn.Linear(in_dim, d_snn) for _ in range(num_powers)
        ])
        self.upper_transformers = nn.ModuleList([
            nn.Linear(in_dim, d_snn) for _ in range(num_powers)
        ])
        
        # Aggregation layer (P operation)
        self.aggregation = nn.Sequential(
            nn.Linear(d_snn * num_powers * 2, d_snn),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Selection and reduction
        self.selection = nn.Sequential(
            nn.Linear(d_snn, d_snn),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.reduction = nn.Linear(d_snn, out_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        L_d: torch.Tensor,
        L_u: torch.Tensor,
        identity: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through simplicial convolution layer.
        
        Args:
            x: Input features [num_nodes, in_dim]
            L_d: Lower Laplacian matrix [num_nodes, num_nodes]
            L_u: Upper Laplacian matrix [num_nodes, num_nodes]
            identity: Optional identity matrix for residual connection
            
        Returns:
            Output features [num_nodes, out_dim]
        """
        # Normalize Laplacians
        L_d_norm = self._normalize_laplacian(L_d)
        L_u_norm = self._normalize_laplacian(L_u)
        
        # Compute powers and aggregate
        lower_features = []
        upper_features = []
        
        # Lower Laplacian powers
        L_d_power = torch.eye(L_d_norm.shape[0], device=x.device, dtype=x.dtype)
        for p in range(self.num_powers):
            if p > 0:
                L_d_power = torch.mm(L_d_power, L_d_norm)
            transformed = self.lower_transformers[p](x)
            lower_feat = torch.mm(L_d_power, transformed)
            lower_features.append(lower_feat)
        
        # Upper Laplacian powers
        L_u_power = torch.eye(L_u_norm.shape[0], device=x.device, dtype=x.dtype)
        for p in range(self.num_powers):
            if p > 0:
                L_u_power = torch.mm(L_u_power, L_u_norm)
            transformed = self.upper_transformers[p](x)
            upper_feat = torch.mm(L_u_power, transformed)
            upper_features.append(upper_feat)
        
        # Concatenate all features
        all_features = torch.cat(lower_features + upper_features, dim=1)
        
        # Add identity/residual connection if provided
        if identity is not None:
            # Project identity to match dimension if needed
            if identity.shape[1] != self.d_snn * self.num_powers * 2:
                # Simple projection: pad or truncate
                if identity.shape[1] < self.d_snn * self.num_powers * 2:
                    padding = torch.zeros(identity.shape[0], 
                                         self.d_snn * self.num_powers * 2 - identity.shape[1],
                                         device=identity.device)
                    identity_proj = torch.cat([identity, padding], dim=1)
                else:
                    identity_proj = identity[:, :self.d_snn * self.num_powers * 2]
            else:
                identity_proj = identity
            all_features = all_features + identity_proj
        
        # Aggregation → Selection → Reduction (P operation)
        aggregated = self.aggregation(all_features)
        selected = self.selection(aggregated)
        output = self.reduction(selected)
        
        return output
    
    def _normalize_laplacian(self, L: torch.Tensor) -> torch.Tensor:
        """
        Normalize Laplacian matrix.
        
        Args:
            L: Laplacian matrix
            
        Returns:
            Normalized Laplacian
        """
        # Add small epsilon to avoid division by zero
        diag = torch.diag(L)
        diag_sqrt = torch.sqrt(torch.clamp(diag, min=1e-8))
        D_inv_sqrt = torch.diag(1.0 / diag_sqrt)
        
        # Normalized Laplacian: D^(-1/2) L D^(-1/2)
        L_norm = torch.mm(torch.mm(D_inv_sqrt, L), D_inv_sqrt)
        
        return L_norm


class MultiSimplicialLayer(nn.Module):
    """
    Multi-layer simplicial neural network.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        k_snn: int = 2,
        d_snn: int = 64,
        num_powers: int = 2
    ):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            output_dim: Output dimension
            k_snn: Number of SNN layers
            d_snn: SNN hidden dimension
            num_powers: Number of Laplacian powers
        """
        super(MultiSimplicialLayer, self).__init__()
        
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(
            SimplicialConvolutionLayer(input_dim, hidden_dim, k_snn, d_snn, num_powers)
        )
        
        # Intermediate layers
        for _ in range(k_snn - 1):
            self.layers.append(
                SimplicialConvolutionLayer(hidden_dim, hidden_dim, k_snn, d_snn, num_powers)
            )
        
        # Output layer
        self.layers.append(
            SimplicialConvolutionLayer(hidden_dim, output_dim, k_snn, d_snn, num_powers)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        L_d: torch.Tensor,
        L_u: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through multi-layer SNN.
        
        Args:
            x: Input features
            L_d: Lower Laplacian
            L_u: Upper Laplacian
            
        Returns:
            Output features
        """
        for layer in self.layers:
            x = layer(x, L_d, L_u, identity=x)
            x = F.relu(x)
        
        return x

