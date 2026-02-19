"""
Global GCN Layer
Implements a one-layer Graph Convolutional Network for global graph enhancement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GlobalGCNLayer(nn.Module):
    """
    One-layer GCN for global graph-level feature enhancement.
    """
    
    def __init__(self, in_dim: int, out_dim: int):
        """
        Args:
            in_dim: Input feature dimension
            out_dim: Output feature dimension
        """
        super(GlobalGCNLayer, self).__init__()
        
        self.gcn = GCNConv(in_dim, out_dim)
        self.norm = nn.BatchNorm1d(out_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GCN layer.
        
        Args:
            x: Node features [num_nodes, in_dim]
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            Output features [num_nodes, out_dim]
        """
        x = self.gcn(x, edge_index)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        return x

