"""
Simplicial Complex Utilities
Builds simplicial complexes from molecular graphs and computes incidence matrices and Laplacians.
"""

from typing import Tuple, List, Optional
import numpy as np
import torch
import torch.sparse as sparse
from torch_geometric.data import Data


class SimplicialComplexBuilder:
    """
    Builds simplicial complexes (K=2) from molecular graphs.
    Extracts 0-simplices (nodes), 1-simplices (edges), and 2-simplices (triangles).
    """
    
    def __init__(self):
        self.B1: Optional[torch.Tensor] = None  # Edge-node incidence matrix
        self.B2: Optional[torch.Tensor] = None  # Triangle-edge incidence matrix
        self.L0_d: Optional[torch.Tensor] = None  # Lower Laplacian for 0-simplices
        self.L0_u: Optional[torch.Tensor] = None  # Upper Laplacian for 0-simplices
        self.L1_d: Optional[torch.Tensor] = None  # Lower Laplacian for 1-simplices
        self.L1_u: Optional[torch.Tensor] = None  # Upper Laplacian for 1-simplices
        self.L2_d: Optional[torch.Tensor] = None  # Lower Laplacian for 2-simplices
        self.triangles: Optional[List[Tuple[int, int, int]]] = None
        
    def build_from_graph(self, data: Data) -> 'SimplicialComplexBuilder':
        """
        Build simplicial complex from a PyTorch Geometric Data object.
        
        Args:
            data: PyTorch Geometric Data object with edge_index
            
        Returns:
            self for method chaining
        """
        edge_index = data.edge_index.cpu().numpy()
        num_nodes = data.x.shape[0]
        num_edges = edge_index.shape[1] // 2  # Undirected graph
        
        # Extract unique edges (undirected)
        edges = set()
        edge_list = []
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            if src < dst:  # Only keep one direction
                edges.add((src, dst))
                edge_list.append((src, dst))
        
        # Build B1: Edge-Node incidence matrix (num_edges x num_nodes)
        B1 = np.zeros((num_edges, num_nodes))
        edge_to_idx = {edge: idx for idx, edge in enumerate(edge_list)}
        
        for idx, (src, dst) in enumerate(edge_list):
            B1[idx, src] = 1
            B1[idx, dst] = -1
        
        self.B1 = torch.tensor(B1, dtype=torch.float32)
        
        # Find triangles (2-simplices)
        triangles = self._find_triangles(edge_list, num_nodes)
        self.triangles = triangles
        num_triangles = len(triangles)
        
        if num_triangles > 0:
            # Build B2: Triangle-Edge incidence matrix (num_triangles x num_edges)
            B2 = np.zeros((num_triangles, num_edges))
            
            for tri_idx, (v0, v1, v2) in enumerate(triangles):
                # Each triangle has 3 edges
                edges_in_triangle = [
                    (min(v0, v1), max(v0, v1)),
                    (min(v1, v2), max(v1, v2)),
                    (min(v0, v2), max(v0, v2))
                ]
                
                for edge in edges_in_triangle:
                    if edge in edge_to_idx:
                        edge_idx = edge_to_idx[edge]
                        # Orientation: +1 if edge direction matches triangle orientation
                        B2[tri_idx, edge_idx] = 1.0
            
            self.B2 = torch.tensor(B2, dtype=torch.float32)
        else:
            # No triangles found
            self.B2 = torch.zeros((0, num_edges), dtype=torch.float32)
        
        # Compute Laplacians
        self._compute_laplacians()
        
        return self
    
    def _find_triangles(self, edges: List[Tuple[int, int]], num_nodes: int) -> List[Tuple[int, int, int]]:
        """
        Find all triangles in the graph.
        
        Args:
            edges: List of edges as (src, dst) tuples
            num_nodes: Number of nodes in the graph
            
        Returns:
            List of triangles as (v0, v1, v2) tuples
        """
        # Build adjacency list
        adj_list = {i: [] for i in range(num_nodes)}
        for src, dst in edges:
            adj_list[src].append(dst)
            adj_list[dst].append(src)
        
        triangles = []
        visited = set()
        
        for v0 in range(num_nodes):
            neighbors_v0 = adj_list[v0]
            for v1 in neighbors_v0:
                if v1 <= v0:
                    continue
                neighbors_v1 = adj_list[v1]
                # Find common neighbors
                for v2 in neighbors_v1:
                    if v2 > v1 and v2 in neighbors_v0:
                        # Found triangle (v0, v1, v2)
                        triangle = tuple(sorted([v0, v1, v2]))
                        if triangle not in visited:
                            triangles.append(triangle)
                            visited.add(triangle)
        
        return triangles
    
    def _compute_laplacians(self):
        """Compute all Laplacian matrices."""
        if self.B1 is None:
            return
        
        # L0 (node-level): L0 = B1^T @ B1
        L0 = torch.mm(self.B1.t(), self.B1)
        self.L0_d = L0  # Lower Laplacian for 0-simplices
        self.L0_u = L0  # For nodes, lower and upper are the same
        
        if self.B2 is not None and self.B2.shape[0] > 0:
            # L1 (edge-level): L1_d = B1 @ B1^T, L1_u = B2^T @ B2
            L1_d = torch.mm(self.B1, self.B1.t())
            L1_u = torch.mm(self.B2.t(), self.B2)
            self.L1_d = L1_d
            self.L1_u = L1_u
            
            # L2 (triangle-level): L2_d = B2 @ B2^T
            L2_d = torch.mm(self.B2, self.B2.t())
            self.L2_d = L2_d
        else:
            # No triangles, so L1_u and L2_d are zero matrices
            num_edges = self.B1.shape[0]
            self.L1_d = torch.mm(self.B1, self.B1.t())
            self.L1_u = torch.zeros((num_edges, num_edges), dtype=torch.float32)
            self.L2_d = torch.zeros((0, 0), dtype=torch.float32)
    
    def get_laplacians(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get all Laplacian matrices.
        
        Returns:
            Tuple of (L0, L1_d, L1_u, L2_d)
        """
        L0 = self.L0_d if self.L0_d is not None else torch.zeros((1, 1))
        L1_d = self.L1_d if self.L1_d is not None else torch.zeros((1, 1))
        L1_u = self.L1_u if self.L1_u is not None else torch.zeros((1, 1))
        L2_d = self.L2_d if self.L2_d is not None else torch.zeros((1, 1))
        
        return L0, L1_d, L1_u, L2_d
    
    def get_incidence_matrices(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get incidence matrices.
        
        Returns:
            Tuple of (B1, B2)
        """
        B1 = self.B1 if self.B1 is not None else torch.zeros((1, 1))
        B2 = self.B2 if self.B2 is not None else torch.zeros((1, 1))
        
        return B1, B2

