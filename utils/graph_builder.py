"""
Graph Builder Module
Converts SMILES strings to PyTorch Geometric Data objects.
"""

from typing import List, Optional, Tuple
import numpy as np
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import rdmolops


def smiles_to_graph(smiles: str, label: Optional[float] = None) -> Data:
    """
    Convert a SMILES string to a PyTorch Geometric Data object.
    
    Args:
        smiles: SMILES string representation of the molecule
        label: Optional label for the molecule
        
    Returns:
        Data: PyTorch Geometric Data object with node features and edge indices
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        
        # Get atom features (nodes)
        num_atoms = mol.GetNumAtoms()
        node_features = []
        
        for atom in mol.GetAtoms():
            # Basic atom features: atomic number, degree, formal charge, hybridization
            features = [
                float(atom.GetAtomicNum()),
                float(atom.GetDegree()),
                float(atom.GetFormalCharge()),
                float(atom.GetHybridization()),
                float(atom.GetNumRadicalElectrons()),
                float(atom.GetIsAromatic()),
            ]
            node_features.append(features)
        
        node_features = torch.tensor(node_features, dtype=torch.float)
        
        # Get edge indices (bonds)
        edge_indices = []
        edge_attrs = []
        
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            # Add both directions for undirected graph
            edge_indices.append([i, j])
            edge_indices.append([j, i])
            
            # Bond features: bond type, is conjugated, is in ring
            bond_type = float(bond.GetBondTypeAsDouble())
            is_conjugated = float(bond.GetIsConjugated())
            is_in_ring = float(bond.IsInRing())
            
            edge_attrs.append([bond_type, is_conjugated, is_in_ring])
            edge_attrs.append([bond_type, is_conjugated, is_in_ring])
        
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float) if edge_attrs else None
        
        # Create PyTorch Geometric Data object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor([label], dtype=torch.long) if label is not None else None,
            smiles=smiles
        )
        
        return data
        
    except Exception as e:
        raise ValueError(f"Error processing SMILES {smiles}: {str(e)}")


def load_dataset_from_csv(
    csv_path: str,
    smiles_col: str = "SMILES",
    label_col: str = "label"
) -> List[Data]:
    """
    Load dataset from CSV file and convert to list of Data objects.
    
    Args:
        csv_path: Path to CSV file
        smiles_col: Name of column containing SMILES strings
        label_col: Name of column containing labels
        
    Returns:
        List of Data objects
    """
    import pandas as pd
    
    df = pd.read_csv(csv_path)
    
    if smiles_col not in df.columns:
        raise ValueError(f"Column '{smiles_col}' not found in dataset")
    if label_col not in df.columns:
        raise ValueError(f"Column '{label_col}' not found in dataset")
    
    graphs = []
    for idx, row in df.iterrows():
        try:
            graph = smiles_to_graph(row[smiles_col], label=row[label_col])
            graphs.append(graph)
        except Exception as e:
            print(f"Warning: Skipping row {idx}: {e}")
            continue
    
    return graphs

