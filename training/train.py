"""
Training Script for SFGE Model
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.sfge_model import SFGEModel
from utils.graph_builder import load_dataset_from_csv
from utils.simplicial_utils import SimplicialComplexBuilder
from utils.metrics import compute_metrics


class GraphDataset(Dataset):
    """Dataset class for molecular graphs with simplicial complexes."""
    
    def __init__(self, graphs, simplicial_complexes):
        self.graphs = graphs
        self.simplicial_complexes = simplicial_complexes
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return self.graphs[idx], self.simplicial_complexes[idx]


def build_simplicial_complexes(graphs, device='cpu'):
    """Build simplicial complexes for all graphs."""
    builder = SimplicialComplexBuilder()
    complexes = []
    
    for graph in tqdm(graphs, desc="Building simplicial complexes"):
        builder.build_from_graph(graph)
        L0, L1_d, L1_u, L2_d = builder.get_laplacians()
        
        # Use L0 as L_d and L1_d as L_u for node-level processing
        # For simplicity, we'll use L0 for both lower and upper
        L_d = L0.to(device)
        L_u = L0.to(device)  # Can be modified to use L1_u if needed
        
        complexes.append({
            'L_d': L_d,
            'L_u': L_u
        })
    
    return complexes


def collate_fn(batch):
    """Custom collate function for batching graphs."""
    graphs, complexes = zip(*batch)
    
    # For now, we'll process one graph at a time
    # In a full implementation, you might want to batch multiple graphs
    return graphs, complexes


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch_idx, (graphs, complexes) in enumerate(dataloader):
        for graph, complex_data in zip(graphs, complexes):
            # Move data to device
            x = graph.x.to(device)
            edge_index = graph.edge_index.to(device)
            L_d = complex_data['L_d'].to(device)
            L_u = complex_data['L_u'].to(device)
            y = graph.y.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            logits = model(x, edge_index, L_d, L_u)
            
            # For graph-level classification, we need to aggregate node-level predictions
            # Simple approach: mean pooling
            graph_logits = logits.mean(dim=0, keepdim=True)
            
            # Compute loss
            loss = criterion(graph_logits, y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Store predictions
            preds = torch.argmax(graph_logits, dim=1).cpu().numpy()
            labels = y.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss, np.array(all_preds), np.array(all_labels)


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probas = []
    
    with torch.no_grad():
        for graphs, complexes in dataloader:
            for graph, complex_data in zip(graphs, complexes):
                x = graph.x.to(device)
                edge_index = graph.edge_index.to(device)
                L_d = complex_data['L_d'].to(device)
                L_u = complex_data['L_u'].to(device)
                y = graph.y.to(device)
                
                # Forward pass
                logits = model(x, edge_index, L_d, L_u)
                graph_logits = logits.mean(dim=0, keepdim=True)
                
                # Compute loss
                loss = criterion(graph_logits, y)
                total_loss += loss.item()
                
                # Store predictions
                probas = torch.softmax(graph_logits, dim=1).cpu().numpy()
                preds = torch.argmax(graph_logits, dim=1).cpu().numpy()
                labels = y.cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels)
                all_probas.extend(probas[:, 1])  # Probability of positive class
    
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss, np.array(all_preds), np.array(all_labels), np.array(all_probas)


def train_model(
    train_graphs,
    val_graphs,
    model,
    device,
    num_epochs=100,
    lr=0.001,
    patience=15,
    batch_size=1,
    checkpoint_dir='outputs/checkpoints'
):
    """Main training function."""
    
    # Build simplicial complexes
    print("Building simplicial complexes for training set...")
    train_complexes = build_simplicial_complexes(train_graphs, device)
    print("Building simplicial complexes for validation set...")
    val_complexes = build_simplicial_complexes(val_graphs, device)
    
    # Create datasets and dataloaders
    train_dataset = GraphDataset(train_graphs, train_complexes)
    val_dataset = GraphDataset(val_graphs, val_complexes)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'val_auc': []
    }
    
    best_val_loss = float('inf')
    best_val_auc = 0.0
    patience_counter = 0
    best_model_state = None
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Device: {device}")
    print(f"Learning rate: {lr}")
    print(f"Early stopping patience: {patience}\n")
    
    for epoch in range(num_epochs):
        # Train
        train_loss, train_preds, train_labels = train_epoch(model, train_loader, criterion, optimizer, device)
        train_metrics = compute_metrics(train_labels, train_preds, train_preds)
        train_acc = train_metrics['accuracy']
        
        # Validate
        val_loss, val_preds, val_labels, val_probas = validate(model, val_loader, criterion, device)
        val_metrics = compute_metrics(val_labels, val_preds, val_probas)
        val_acc = val_metrics['accuracy']
        val_auc = val_metrics['auc']
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")
        
        # Early stopping and checkpointing
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            
            # Save checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_auc': val_auc,
                'history': history
            }, checkpoint_path)
            print(f"  ✓ Saved best model (AUC: {val_auc:.4f})")
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
        
        print()
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history

