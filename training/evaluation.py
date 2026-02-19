"""
Evaluation Script for SFGE Model
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.sfge_model import SFGEModel
from utils.graph_builder import load_dataset_from_csv
from utils.simplicial_utils import SimplicialComplexBuilder
from utils.metrics import compute_metrics, get_roc_curve_data


def build_simplicial_complexes(graphs, device='cpu'):
    """Build simplicial complexes for all graphs."""
    builder = SimplicialComplexBuilder()
    complexes = []
    
    for graph in tqdm(graphs, desc="Building simplicial complexes"):
        builder.build_from_graph(graph)
        L0, L1_d, L1_u, L2_d = builder.get_laplacians()
        
        L_d = L0.to(device)
        L_u = L0.to(device)
        
        complexes.append({
            'L_d': L_d,
            'L_u': L_u
        })
    
    return complexes


def evaluate_model(model, graphs, complexes, device, output_dir='outputs'):
    """Evaluate the model on a dataset."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probas = []
    
    with torch.no_grad():
        for graph, complex_data in tqdm(zip(graphs, complexes), desc="Evaluating"):
            x = graph.x.to(device)
            edge_index = graph.edge_index.to(device)
            L_d = complex_data['L_d'].to(device)
            L_u = complex_data['L_u'].to(device)
            y = graph.y.to(device)
            
            # Forward pass
            logits = model(x, edge_index, L_d, L_u)
            graph_logits = logits.mean(dim=0, keepdim=True)
            
            # Get predictions
            probas = torch.softmax(graph_logits, dim=1).cpu().numpy()
            preds = torch.argmax(graph_logits, dim=1).cpu().numpy()
            labels = y.cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels)
            all_probas.extend(probas[:, 1])
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probas = np.array(all_probas)
    
    # Compute metrics
    metrics = compute_metrics(all_labels, all_preds, all_probas)
    
    # Generate plots
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
    
    # Confusion Matrix
    plot_confusion_matrix(all_labels, all_preds, 
                         save_path=os.path.join(output_dir, 'figures', 'confusion_matrix.png'))
    
    # ROC Curve
    plot_roc_curve(all_labels, all_probas,
                   save_path=os.path.join(output_dir, 'figures', 'roc_curve.png'))
    
    return metrics, all_preds, all_labels, all_probas


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-toxic', 'Toxic'],
                yticklabels=['Non-toxic', 'Toxic'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_roc_curve(y_true, y_proba, save_path):
    """Plot and save ROC curve."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC curve saved to {save_path}")

