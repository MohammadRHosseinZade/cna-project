"""
Main Entry Point for SFGE Model Training and Evaluation
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
import json
from datetime import datetime
import random

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

from models.sfge_model import SFGEModel
from utils.graph_builder import load_dataset_from_csv
from training.train import train_model
from training.evaluation import evaluate_model, build_simplicial_complexes


def load_splits(splits_dir='splits'):
    """Load predefined train/val/test splits."""
    train_path = os.path.join(splits_dir, 'train.txt')
    val_path = os.path.join(splits_dir, 'val.txt')
    test_path = os.path.join(splits_dir, 'test.txt')
    
    splits = {}
    
    if os.path.exists(train_path):
        with open(train_path, 'r') as f:
            splits['train'] = [int(line.strip()) for line in f if line.strip()]
    
    if os.path.exists(val_path):
        with open(val_path, 'r') as f:
            splits['val'] = [int(line.strip()) for line in f if line.strip()]
    
    if os.path.exists(test_path):
        with open(test_path, 'r') as f:
            splits['test'] = [int(line.strip()) for line in f if line.strip()]
    
    return splits


def split_dataset(graphs, splits):
    """Split dataset based on predefined splits."""
    train_graphs = [graphs[i] for i in splits.get('train', [])]
    val_graphs = [graphs[i] for i in splits.get('val', [])]
    test_graphs = [graphs[i] for i in splits.get('test', [])]
    
    return train_graphs, val_graphs, test_graphs


def generate_report(history, test_metrics, args, splits, output_dir='outputs'):
    """Generate evaluation report."""
    report_path = os.path.join(output_dir, 'report.md')
    
    report = f"""# SFGE Model Evaluation Report

## Experimental Setup

### Dataset
- **Dataset**: ApisTox
- **Task**: Binary Classification (Toxic vs Non-toxic)
- **Training samples**: {len(splits.get('train', []))}
- **Validation samples**: {len(splits.get('val', []))}
- **Test samples**: {len(splits.get('test', []))}

### Model Architecture
- **Model**: SFGE (Simplicial-Fuzzy-Graph-Enhanced)
- **Components**:
  - Simplicial Neural Network (SNN) with k_SNN={args.k_snn} layers
  - Global GCN layer
  - Multi-view linear fusion
  - Robust Fuzzy Inference Engine (10 rules)
  - Final MLP classifier

### Hyperparameters
- **Learning rate**: {args.lr}
- **Batch size**: {args.batch_size}
- **Epochs**: {args.num_epochs}
- **Early stopping patience**: {args.patience}
- **Hidden dimension**: {args.hidden_dim}
- **SNN dimension**: {args.d_snn}
- **Dropout**: 0.1

## Training Results

### Training History
- **Final training loss**: {history['train_loss'][-1]:.4f}
- **Final training accuracy**: {history['train_acc'][-1]:.4f}
- **Final validation loss**: {history['val_loss'][-1]:.4f}
- **Final validation accuracy**: {history['val_acc'][-1]:.4f}
- **Best validation AUC**: {max(history['val_auc']):.4f}

### Training Curves
Training and validation metrics were logged throughout training. Check `outputs/logs/` for detailed logs.

## Test Set Evaluation

### Metrics
- **Accuracy**: {test_metrics['accuracy']:.4f}
- **F1-Macro**: {test_metrics['f1_macro']:.4f}
- **F1-Micro**: {test_metrics['f1_micro']:.4f}
- **F1-Weighted**: {test_metrics['f1_weighted']:.4f}
- **AUC-ROC**: {test_metrics['auc']:.4f}

### Confusion Matrix
```
{test_metrics['confusion_matrix']}
```

## Model Architecture Diagram

```
Input (Molecular Graph)
    |
    ├─→ Simplicial Complex Builder
    |   ├─→ 0-simplices (nodes)
    |   ├─→ 1-simplices (edges)
    |   └─→ 2-simplices (triangles)
    |
    ├─→ Simplicial Neural Network (SNN)
    |   ├─→ Lower Laplacian Convolution
    |   └─→ Upper Laplacian Convolution
    |
    ├─→ Global GCN Layer
    |
    ├─→ Multi-view Fusion
    |
    ├─→ Robust Fuzzy Inference Engine
    |   ├─→ 10 Fuzzy Rules
    |   ├─→ Membership Functions
    |   └─→ Defuzzification MLP
    |
    └─→ Final MLP Classifier
        └─→ Binary Classification Output
```

## Discussion

### Results Analysis
The SFGE model combines topological information from simplicial complexes with graph neural networks and fuzzy inference to achieve robust molecular toxicity prediction.

### Key Features
1. **Simplicial Complexes**: Capture higher-order topological structures (triangles) in molecular graphs
2. **Multi-scale Processing**: Lower and upper Laplacians capture different scales of connectivity
3. **Fuzzy Inference**: Handles uncertainty and provides interpretable rule-based reasoning
4. **Graph Enhancement**: GCN layer captures global graph structure

### Limitations
1. Computational complexity increases with graph size and number of triangles
2. Requires careful hyperparameter tuning
3. Memory requirements can be high for large molecules

### Future Improvements
1. Implement batching for multiple graphs
2. Add attention mechanisms for better feature aggregation
3. Explore different fuzzy membership functions
4. Implement graph-level pooling strategies
5. Add interpretability tools for fuzzy rules

## Reproducibility

- **Random seed**: 42
- **PyTorch version**: {torch.__version__}
- **CUDA available**: {torch.cuda.is_available()}
- **Device used**: {args.device}

## Files Generated

- Model checkpoint: `outputs/checkpoints/best_model.pt`
- Confusion matrix: `outputs/figures/confusion_matrix.png`
- ROC curve: `outputs/figures/roc_curve.png`
- Training logs: `outputs/logs/`

---
*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\nReport saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Train and evaluate SFGE model on ApisTox dataset')
    parser.add_argument('--data_path', type=str, default='dataset_final.csv',
                       help='Path to dataset CSV file')
    parser.add_argument('--splits_dir', type=str, default='splits',
                       help='Directory containing train/val/test split files')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='Hidden dimension')
    parser.add_argument('--k_snn', type=int, default=2,
                       help='Number of SNN layers')
    parser.add_argument('--d_snn', type=int, default=64,
                       help='SNN hidden dimension')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print("=" * 80)
    print("SFGE Model Training and Evaluation")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    print()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'figures'), exist_ok=True)
    
    # Load dataset
    print("Step 1: Loading dataset...")
    if os.path.exists(args.data_path):
        data_path = args.data_path
    elif os.path.exists(os.path.join('data', args.data_path)):
        data_path = os.path.join('data', args.data_path)
    else:
        raise FileNotFoundError(f"Dataset file not found: {args.data_path}")
    
    graphs = load_dataset_from_csv(data_path)
    print(f"Loaded {len(graphs)} molecular graphs")
    
    # Load splits
    print("\nStep 2: Loading data splits...")
    splits = load_splits(args.splits_dir)
    
    if not splits:
        print("Warning: No split files found. Using random split (80/10/10)...")
        indices = list(range(len(graphs)))
        np.random.shuffle(indices)
        n_train = int(0.8 * len(graphs))
        n_val = int(0.1 * len(graphs))
        splits = {
            'train': indices[:n_train],
            'val': indices[n_train:n_train+n_val],
            'test': indices[n_train+n_val:]
        }
    
    train_graphs, val_graphs, test_graphs = split_dataset(graphs, splits)
    print(f"Train: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}")
    
    # Determine input dimension
    input_dim = graphs[0].x.shape[1]
    print(f"Input feature dimension: {input_dim}")
    
    # Create model
    print("\nStep 3: Creating SFGE model...")
    model = SFGEModel(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=2,
        k_snn=args.k_snn,
        d_snn=args.d_snn,
        num_fuzzy_rules=10,
        dropout=0.1
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print("\nStep 4: Training model...")
    trained_model, history = train_model(
        train_graphs=train_graphs,
        val_graphs=val_graphs,
        model=model,
        device=device,
        num_epochs=args.num_epochs,
        lr=args.lr,
        patience=args.patience,
        batch_size=args.batch_size,
        checkpoint_dir=os.path.join(args.output_dir, 'checkpoints')
    )
    
    # Evaluate on test set
    print("\nStep 5: Evaluating on test set...")
    test_complexes = build_simplicial_complexes(test_graphs, device)
    test_metrics, _, _, _ = evaluate_model(
        trained_model,
        test_graphs,
        test_complexes,
        device,
        output_dir=args.output_dir
    )
    
    print("\nTest Set Results:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  F1-Macro: {test_metrics['f1_macro']:.4f}")
    print(f"  AUC-ROC: {test_metrics['auc']:.4f}")
    
    # Save history
    history_path = os.path.join(args.output_dir, 'logs', 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.items()}, f, indent=2)
    print(f"\nTraining history saved to {history_path}")
    
    # Generate report
    print("\nStep 6: Generating evaluation report...")
    generate_report(history, test_metrics, args, splits, args.output_dir)
    
    print("\n" + "=" * 80)
    print("Training and evaluation completed successfully!")
    print("=" * 80)
    print(f"\nOutputs saved to: {args.output_dir}/")
    print("  - Model checkpoint: checkpoints/best_model.pt")
    print("  - Evaluation report: report.md")
    print("  - Figures: figures/")
    print("  - Training logs: logs/")


if __name__ == '__main__':
    main()

