# SFGE Model: Simplicial-Fuzzy-Graph-Enhanced Model for ApisTox

A research-grade implementation of the SFGE (Simplicial-Fuzzy-Graph-Enhanced) model for molecular toxicity prediction on the ApisTox dataset. This model combines Topological Data Analysis (TDA), Simplicial Neural Networks (SNN), Graph Neural Networks (GNN), and Neuro-Fuzzy Systems to achieve robust molecular property prediction.

## Table of Contents

- [Project Overview](#project-overview)
- [Theoretical Background](#theoretical-background)
- [Dataset Description](#dataset-description)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Environment Setup](#environment-setup)
- [How to Run](#how-to-run)
- [Expected Outputs](#expected-outputs)
- [Reproducibility](#reproducibility)
- [Hardware Requirements](#hardware-requirements)
- [Citation](#citation)

## Project Overview

The SFGE model is a novel architecture that integrates:

1. **Simplicial Neural Networks (SNN)**: Processes molecular graphs as simplicial complexes, capturing higher-order topological structures (nodes, edges, triangles)
2. **Graph Convolutional Networks (GCN)**: Captures global graph structure and connectivity patterns
3. **Robust Fuzzy Inference Engine**: Provides interpretable rule-based reasoning with 10 fuzzy rules
4. **Multi-view Fusion**: Combines simplicial and graph-based representations

This implementation provides a complete pipeline from molecular SMILES strings to binary toxicity classification.

## Theoretical Background

### Simplicial Complexes

A simplicial complex K of dimension 2 consists of:
- **0-simplices**: Nodes (atoms)
- **1-simplices**: Edges (bonds)
- **2-simplices**: Triangles (3-cycles in the molecular graph)

### Simplicial Laplacians

The model uses combinatorial Laplacians:
- **Lower Laplacian** (L_d): Captures downward connections
- **Upper Laplacian** (L_u): Captures upward connections
- **Full Laplacian**: L = L_d + L_u

### SFGE Architecture

The forward pass follows:

```
X' = P( Σ L_d^p X + Σ L_u^p X + X_H )
```

where:
- P: Aggregation → Selection → Reduction operation
- L_d^p, L_u^p: Powers of lower and upper Laplacians
- X_H: Global GCN enhancement
- Final output: Fuzzy inference → MLP classifier

### Fuzzy Inference

The robust fuzzy engine implements:
- 10 fuzzy rules with Gaussian membership functions
- Normalized firing strength: φ_k = μ_k / Σ μ_j
- Adaptive inference with MLP-based defuzzification

## Dataset Description

The ApisTox dataset contains molecular compounds with:
- **SMILES strings**: Molecular structure representation
- **Binary labels**: Toxic (1) vs Non-toxic (0)
- **Additional features**: CAS numbers, toxicity types, agrochemical categories

**IMPORTANT**: Before running the code, you must place `dataset_final.csv` in the project root directory or in the `data/` directory.

## Project Structure

```
sfge_apistox/
│
├── data/
│   ├── dataset_final.csv          # Dataset file (place here or in root)
│   └── splits/                     # Predefined train/val/test splits (optional)
│       ├── train.txt
│       ├── val.txt
│       └── test.txt
│
├── models/
│   ├── simplicial_layers.py        # Simplicial convolution layers
│   ├── gcn_global_layer.py         # Global GCN layer
│   ├── fuzzy_engine.py             # Robust fuzzy inference engine
│   └── sfge_model.py               # Complete SFGE model
│
├── training/
│   ├── train.py                    # Training script
│   └── evaluation.py               # Evaluation script
│
├── utils/
│   ├── simplicial_utils.py         # Simplicial complex builder
│   ├── graph_builder.py            # SMILES to graph converter
│   └── metrics.py                  # Evaluation metrics
│
├── outputs/
│   ├── logs/                       # Training logs
│   ├── checkpoints/                # Model checkpoints
│   ├── figures/                    # Evaluation plots
│   └── report.md                   # Auto-generated evaluation report
│
├── main.py                         # Main entry point
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended for faster training)
- pip package manager

### Step 1: Clone or Download the Project

Ensure you have the project directory with all the code files.

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: RDKit installation may require additional steps on some systems. If you encounter issues:

```bash
# For conda users (recommended for RDKit)
conda create -n sfge python=3.9
conda activate sfge
conda install -c conda-forge rdkit
pip install -r requirements.txt
```

### Step 3: Prepare the Dataset

**CRITICAL**: Place `dataset_final.csv` in one of the following locations:
- Project root directory: `/path/to/project/dataset_final.csv`
- Data directory: `/path/to/project/data/dataset_final.csv`

The script will automatically search for the dataset in both locations.

### Step 4: Prepare Data Splits (Optional)

If you have predefined train/validation/test splits, place them in `splits/`:
- `splits/train.txt`: One index per line (0-based)
- `splits/val.txt`: One index per line (0-based)
- `splits/test.txt`: One index per line (0-based)

If split files are not provided, the script will automatically create an 80/10/10 random split.

## Environment Setup

### Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import torch; import torch_geometric; import rdkit; print('All packages installed successfully!')"
```

## How to Run

### Basic Usage

```bash
python main.py
```

### Advanced Usage with Custom Parameters

```bash
python main.py \
    --data_path dataset_final.csv \
    --splits_dir splits \
    --num_epochs 100 \
    --lr 0.001 \
    --batch_size 1 \
    --patience 15 \
    --hidden_dim 64 \
    --k_snn 2 \
    --d_snn 64 \
    --device auto \
    --output_dir outputs
```

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_path` | `dataset_final.csv` | Path to dataset CSV file |
| `--splits_dir` | `splits` | Directory containing split files |
| `--num_epochs` | `100` | Number of training epochs |
| `--lr` | `0.001` | Learning rate |
| `--batch_size` | `1` | Batch size (currently supports 1) |
| `--patience` | `15` | Early stopping patience |
| `--hidden_dim` | `64` | Hidden dimension |
| `--k_snn` | `2` | Number of SNN layers |
| `--d_snn` | `64` | SNN hidden dimension |
| `--device` | `auto` | Device (auto/cpu/cuda) |
| `--output_dir` | `outputs` | Output directory |

## Expected Outputs

After running `main.py`, the following outputs will be generated:

### 1. Model Checkpoint
- **Location**: `outputs/checkpoints/best_model.pt`
- **Contains**: Best model state, optimizer state, training history

### 2. Evaluation Report
- **Location**: `outputs/report.md`
- **Contains**: 
  - Experimental setup
  - Hyperparameters
  - Training history
  - Test set metrics
  - Model architecture diagram
  - Discussion and limitations

### 3. Training Logs
- **Location**: `outputs/logs/training_history.json`
- **Contains**: Training and validation metrics per epoch

### 4. Evaluation Figures
- **Location**: `outputs/figures/`
  - `confusion_matrix.png`: Confusion matrix visualization
  - `roc_curve.png`: ROC curve with AUC score

### 5. Console Output
The script prints:
- Dataset loading progress
- Training progress (loss, accuracy, AUC per epoch)
- Test set evaluation metrics
- File save locations

## Reproducibility

The implementation includes several reproducibility features:

1. **Fixed Random Seed**: Set to 42 for all random operations
2. **Deterministic Operations**: PyTorch deterministic mode enabled
3. **Checkpointing**: Best model saved automatically
4. **Logging**: Complete training history saved

To reproduce results:
1. Use the same random seed (42)
2. Use the same data splits
3. Use the same hyperparameters
4. Use the same PyTorch version

## Hardware Requirements

### Minimum Requirements
- **CPU**: 4 cores, 8GB RAM
- **Storage**: 2GB free space
- **GPU**: Optional (CPU training is supported but slower)

### Recommended Requirements
- **CPU**: 8+ cores, 16GB+ RAM
- **GPU**: NVIDIA GPU with CUDA support (8GB+ VRAM)
- **Storage**: 5GB+ free space

### GPU Support

The code automatically detects and uses GPU if available. To force CPU usage:

```bash
python main.py --device cpu
```

## Model Architecture Details

### Input Processing
1. SMILES → Molecular Graph (RDKit)
2. Graph → Simplicial Complex (K=2)
3. Compute incidence matrices (B1, B2)
4. Compute Laplacians (L0, L1_d, L1_u, L2_d)

### Forward Pass
1. **Simplicial Convolution**: Process features with lower/upper Laplacians
2. **GCN Enhancement**: Global graph convolution
3. **Multi-view Fusion**: Combine simplicial and graph features
4. **Fuzzy Inference**: Apply 10 fuzzy rules with defuzzification
5. **Classification**: Final MLP with softmax

### Loss Function
- CrossEntropy Loss for binary classification

### Optimization
- Adam optimizer
- Learning rate: 0.001 (default)
- Early stopping with patience: 15 epochs

## Troubleshooting

### Common Issues

1. **RDKit Import Error**
   ```bash
   # Use conda for RDKit
   conda install -c conda-forge rdkit
   ```

2. **CUDA Out of Memory**
   - Reduce batch size (already at 1)
   - Use CPU: `--device cpu`
   - Reduce hidden dimensions: `--hidden_dim 32`

3. **Dataset Not Found**
   - Ensure `dataset_final.csv` is in project root or `data/` directory
   - Check file path with: `ls -la dataset_final.csv`

4. **Invalid SMILES**
   - The script will skip invalid SMILES with a warning
   - Check console output for skipped molecules

## Citation

If you use this implementation, please cite:

```bibtex
@software{sfge_apistox,
  title={SFGE Model: Simplicial-Fuzzy-Graph-Enhanced Model for ApisTox},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/sfge-apistox}
}
```

## License

[Specify your license here]

## Acknowledgments

- ApisTox dataset providers
- PyTorch Geometric team
- RDKit developers
- Research community in Topological Data Analysis and Graph Neural Networks

## Contact

For questions, issues, or contributions, please open an issue on the project repository.

---

**Last Updated**: 2024

**Version**: 1.0.0

