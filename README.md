# CBM-MOF: Screening Metal-Organic Frameworks used for CoalBed Methane Separation 

This project implements machine learning models for predicting properties of Metal-Organic Frameworks (MOFs) using two different architectures:
- **MOFTransformer**: Transformer-based architecture with grid representation
- **CGCNN**: Crystal Graph Convolutional Neural Network

## Project Structure
    
```
src/
├── moftransformer/          # MOFTransformer implementation
│   ├── datamodule/         # Data loading and preprocessing modules
│   ├── module/             # Model implementation and training logic
│   └── ...
├── cgcnn/                  # CGCNN implementation
│   ├── datamodule/         # Data loading and preprocessing modules
│   ├── module/             # Model implementation and training logic
│   └── ...
└── jupyter/                # Jupyter notebooks for analysis

data/
├── processed/              # Processed datasets
└── raw/                   # Raw data files

results/
├── cbm_screening/         # Screening results
└── ...
```

## Models

### MOFTransformer
A transformer-based model that combines:
- Crystal graph representation
- 3D energy grid data
- Attention mechanisms for property prediction

### CGCNN
A crystal graph convolutional neural network that:
- Represents crystals as graphs with atoms as nodes
- Uses bond information as edge features
- Employs graph convolutions for property prediction

## Features

Both models support:
- Multi-task learning for multiple MOF properties
- Classification and regression tasks
- PyTorch Lightning framework for training
- Comprehensive evaluation metrics
- Uncertainty quantification

## Usage

The models can be trained and evaluated using the provided configuration files and command-line interfaces. Each model follows a consistent PyTorch Lightning structure for easy experimentation and comparison.

## Requirements

- Python 3.9+
- PyTorch
- PyTorch Lightning
- Additional dependencies listed in requirements files

## Installation

```bash
pip install -r requirements.txt
```

## Training

Example training command:
```bash
python src/moftransformer/main.py --task_cfg ads_qst_ch4_n2  --load_path /home/zhangsd/repos/CBM-MOF/src/moftransformer/models/pmtransformer.ckpt  --model_name moftransformer  --learning_rate 1e-06  --lr_mult 100  --devices 2  --root_dataset /home/zhangsd/repos/CBM-MOF/src/moftransformer/data/round1/mof_split_val500_test0_seed3  --noise_var 0.0 
python -u src/cgcnn/main.py --task_cfg ads_qst_ch4_n2 --model_cfg att_cgcnn --batch_size 32 --max_epochs 500 --max_graph_len 200 --atom_fea_len 256 --extra_fea_len 16 --h_fea_len 128 --n_conv 6 --n_h 4 --dropout_prob 0.5 --loss_aggregation fixed_weight_sum
```

## Data Preparation

Use the provided Jupyter notebooks in `src/jupyter/` for data preprocessing and analysis.