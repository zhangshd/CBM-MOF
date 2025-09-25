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

## Inference

The project includes a comprehensive inference script for MOFTransformer models that can predict MOF properties for CBM separation applications. The model predicts adsorption capacities and heat of adsorption directly from MOF crystal structure.

### Command Line Usage

```bash
# Basic usage
python src/moftransformer/inference.py \
    --cif_dir /path/to/cif/files \
    --model_dir /path/to/trained/model \
    --output_dir /path/to/output

# Advanced usage with uncertainty quantification
python src/moftransformer/inference.py \
    --cif_dir /path/to/cif/files \
    --model_dir /path/to/trained/model \
    --output_dir /path/to/output \
    --uncertainty_trees /path/to/uncertainty_trees.pkl \
    --batch_size 16
```

### Programmatic Usage

```python
from src.moftransformer.inference import inference
from pathlib import Path

# Define paths
cif_files = ["structure1.cif", "structure2.cif"]
model_dir = "results/moftransformer_models/trained_model"
output_dir = "inference_results"

# Run inference
results = inference(
    cif_list=cif_files,
    model_dir=model_dir,
    saved_dir=output_dir,
    clean=True
)
```

### Inference Parameters

- `--cif_dir`: Directory containing CIF files or path to single CIF file
- `--model_dir`: Directory containing trained MOFTransformer model
- `--output_dir`: Directory to save inference results (default: 'inference_results')
- `--uncertainty_trees`: Path to uncertainty quantification trees file (optional)
- `--clean`: Clean CIF files before processing (default: True)
- `--batch_size`: Batch size for inference (default: 8)

### Model Outputs

The model predicts the following properties directly from MOF structure:
- `logAdsCH4_10kPa`: Log adsorption capacity of CH4 at 10 kPa
- `logAdsCH4_100kPa`: Log adsorption capacity of CH4 at 100 kPa  
- `logAdsCH4_1000kPa`: Log adsorption capacity of CH4 at 1000 kPa
- `logAdsN2_10kPa`: Log adsorption capacity of N2 at 10 kPa
- `logAdsN2_100kPa`: Log adsorption capacity of N2 at 100 kPa
- `logAdsN2_1000kPa`: Log adsorption capacity of N2 at 1000 kPa
- `QstCH4`: Heat of adsorption for CH4
- `QstN2`: Heat of adsorption for N2

### Output

The inference script generates:
- CSV files with predictions for each task
- Uncertainty quantification when available (requires uncertainty trees)
- Organized results by MOF structure

## Data Preparation

Use the provided Jupyter notebooks in `src/jupyter/` for data preprocessing and analysis.