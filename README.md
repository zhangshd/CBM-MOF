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
├── ml/                     # Machine learning utilities and traditional ML models
│   ├── module.py          # Regression and classification model classes
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

### Traditional Machine Learning Models
The project includes a comprehensive machine learning module (`src/ml/module.py`) that provides:
- Regression models with advanced features
- Classification models with multi-class support
- Feature selection and scaling utilities
- K-fold cross-validation support
- **Target variable transformation** (NEW):
  - Yeo-Johnson transformation for any real values
  - Box-Cox transformation for positive values
  - Automatic inverse transformation for predictions and metrics
  - Improves model performance on skewed distributions

#### Target Transformation Feature
The new target transformation feature allows you to transform the target variable before training to improve model performance:

```python
from src.ml.module import RegressionModel

model = RegressionModel(random_state=42)
model.load_data(train_X, train_y, test_X, test_y)

# Apply Yeo-Johnson transformation to target variable
model.transform_target(method="yeo-johnson", saved_dir="./models")

# Continue with normal workflow
model.scale_feature(feature_range=(0, 1))
model.select_feature(feature_selector='f1', select_des_num=100)
model.kfold_split(k=5, kfold_type="normal")

# Train model - metrics are calculated on original scale
estimator = RandomForestRegressor(n_estimators=100)
model.train(estimator, params={}, saved_dir="./models")
```

**Key Features:**
- **Optional**: Only activated when `transform_target()` is called
- **Automatic**: Predictions and metrics are automatically on original scale
- **Saved**: Transformer object is saved with the model
- **Flexible**: Supports both Yeo-Johnson and Box-Cox methods

## Features

Both models support:
- Multi-task learning for multiple MOF properties
- Classification and regression tasks
- PyTorch Lightning framework for training
- Comprehensive evaluation metrics
- Uncertainty quantification
- Advanced data normalization with PowerTransformerNormalizer

## Data Normalization

The project includes an advanced data normalization system with the new `PowerTransformerNormalizer` that provides superior handling of non-Gaussian distributions:

### PowerTransformerNormalizer Features

- **Advanced Power Transformations**: Supports both Box-Cox (for positive data) and Yeo-Johnson (for any real values) transformations
- **GPU Acceleration**: Pure PyTorch implementation for efficient GPU computation during training and inference
- **Backward Compatibility**: Drop-in replacement for the original Normalizer with identical interface
- **Robust Handling**: Automatically handles NaN values, outliers, and edge cases
- **State Persistence**: Full serialization support for model checkpoints

### Usage Examples

```python
from moftransformer.datamodule.power_transformer import PowerTransformerNormalizer

# Basic usage with Yeo-Johnson transformation (recommended)
normalizer = PowerTransformerNormalizer(method='yeo-johnson')
normalizer.fit(training_data)

# Normalize data for training
normalized_data = normalizer.norm(raw_data)

# Denormalize predictions back to original scale
predictions = normalizer.denorm(model_output)

# For positive-only data, use Box-Cox transformation
box_cox_normalizer = PowerTransformerNormalizer(method='box-cox')

```

### Key Advantages

1. **Better Distribution Handling**: Power transformations can make highly skewed data more Gaussian-like, improving model performance
2. **Numerical Stability**: Advanced numerical techniques prevent overflow/underflow issues
3. **Device Flexibility**: Seamless handling of CPU/GPU data transfers during training
4. **Scientific Accuracy**: Maintains precision for scientific applications with careful handling of edge cases

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

### Traditional ML Inference

Use the machine learning inference pipeline in `src/ml/inference.py` to evaluate the classical models on new structures or precomputed features.

```bash
# Run end-to-end inference from CIF files
python src/ml/inference.py --input_path /path/to/cifs --output_path results.csv

# Skip featurization by providing a precomputed feature table generated by the helper script
python src/ml/inference.py --features_path features.csv --output_path results.csv
```

Key options:
- `--input_path`: Single CIF file or directory; the script cleans structures, generates features, and predicts all seven stability properties.
- `--features_path`: Precomputed feature table matching the schema of `generate_features`; bypasses CIF processing and runs predictions directly.
- `--prob_radius`: Probe radius for Zeo++ feature generation (used only with `--input_path`).

## Data Preparation

Use the provided Jupyter notebooks in `src/jupyter/` for data preprocessing and analysis.