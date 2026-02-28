#!/bin/bash
# Run SHAP feature importance analysis for all trained models
# Author: zhangshd
# Date: October 21, 2025

echo "Starting SHAP Feature Importance Analysis..."

# Run analysis for all tasks in the config
python src/ml/shap_analysis.py \
    --config configs/ml_model_config.yaml \
    --output_dir results/shap_analysis

echo "Analysis completed!"
