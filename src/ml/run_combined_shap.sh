#!/bin/bash
# Generate combined SHAP visualizations for all trained models
# Author: zhangshd
# Date: October 21, 2025

echo "Generating combined SHAP visualizations..."

# Generate both beeswarm and bar plots
python src/ml/plot_combined_shap.py \
    --config configs/ml_model_config.yaml \
    --output_dir results/shap_analysis \
    --max_display 10 \
    --plot_type both

echo "Combined visualizations generated!"
echo "Check results in: results/shap_analysis/combined_shap_*.png"
