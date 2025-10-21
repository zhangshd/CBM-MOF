"""
SHAP Feature Importance Analysis for Trained ML Models
This script analyzes feature importance using SHAP (SHapley Additive exPlanations) values
for trained machine learning models.
Author: zhangshd
Date: October 21, 2025
"""
import os, sys
import yaml
import joblib
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import shap
from pathlib import Path
from typing import Dict, List, Tuple

# Get the directory of the script
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# Get the root directory of the project
ROOT_DIR = SCRIPT_DIR.parent.parent
# Add the src directory to the Python path
sys.path.append(str(ROOT_DIR))
sys.path.append(str(SCRIPT_DIR.parent))

# Nature journal color palette
NATURE_COLORS = {
    'blue': '#0173B2',
    'orange': '#DE8F05',
    'green': '#029E73',
    'red': '#CC78BC',
    'cyan': '#56B4E9',
    'magenta': '#CA9161',
    'yellow': '#ECE133',
    'purple': '#949494'
}

# Create Nature-style colormap for SHAP beeswarm plots
def create_nature_cmap():
    """Create a custom colormap using Nature journal colors."""
    colors = [NATURE_COLORS['blue'], '#FFFFFF', NATURE_COLORS['orange']]
    n_bins = 100
    cmap = mcolors.LinearSegmentedColormap.from_list('nature_shap', colors, N=n_bins)
    return cmap

NATURE_CMAP = create_nature_cmap()



def load_config(config_path: str) -> dict:
    """
    Load model configuration from YAML file.
    
    Parameters:
    -----------
    config_path : str
        Path to the configuration YAML file
        
    Returns:
    --------
    config : dict
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_trained_model(model_dir: str, model_name: str, data_dir: str) -> Tuple[object, np.ndarray, np.ndarray, List[str]]:
    """
    Load trained model and associated preprocessing objects.
    
    Parameters:
    -----------
    model_dir : str
        Directory containing the trained model files
        
    Returns:
    --------
    model : object
        Trained model object (full model if available, otherwise the first fold)
    X_train : np.ndarray
        Training features used for SHAP background
    feature_names : list
        List of feature names after selection
    scaler : object
        Feature scaler object
    variance_filter : object
        Variance filter object
    selector : object
        Feature selector object
    """
    print(f"Loading model from: {model_dir}")
    
    # Try to load total model first
    total_model_files = list(Path(model_dir).glob(f"total_model_*_{model_name}*.model"))
    
    if len(total_model_files) > 0:
        # Load the total model (contains everything)
        total_model_file = str(total_model_files[0])
        print(f"Loading total model: {total_model_file}")
        
        with open(total_model_file, 'rb') as f:
            total_model_obj = joblib.load(f)
        
        # Extract model, data and preprocessing objects from total model
        if total_model_obj.full_trained:
            model = total_model_obj.model
        else:
            model = total_model_obj.models[0]  # Use first fold model
        
        X_train_selected = total_model_obj.train_X_selected
        X_test_selected = total_model_obj.test_X_selected if hasattr(total_model_obj, 'test_X_selected') else None
        
        # Get original feature names from CSV
        feature_names = get_feature_names_from_data(data_dir)
        
        # Get selected feature indices
        if hasattr(total_model_obj, 'variance_filter') and hasattr(total_model_obj, 'selector'):
            # Get variance filtered features
            variance_mask = total_model_obj.variance_filter.get_support()
            variance_filtered_names = [name for name, mask in zip(feature_names, variance_mask) if mask]
            
            # Get selected features
            selector_mask = total_model_obj.selector.get_support()
            selected_feature_names = [name for name, mask in zip(variance_filtered_names, selector_mask) if mask]
        else:
            selected_feature_names = feature_names[:X_train_selected.shape[1]]
        
        return model, X_train_selected, X_test_selected, selected_feature_names
    
    else:
        raise FileNotFoundError(f"No total model file found in {model_dir}. "
                              f"Please ensure the model was trained using src/ml/module.py")


def get_feature_names_from_data(data_dir: str) -> List[str]:
    """
    Get feature names from the original data CSV file.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing the input feature files
        
    Returns:
    --------
    feature_names : list
        List of feature names
    """
    # Look for CSV files in the parent directories
    data_file_candidates = [
        "RAC_and_zeo_features.csv",
        "RAC_and_zeo_features_with_id_prop.csv"
    ]

    data_dir = Path(data_dir)
    
    for candidate in data_file_candidates:
        data_file = data_dir / candidate
        if data_file.exists():
            print(f"Loading feature names from: {data_file}")
            df = pd.read_csv(data_file, nrows=1)
            # Remove non-feature columns
            non_feature_cols = ['name', 'MofName', 'cif_file', 'Partition', 'AdsCH4_10kPa', 'AdsCH4_100kPa', 'AdsCH4_1000kPa',
                              'AdsN2_10kPa', 'AdsN2_100kPa', 'AdsN2_1000kPa', 'QstCH4', 'QstN2', 
                              'PSA_WC_CH4', 'PSA_WC_N2', 'PSA_alpha_CH4_N2', 
                              'VSA_WC_CH4', 'VSA_WC_N2', 'VSA_alpha_CH4_N2',
                              'PSA_API_CH4', 'VSA_API_CH4']
            feature_cols = [col for col in df.columns if col not in non_feature_cols]
            print(f"{len(feature_cols)} Feature columns found: {feature_cols}")
            return feature_cols

    raise FileNotFoundError(f"Could not find feature data file in {data_dir}")


def compute_shap_values(model: object, X_background: np.ndarray, X_explain: np.ndarray,
                       model_type: str = "tree") -> shap.Explanation:
    """
    Compute SHAP values for the model.
    
    Parameters:
    -----------
    model : object
        Trained model
    X_background : np.ndarray
        Background dataset for SHAP (typically training data)
    X_explain : np.ndarray
        Dataset to explain (can be same as background or test set)
    model_type : str
        Type of model for SHAP explainer selection
        
    Returns:
    --------
    shap_values : shap.Explanation
        SHAP values for the explained dataset
    """
    print(f"Computing SHAP values using {model_type} explainer...")
    
    # Select appropriate explainer based on model type
    model_name = model.__class__.__name__
    
    if model_type == "tree" or "Forest" in model_name or "XGB" in model_name or "LGBM" in model_name:
        # Tree-based models
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_explain)
    else:
        # Use KernelExplainer for other models (slower but model-agnostic)
        # Sample background data to speed up computation
        if len(X_background) > 100:
            background_sample = shap.sample(X_background, 100, random_state=42)
        else:
            background_sample = X_background
        
        explainer = shap.KernelExplainer(model.predict, background_sample)
        shap_values = explainer.shap_values(X_explain)
    
    print(f"SHAP values computed. Shape: {np.array(shap_values).shape}")
    return shap_values, explainer


def plot_shap_summary(shap_values: np.ndarray, X_explain: np.ndarray, 
                     feature_names: List[str], output_dir: str, 
                     task_name: str, plot_type: str = "bar"):
    """
    Create SHAP summary plots.
    
    Parameters:
    -----------
    shap_values : np.ndarray
        SHAP values
    X_explain : np.ndarray
        Dataset being explained
    feature_names : list
        List of feature names
    output_dir : str
        Directory to save plots
    task_name : str
        Name of the task (for file naming)
    plot_type : str
        Type of summary plot ('bar' or 'dot')
    """
    plt.figure(figsize=(10, 8))
    
    if plot_type == "bar":
        # Bar plot showing mean absolute SHAP values
        shap.summary_plot(shap_values, X_explain, feature_names=feature_names, 
                         plot_type="bar", show=False, max_display=10)
        output_file = os.path.join(output_dir, f"{task_name}_shap_importance_bar.png")
    else:
        # Beeswarm plot showing SHAP value distribution with Nature colormap
        shap.summary_plot(shap_values, X_explain, feature_names=feature_names, 
                         show=False, max_display=10, cmap=NATURE_CMAP)
        output_file = os.path.join(output_dir, f"{task_name}_shap_importance_beeswarm.png")
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved SHAP summary plot: {output_file}")


def save_shap_importance(shap_values: np.ndarray, feature_names: List[str], 
                        output_dir: str, task_name: str):
    """
    Save feature importance scores to CSV.
    
    Parameters:
    -----------
    shap_values : np.ndarray
        SHAP values
    feature_names : list
        List of feature names
    output_dir : str
        Directory to save CSV file
    task_name : str
        Name of the task (for file naming)
    """
    # Calculate mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs_shap,
        'importance_rank': range(1, len(feature_names) + 1)
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('mean_abs_shap', ascending=False)
    importance_df['importance_rank'] = range(1, len(importance_df) + 1)
    
    # Save to CSV
    output_file = os.path.join(output_dir, f"{task_name}_shap_importance.csv")
    importance_df.to_csv(output_file, index=False)
    print(f"Saved feature importance scores: {output_file}")
    
    # Print top 10 features
    print(f"\nTop 10 most important features for {task_name}:")
    print(importance_df.head(10).to_string(index=False))
    print()


def analyze_task(task_name: str, model_dir: str, model_name: str, data_dir: str, output_dir: str):
    """
    Perform SHAP analysis for a single task.
    
    Parameters:
    -----------
    task_name : str
        Name of the task
    model_dir : str
        Directory containing the trained model
    model_name : str
        Name of the model
    data_dir : str
        Directory containing the input feature files
    output_dir : str
        Directory to save analysis results
    """
    print("=" * 80)
    print(f"Analyzing task: {task_name}")
    print("=" * 80)
    
    # Create output directory for this task
    task_output_dir = os.path.join(output_dir, task_name)
    os.makedirs(task_output_dir, exist_ok=True)
    
    # Load model and data
    model, X_train, X_test, feature_names = load_trained_model(model_dir, model_name, data_dir)
    
    print(f"Model type: {model.__class__.__name__}")
    print(f"Number of training samples: {X_train.shape[0]}")
    print(f"Number of features: {X_train.shape[1]}")

    # Compute SHAP values
    if X_test is not None:
        X_explain = X_test
        print(f"Number of test samples: {X_test.shape[0]}")
    else:
        X_explain = X_train
    shap_values, explainer = compute_shap_values(model, X_train, X_explain, model_type="tree")
    # Generate plots
    print("\nGenerating SHAP visualizations...")
    plot_shap_summary(shap_values, X_explain, feature_names, task_output_dir, task_name, plot_type="bar")
    plot_shap_summary(shap_values, X_explain, feature_names, task_output_dir, task_name, plot_type="dot")
    
    # Save importance scores
    save_shap_importance(shap_values, feature_names, task_output_dir, task_name)
    
    print(f"\nAnalysis completed for {task_name}")
    print()


def main():
    """Main function to run SHAP analysis on trained models."""
    parser = argparse.ArgumentParser(description="SHAP Feature Importance Analysis for Trained ML Models")
    parser.add_argument("--config", type=str, default="configs/ml_model_config.yaml",
                       help="Path to model configuration YAML file")
    parser.add_argument("--output_dir", type=str, default="results/shap_analysis",
                       help="Directory to save SHAP analysis results")
    parser.add_argument("--tasks", type=str, nargs="+", default=None,
                       help="Specific tasks to analyze (default: all tasks in config)")
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Get tasks to analyze
    if args.tasks is not None:
        tasks_to_analyze = args.tasks
    else:
        tasks_to_analyze = config.get('tasks', list(config['model_dirs_map'].keys()))
    
    print(f"\nTasks to analyze: {tasks_to_analyze}")
    print(f"Output directory: {args.output_dir}\n")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Analyze each task
    for task in tasks_to_analyze:
        # Get model directory from config
        task_key = f"ML-{task}"
        
        if task_key not in config['model_dirs_map']:
            print(f"Warning: Task {task} not found in config. Skipping...")
            continue
        
        model_info = config['model_dirs_map'][task_key]
        model_dir = model_info['Path'][0]  # Get first path
        model_name = model_info.get('Model', 'UnknownModel')
        data_dir = config["data_dir"]
        
        # Check if model directory exists
        if not os.path.exists(model_dir):
            print(f"Warning: Model directory {model_dir} does not exist. Skipping {task}...")
            continue
        
        # Perform analysis
        analyze_task(task, model_dir, model_name, data_dir, args.output_dir)
    
    print("=" * 80)
    print("SHAP analysis completed for all tasks!")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
