"""
Combined SHAP Analysis Visualization
This script generates a combined figure with SHAP beeswarm plots for all tasks.
Uses Nature journal publication style.
Author: zhangshd
Date: October 21, 2025
"""
import os
import sys
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

# Add project paths
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = SCRIPT_DIR.parent.parent
sys.path.append(str(ROOT_DIR))
sys.path.append(str(SCRIPT_DIR.parent))

# Set publication-quality plotting style (Nature journal style)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['figure.titlesize'] = 15
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['patch.linewidth'] = 0.5
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['xtick.major.size'] = 4
plt.rcParams['ytick.major.size'] = 4
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.1

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
# Blue (low values) -> White (medium) -> Orange (high values)
def create_nature_cmap():
    """Create a custom colormap using Nature journal colors."""
    colors = [NATURE_COLORS['blue'], '#FFFFFF', NATURE_COLORS['orange']]
    n_bins = 100
    cmap = mcolors.LinearSegmentedColormap.from_list('nature_shap', colors, N=n_bins)
    return cmap

NATURE_CMAP = create_nature_cmap()

# Import functions from shap_analysis
from shap_analysis import load_config, load_trained_model, compute_shap_values


def format_task_name(task_name: str) -> str:
    """
    Format task name for display.
    
    Parameters:
    -----------
    task_name : str
        Original task name
        
    Returns:
    --------
    formatted_name : str
        Formatted task name for display
    """
    name_map = {
        # 'AdsCH4_10kPa': 'CH₄ Ads. (10 kPa)',
        # 'AdsCH4_100kPa': 'CH₄ Ads. (100 kPa)',
        # 'AdsCH4_1000kPa': 'CH₄ Ads. (1000 kPa)',
        # 'AdsN2_10kPa': 'N₂ Ads. (10 kPa)',
        # 'AdsN2_100kPa': 'N₂ Ads. (100 kPa)',
        # 'AdsN2_1000kPa': 'N₂ Ads. (1000 kPa)',
        # 'QstCH4': 'CH₄ Heat of Ads.',
        # 'QstN2': 'N₂ Heat of Ads.',
        # 'PSA_API_CH4': 'PSA API (CH₄)',
        # 'VSA_API_CH4': 'VSA API (CH₄)'
    }
    return name_map.get(task_name, task_name)


def plot_combined_shap_beeswarm(tasks_data: Dict, output_path: str, 
                                max_display: int = 10, figsize: Tuple = None):
    """
    Create a combined figure with SHAP beeswarm plots for multiple tasks.
    
    Parameters:
    -----------
    tasks_data : dict
        Dictionary with task names as keys and tuples of (shap_values, X_explain, feature_names) as values
    output_path : str
        Path to save the combined figure
    max_display : int
        Maximum number of features to display per subplot
    figsize : tuple
        Figure size (width, height). If None, calculated automatically
    """
    n_tasks = len(tasks_data)
    
    # Calculate subplot layout
    if n_tasks <= 2:
        n_rows, n_cols = 1, n_tasks
    elif n_tasks <= 4:
        n_rows, n_cols = 2, 2
    elif n_tasks <= 6:
        n_rows, n_cols = 2, 3
    elif n_tasks <= 8:
        n_rows, n_cols = 2, 4
    elif n_tasks <= 9:
        n_rows, n_cols = 3, 3
    else:
        n_rows, n_cols = 4, 3
    
    # Set figure size
    if figsize is None:
        figsize = (4 * n_cols, 3 * n_rows+1)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Flatten axes array for easier indexing
    if n_tasks == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    # Plot each task
    for idx, (task_name, (shap_values, X_explain, feature_names)) in enumerate(tasks_data.items()):
        ax = axes[idx]
        
        # Create SHAP summary plot on this axis with Nature colormap
        plt.sca(ax)
        shap.summary_plot(
            shap_values, 
            X_explain, 
            feature_names=feature_names,
            show=False, 
            max_display=max_display,
            plot_size=None,  # We control size via figsize
            cmap=NATURE_CMAP,  # Use Nature journal colormap
        )
        
        # Format title with subplot label
        subplot_label = chr(97 + idx)  # a, b, c, ...
        formatted_name = format_task_name(task_name)
        ax.set_title(f'({subplot_label}) {formatted_name}', 
                    fontweight='bold', loc='left', pad=10, )
        
        # Style the axis
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.0)
        ax.spines['bottom'].set_linewidth(1.0)
        ax.tick_params(axis='both', which='major')
        
        # Adjust xlabel
        ax.set_xlabel('SHAP value', fontweight='bold')
        cbar = plt.gcf().axes[-1]  
        cbar.tick_params(labelsize=plt.rcParams['xtick.labelsize']-1)  
        # if hasattr(cbar, 'set_ylabel'):
        cbar.set_ylabel('Feature value', fontsize=plt.rcParams['axes.labelsize'])
    
    # Hide unused subplots
    for idx in range(n_tasks, len(axes)):
        axes[idx].set_visible(False)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    print(f"Combined SHAP beeswarm plot saved to: {output_path}")


def plot_combined_shap_bar(tasks_data: Dict, output_path: str, 
                           max_display: int = 10, figsize: Tuple = None):
    """
    Create a combined figure with SHAP bar plots for multiple tasks.
    
    Parameters:
    -----------
    tasks_data : dict
        Dictionary with task names as keys and tuples of (shap_values, X_explain, feature_names) as values
    output_path : str
        Path to save the combined figure
    max_display : int
        Maximum number of features to display per subplot
    figsize : tuple
        Figure size (width, height). If None, calculated automatically
    """
    n_tasks = len(tasks_data)
    
    # Calculate subplot layout
    if n_tasks <= 2:
        n_rows, n_cols = 1, n_tasks
    elif n_tasks <= 4:
        n_rows, n_cols = 2, 2
    elif n_tasks <= 6:
        n_rows, n_cols = 2, 3
    # elif n_tasks <= 8:
    #     n_rows, n_cols = 2, 4
    elif n_tasks <= 9:
        n_rows, n_cols = 3, 3
    else:
        n_rows, n_cols = 4, 3


    print("n_rows, n_cols:", n_rows, n_cols)
    
    # Set figure size
    if figsize is None:
        figsize = (6 * n_cols, 5 * n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Flatten axes array for easier indexing
    if n_tasks == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    # Plot each task
    for idx, (task_name, (shap_values, X_explain, feature_names)) in enumerate(tasks_data.items()):
        ax = axes[idx]
        
        # Create SHAP summary plot on this axis
        plt.sca(ax)
        shap.summary_plot(
            shap_values, 
            X_explain, 
            feature_names=feature_names,
            plot_type='bar',
            show=False, 
            max_display=max_display,
            plot_size=None,
            color=NATURE_COLORS['blue']
        )
        
        # Format title with subplot label
        subplot_label = chr(97 + idx)  # a, b, c, ...
        formatted_name = format_task_name(task_name)
        ax.set_title(f'({subplot_label}) {formatted_name}', 
                    fontweight='bold', loc='left', pad=10, x=0)
        
        # Style the axis
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.0)
        ax.spines['bottom'].set_linewidth(1.0)
        ax.tick_params(axis='both', which='major', labelsize=9)
        
        # Adjust xlabel
        ax.set_xlabel('mean(|SHAP value|)', fontweight='bold')
    
    # Hide unused subplots
    for idx in range(n_tasks, len(axes)):
        axes[idx].set_visible(False)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    print(f"Combined SHAP bar plot saved to: {output_path}")


def main():
    """Main function to create combined SHAP visualizations."""
    parser = argparse.ArgumentParser(
        description="Generate combined SHAP visualizations for multiple tasks"
    )
    parser.add_argument("--config", type=str, default="configs/ml_model_config.yaml",
                       help="Path to model configuration YAML file")
    parser.add_argument("--output_dir", type=str, default="results/shap_analysis",
                       help="Directory to save combined plots")
    parser.add_argument("--tasks", type=str, nargs="+", default=None,
                       help="Specific tasks to include (default: all tasks in config)")
    parser.add_argument("--max_display", type=int, default=10,
                       help="Maximum number of features to display per subplot")
    parser.add_argument("--plot_type", type=str, choices=['beeswarm', 'bar', 'both'], 
                       default='both', help="Type of plots to generate")
    
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
    
    # Collect SHAP data for all tasks
    tasks_data = {}
    
    for task in tasks_to_analyze:
        print("=" * 80)
        print(f"Processing task: {task}")
        
        # Get model directory from config
        task_key = f"ML-{task}"
        
        if task_key not in config['model_dirs_map']:
            print(f"Warning: Task {task} not found in config. Skipping...")
            continue
        
        model_info = config['model_dirs_map'][task_key]
        model_dir = model_info['Path'][0]
        model_name = model_info.get('Model', 'UnknownModel')
        data_dir = config["data_dir"]
        
        # Check if model directory exists
        if not os.path.exists(model_dir):
            print(f"Warning: Model directory {model_dir} does not exist. Skipping {task}...")
            continue
        
        try:
            # Load model and data
            model, X_train, X_test, feature_names = load_trained_model(
                model_dir, model_name, data_dir
            )
            
            print(f"Model type: {model.__class__.__name__}")
            print(f"Number of training samples: {X_train.shape[0]}")
            print(f"Number of features: {X_train.shape[1]}")
            
            # Compute SHAP values
            if X_test is not None:
                X_explain = X_test
                print(f"Number of test samples: {X_test.shape[0]}")
            else:
                X_explain = X_train
            
            shap_values, explainer = compute_shap_values(
                model, X_train, X_explain, model_type="tree"
            )
            
            # Store data
            tasks_data[task] = (shap_values, X_explain, feature_names)
            print(f"Successfully processed {task}")
            
        except Exception as e:
            print(f"Error processing {task}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not tasks_data:
        print("\nNo tasks were successfully processed. Exiting...")
        return
    
    print("\n" + "=" * 80)
    print(f"Successfully loaded {len(tasks_data)} tasks")
    print("=" * 80)
    
    # Generate combined plots
    if args.plot_type in ['beeswarm', 'both']:
        print("\nGenerating combined SHAP beeswarm plot...")
        beeswarm_output = os.path.join(args.output_dir, "combined_shap_beeswarm.png")
        plot_combined_shap_beeswarm(
            tasks_data, 
            beeswarm_output, 
            max_display=args.max_display
        )
    
    if args.plot_type in ['bar', 'both']:
        print("\nGenerating combined SHAP bar plot...")
        bar_output = os.path.join(args.output_dir, "combined_shap_bar.png")
        plot_combined_shap_bar(
            tasks_data, 
            bar_output, 
            max_display=args.max_display
        )
    
    print("\n" + "=" * 80)
    print("Combined SHAP visualizations completed!")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
