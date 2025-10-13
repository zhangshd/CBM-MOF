#!/usr/bin/env python
"""
MOF Inference script for MOFSNN project.

This script provides functionality to run inference on single CIF files or directories
containing multiple CIF files using pre-trained ML models. It predicts all 7 stability
properties (TSD, SSD, WS24_water, WS24_water4, WS24_acid, WS24_base, WS24_boiling).

Usage:
    python mof_inference.py --input_path /path/to/cif_or_directory --output_path /path/to/output.csv

Author: zhangshd
Date: 2025-05-16
"""

import os
import sys
import yaml
import argparse
import numpy as np
import pandas as pd
import tempfile
import shutil
import subprocess
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import warnings
import time

# Ignore specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen.io.cif")
warnings.filterwarnings("ignore", category=UserWarning, module="ase.io.cif")
warnings.filterwarnings("ignore", category=UserWarning, module="moftransformer.utils.prepare_data")

# Get the directory of the script
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# Get the root directory of the project
ROOT_DIR = SCRIPT_DIR.parent.parent
# Add the src directory to the Python path
sys.path.append(str(ROOT_DIR))
sys.path.append(str(SCRIPT_DIR.parent))

# Define a setup_logger function directly in this script to avoid import issues
import logging
def setup_logger(name, log_file, level=logging.INFO):
    """Set up a logger with file and console handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create handlers
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()
    
    # Create formatters and add them to handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Import project-specific modules
try:
    from cgcnn.datamodule.clean_cif import clean_cif
    from ml.module import ClassificationModel, RegressionModel
except ImportError:
    raise ImportError("Required project modules not found. Make sure the project structure is correct.")

def load_config(config_file: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_file: Path to the configuration file

    Returns:
        Dictionary containing the configuration
    """
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")

    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            return config
    except Exception as e:
        raise RuntimeError(f"Error loading config file {config_file}: {e}")

def get_model_info(config: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """
    Extract model paths from config for all 7 stability tasks.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary mapping task names to model paths
    """
    model_info = {}
    task_list = ["TSD", "SSD", "WS24_water", "WS24_water4", "WS24_acid", "WS24_base", "WS24_boiling"]
    
    for task in task_list:
        baseline_key = f"Baseline-{task}"
        if baseline_key in config["model_dirs_map"]:
            model_config = config["model_dirs_map"][baseline_key]
            path = model_config["Path"]
            # If path is a list, take the first one
            if isinstance(path, list):
                path = path[0]
            model_info[task] = {
                "path": path,
                "model": model_config["Model"],
                "task_type": config["task_types"][task]
            }
            print(f"Model paths for {task}: {model_info[task]}")
    return model_info

def load_ml_models(model_info: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    """
    Load ML models from the specified paths.
    
    Args:
        model_info: Dictionary mapping task names to model information
        (path and model type)
    Returns:
        Dictionary of loaded models, scalers, and feature selectors
    """
    loaded_models = {}
    
    for task, info in model_info.items():
        model_dir = ROOT_DIR/info["path"]
        model_name = info["model"]
        model_files = list(model_dir.glob(f"total_model_*_{model_name}_*.model"))
        
        if not model_files:
            print(f"Warning: No model files found for {task} in {model_dir}.")
            continue
        model_file = model_files[0]  # Use the first matching model file
        print(f"Using model: {model_file}")

        if info["task_type"] == "classification":
            model = ClassificationModel()
            model.load_total_model(model_file)
        elif info["task_type"] == "regression":
            model = RegressionModel()
            model.load_total_model(model_file)
        else:
            print(f"Warning: Unknown task type {info['task_type']} for {task}. Skipping.")
            continue
        
        loaded_models[task] = {
            "model": model,
            "model_name": model_name,
            "task_type": info["task_type"],

        }
    
    return loaded_models

def clean_and_process_cif(cif_path: str, temp_dir: str) -> Tuple[Optional[str], bool]:
    """
    Clean a CIF file and return the path to the cleaned file.
    
    Args:
        cif_path: Path to the input CIF file
        temp_dir: Directory to store temporary files
        
    Returns:
        Tuple of (clean_cif_path, success) where clean_cif_path can be None if processing fails
    """
    try:
        # Define output path
        base_name = os.path.basename(cif_path)
        clean_cif_path = os.path.join(temp_dir, base_name)
        
        # Process the CIF file using clean_cif function
        from cgcnn.datamodule.clean_cif import clean_cif
        
        # Call the function to clean the CIF file and write it to the output path
        result = clean_cif(
            cif_file=cif_path,
            out_file=clean_cif_path,
            log_file=os.path.join(temp_dir, "clean_cif_log.txt"),
            sanitize=True
        )
        
        # Check if cleaning was successful
        if result is not None and os.path.exists(clean_cif_path):
            return clean_cif_path, True
        else:
            print(f"Cleaned CIF file not found at expected location: {clean_cif_path}")
            return None, False
    except Exception as e:
        print(f"Error cleaning CIF {cif_path}: {e}")
        return None, False

def generate_features(cifs: List[str], temp_dir: str, prob_radius: float = 1.32) -> Optional[pd.DataFrame]:
    """
    Generate features for a cleaned CIF file.
    
    Args:
        cifs: List of paths to the cleaned CIF files
        temp_dir: Directory to store temporary files
        prob_radius: Probe radius for Zeo++ calculations
        
    Returns:
        DataFrame containing the generated features, or None if feature generation failed
    """
    try:
        
        # Create a temporary directory containing only the CIF we want to process
        cif_temp_dir = os.path.join(temp_dir, "cif_temp")
        os.makedirs(cif_temp_dir, exist_ok=True)
        for cif_path in cifs:
            # Copy the cleaned CIF file to the temporary directory
            shutil.copy(cif_path, cif_temp_dir)
        
        cmd = f"python {SCRIPT_DIR/'featuring/feature_generation.py'} --cif_dir {cif_temp_dir} --prob_radius {prob_radius}"
        print("Running command:", cmd)
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            env=os.environ.copy(),
            cwd=str(temp_dir)
        )
        
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            print(f"Feature generation failed with error: {stderr.decode()}")
            return None

        # Check if features were generated
        features_file = os.path.join(temp_dir, "RAC_and_zeo_features.csv")
        if not os.path.exists(features_file):
            print(f"Feature file {features_file} not found")
            return None
            
        # Load the features
        df_feat = pd.read_csv(features_file)
        
        # Rename 'name' column to 'MofName' for consistency
        if 'name' in df_feat.columns:
            df_feat.rename(columns={'name': 'MofName'}, inplace=True)
            
        # Drop cif_file column if it exists
        if 'cif_file' in df_feat.columns:
            df_feat.drop(columns=['cif_file'], inplace=True)
            
        return df_feat
    except Exception as e:
        print(f"Error generating features: {e}")
        return None

def predict(features: pd.DataFrame, loaded_models: Dict[str, Any]) -> pd.DataFrame:
    """
    Make predictions for all properties.
    
    Args:
        features: DataFrame containing features
        loaded_models: Dictionary of loaded models and related objects
        
    Returns:
        DataFrame with prediction results
    """
    results = features[['MofName']].copy()
    
    for task, model_dict in loaded_models.items():
        model = model_dict["model"]
        task_type = model_dict["task_type"]
        print(f"Predicting {task} using model: {model_dict['model_name']}")
        # Make prediction
        if  task_type == "classification":
            # Get probabilities for classification tasks
            y_prob = model.predict(features.loc[:, "Di":], return_prob=True)
            
            # Store probabilities in results
            if y_prob.shape[1] == 2:  # Binary classification
                results[f"{task}_prob"] = y_prob.tolist()
            else:  # Multi-class classification
                results[f"{task}_prob"] = y_prob.tolist()
            
            # Get class predictions
            y_pred = model.predict(features.loc[:, "Di":], return_prob=False).squeeze()
            results[task] = y_pred
        else:
            # Regression task or classification model without predict_proba
            y_pred = model.predict(features.loc[:, "Di":]).squeeze()
            results[task] = y_pred
    
    return results

def process_cif_file(cif_path: Union[str, Path, List[Union[str, Path]]], 
                     loaded_models: Dict[str, Any], prob_radius: float = 1.4,
                     batch_size: int = 1000) -> Optional[pd.DataFrame]:
                     
    """
    Process a single CIF file and make predictions.
    
    Args:
        cif_path: Path to the CIF file or list of paths
        loaded_models: Dictionary of loaded models
        prob_radius: Probe radius for Zeo++ calculations
        
    Returns:
        DataFrame with prediction results, or None if processing failed
    """
    if isinstance(cif_path, (str, Path)):
        cif_paths = [str(cif_path)]
    elif isinstance(cif_path, list):
        cif_paths = [str(p) for p in cif_path]
    else:
        raise ValueError("cif_path must be a string, Path, or list of strings/Paths.")
    
    batch_size = min(batch_size, len(cif_paths))
    batches = [cif_paths[i:i + batch_size] for i in range(0, len(cif_paths), batch_size)]
    all_results = []
    for batch in batches:
        with tempfile.TemporaryDirectory() as temp_dir:
            clean_cifs = []
            for cif in batch:
                # Clean the CIF file
                clean_cif_path, success = clean_and_process_cif(cif, temp_dir)
                if not success or clean_cif_path is None:
                    print(f"Failed to clean CIF file: {cif}")
                    continue
                clean_cifs.append(clean_cif_path)

            if not clean_cifs:
                print(f"No valid CIF files found for feature generation.")
                return None
            
            # Generate features for the batch
            features = generate_features(clean_cifs, temp_dir, prob_radius)
            if features is None:
                print(f"Failed to generate features for CIF files: {clean_cifs}")
                continue
            print(f"Generated features: {features.shape}")
            # Make predictions for the batch
            results = predict(features, loaded_models)
            all_results.append(results)
    if not all_results:
        print("No results to process.")
        return None
    results = pd.concat(all_results, ignore_index=True)
    return results

def process_cif_directory(dir_path: str, loaded_models: Dict[str, Any], prob_radius: float = 1.4) -> Optional[pd.DataFrame]:
    """
    Process all CIF files in a directory and make predictions.
    
    Args:
        dir_path: Path to the directory containing CIF files
        loaded_models: Dictionary of loaded models
        prob_radius: Probe radius for Zeo++ calculations
        
    Returns:
        DataFrame with prediction results for all processed CIF files, or None if no files were successfully processed
    """
    # Find all CIF files in the directory
    cif_files = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.cif'):
                cif_files.append(os.path.join(root, file))
    
    if not cif_files:
        print(f"No CIF files found in directory: {dir_path}")
        return None
    cif_names = [os.path.basename(cif).replace('.cif', '') for cif in cif_files]
    print(f"Found {len(cif_files)} CIF files to process")
    results = process_cif_file(cif_files, loaded_models, prob_radius)
    # Process each CIF file
    
    if results is None:
        print("All CIF files failed to process")
        return None
    failed_cifs = set(cif_names) - set(results['MofName'])
    # Log failed files
    if failed_cifs:
        print(f"Failed to process {len(failed_cifs)} files: {', '.join(failed_cifs)}")

    return results

def time_cost(tick: float) -> str:
    """
    Calculate the time cost since the given tick time.
    
    Args:
        tick: Start time in seconds since epoch
        
    Returns:
        Formatted string representing the time cost
    """
    time_cost = time.time() - tick
    hours, remainder = divmod(time_cost, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"

def main():
    parser = argparse.ArgumentParser(description='MOF ML Model Inference Script')
    parser.add_argument('--input_path', required=True, help='Path to CIF file or directory containing CIF files')
    parser.add_argument('--output_path', required=True, help='Path for the output CSV file')
    parser.add_argument('--config_path', default=os.path.join(ROOT_DIR, 'configs/model_comparison_config.yaml'), 
                        help='Path to model configuration file')
    parser.add_argument('--prob_radius', type=float, default=1.4, help='Probe radius for Zeo++ calculations')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    tick = time.time()
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(tick))}")
    # Set up logging
    log_dir = os.path.join(ROOT_DIR, 'logs/ml_inference')
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger('ml_inference', os.path.join(log_dir, 'ml_inference.log'))
    logger.info("Starting ML model inference")
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config_path}")
    config = load_config(args.config_path)
    
    # Get model paths
    model_info = get_model_info(config)
    logger.info(f"Found {len(model_info)} model paths")
    
    # Load models
    logger.info("Loading ML models")
    loaded_models = load_ml_models(model_info)
    logger.info(f"Loaded {len(loaded_models)} models")
    
    # Process input
    input_path = args.input_path
    if os.path.isfile(input_path) and input_path.endswith('.cif'):
        logger.info(f"Processing single CIF file: {input_path}")
        results = process_cif_file(input_path, loaded_models, args.prob_radius)
    elif os.path.isdir(input_path):
        logger.info(f"Processing directory containing CIF files: {input_path}")
        results = process_cif_directory(input_path, loaded_models, args.prob_radius)
    else:
        logger.error(f"Invalid input path: {input_path}. Must be a CIF file or directory containing CIF files.")
        logger.info(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}")
        logger.info(f"Time cost: {time_cost(tick)}")
        return 1
    
    # Save results
    if results is not None:
        output_path = args.output_path
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        results.to_csv(output_path, index=False)
        logger.info(f"Saved prediction results to {output_path}")
        print(f"Processed {len(results)} MOFs. Results saved to {output_path}")
    else:
        logger.error("No results to save")
        print("No results to save")
        logger.info(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}")
        ## print time cost in a human-readable format, e.g., 1h 23m 45s
        logger.info(f"Time cost: {time_cost(tick)}")
        return 1
    
    logger.info("ML model inference completed successfully")
    logger.info(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}")
    logger.info(f"Time cost: {time_cost(tick)}")
    logger.info(f"Average time per CIF: {(time.time() - tick) / len(results):4f}s")
    return 0

if __name__ == "__main__":
    sys.exit(main())
