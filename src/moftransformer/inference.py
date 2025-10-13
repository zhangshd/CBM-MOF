"""
MOFTransformer Inference Script
This script performs inference using trained MOFTransformer models for CBM-MOF separation predictions.
Author: zhangshd
Date: September 25, 2025
"""
import os
import sys
import argparse
import logging
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Core imports
import torch
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import yaml
import pickle
import functools
import random
import copy
import shutil
from torch.utils.data import DataLoader
from torch.nn.functional import interpolate
from pytorch_lightning import Trainer
from pytorch_lightning.accelerators import find_usable_cuda_devices

# Project imports
from datamodule.prepare_data import make_prepared_data
from datamodule.clean_cif import clean_cif
from datamodule.dataset import Dataset
from module.module import Module
from module.module_utils import get_valid_config
from config import config as _config
from uncertainty import calculate_lsv_from_tree

# Optional imports
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available, uncertainty quantification will be disabled")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model_from_dir(model_dir):
    """
    Load trained MOFTransformer model from directory.
    
    Args:
        model_dir (str or Path): Path to model directory containing hparams.yaml and checkpoints
        
    Returns:
        tuple: (model, trainer) - loaded model and trainer instance
    """
    torch.set_float32_matmul_precision("medium")
    model_dir = Path(model_dir)
    
    # Load hyperparameters
    hparams_file = model_dir / 'hparams.yaml'
    if not hparams_file.exists():
        raise FileNotFoundError(f"hparams.yaml not found in {model_dir}")
        
    with open(hparams_file, 'r') as f:
        hparams = yaml.load(f, Loader=yaml.Loader)
    
    # Update config with loaded hyperparameters
    config = copy.deepcopy(_config())
    config.update(hparams["config"])
    pl.seed_everything(config['seed'])

    # Create trainer
    trainer = Trainer(
        default_root_dir=config["log_dir"], 
        accelerator=config["accelerator"],
        devices=find_usable_cuda_devices(1),
        num_nodes=config["num_nodes"],
        precision=config["precision"],
        benchmark=True,
        max_epochs=1,
        log_every_n_steps=0,
        deterministic=True,
        logger=False,
    )
    
    # Find and load model checkpoint
    checkpoint_dir = model_dir / 'checkpoints'
    if (checkpoint_dir / 'best.ckpt').exists():
        model_file = checkpoint_dir / 'best.ckpt'
    elif (checkpoint_dir / 'val').exists():
        val_checkpoints = list((checkpoint_dir / 'val').glob('*.ckpt'))
        model_file = [f for f in val_checkpoints if 'last' not in f.name]
        if not model_file:
            raise FileNotFoundError(f"No valid checkpoint found in {checkpoint_dir}")
        model_file = model_file[0]
    else:
        raise FileNotFoundError(f"No checkpoint directory found in {model_dir}")
    
    logging.info(f"Loading model from: {model_file}")
    config["load_path"] = model_file
    model = Module(config)
    model.eval()
    
    return model, trainer

def process_cif(cif, saved_dir, clean=True, **kwargs):
    """
    Process CIF file to generate graph and grid data.
    
    Args:
        cif (str or Path): Path to CIF file
        saved_dir (str or Path): Directory to save processed data
        clean (bool): Whether to clean CIF file before processing
        **kwargs: Additional parameters for data preparation
        
    Returns:
        bool: True if successful, None otherwise
    """
    if isinstance(cif, str):
        cif = Path(cif)
    saved_dir = Path(saved_dir)
    
    # Set up logging
    logger = logging.getLogger(f"process_cif_{saved_dir.name}")
    eg_logger = logging.getLogger(f"process_cif_eg_{saved_dir.name}")

    graphdata_dir = saved_dir / "graphs_grids"
    cif_id = cif.stem
    graphdata_dir.mkdir(exist_ok=True, parents=True)
    
    # Define output file paths
    clean_cif_file = graphdata_dir / f"{cif_id}.cif"
    p_graphdata = graphdata_dir / f"{cif_id}.graphdata"
    p_griddata = graphdata_dir / f"{cif_id}.griddata16"
    p_grid = graphdata_dir / f"{cif_id}.grid"
    
    # Clean CIF file if needed
    if not clean_cif_file.exists() and clean:
        try:
            flag = clean_cif(cif, clean_cif_file)
            if not flag:
                logging.warning(f"Failed to clean CIF file: {cif}")
                return None
        except Exception as e:
            logging.error(f"Error cleaning CIF file {cif}: {e}")
            return None
    else:
        try:
            if not clean_cif_file.exists():
                shutil.copy(cif, clean_cif_file)
        except Exception as e:
            logging.error(f"Error copying CIF file {cif}: {e}")
            return None
    
    # Generate graph and grid data if not exists
    if not p_graphdata.exists() or not p_griddata.exists() or not p_grid.exists():
        try:
            flag = make_prepared_data(clean_cif_file, graphdata_dir, logger, eg_logger, **kwargs)
            if not flag:
                logging.warning(f"Failed to prepare data for: {cif}")
                return None
        except Exception as e:
            logging.error(f"Error preparing data for {cif}: {e}")
            return None
    
    return True
    

class InferenceDataset(torch.utils.data.Dataset):
    """
    Dataset class for MOFTransformer inference.
    
    This dataset handles loading and preprocessing of CIF files for inference.
    The model predicts MOF properties (adsorption capacities, heat of adsorption) 
    directly from crystal structure without requiring pressure or composition inputs.
    """
    
    def __init__(self, cif_list, **kwargs):
        """
        Initialize inference dataset.
        
        Args:
            cif_list (list or str): List of CIF file paths or single CIF file path
            **kwargs: Additional dataset parameters
        """
        if isinstance(cif_list, (str, Path)):
            self.cif_list = [Path(cif_list)]
        else:
            self.cif_list = [Path(cif) for cif in cif_list]

        # Dataset parameters
        self.split = "test"
        self.radius = kwargs.get("radius", 8)
        self.max_num_nbr = kwargs.get("max_num_nbr", 12)
        self.dmin = kwargs.get("dmin", 0)
        self.step = kwargs.get("step", 0.2)
        self.task_id = kwargs.get("task_id", 0)
        self.cif_ids = [cif.stem for cif in self.cif_list]
        self.saved_dir = kwargs.get("saved_dir", Path.cwd() / "inference")
        self.clean = kwargs.get("clean", True)
        self.nbr_fea_len = kwargs.get("nbr_fea_len", 64)
        
        logging.info(f"Initialized inference dataset with {len(self.cif_list)} CIF files")

    def setup(self, stage=None):
        """Set up the dataset by processing CIF files and loading data."""
        self.graph_files = {}
        self.grid_files = {}
        self.grid16_files = {}
        
        # Process each CIF file
        for cif in self.cif_list:
            flag = process_cif(cif, self.saved_dir, clean=self.clean, 
                             max_num_nbr=self.max_num_nbr, 
                             radius=self.radius)
            cif_id = cif.stem
            graph_file = self.saved_dir / "graphs_grids" / f"{cif_id}.graphdata"
            grid_file = self.saved_dir / "graphs_grids" / f"{cif_id}.grid"
            grid16_file = self.saved_dir / "graphs_grids" / f"{cif_id}.griddata16"
            
            if flag and graph_file.exists() and grid_file.exists() and grid16_file.exists():
                self.graph_files[cif_id] = graph_file
                self.grid_files[cif_id] = grid_file
                self.grid16_files[cif_id] = grid16_file
            else:
                self.cif_ids = [id for id in self.cif_ids if id != cif_id]
                logging.warning(f"CIF {cif} removed due to data preparation errors")
        
        # Load graph and grid data into memory
        self.graph_data = {}
        self.grid_data = {}
        for cif_id in self.cif_ids:
            self.graph_data[cif_id] = self.get_graph(cif_id)
            self.grid_data[cif_id] = self.get_grid_data(cif_id, False)
        
        logging.info(f"Loaded {len(self.graph_data)} graph data files")
        logging.info(f"Loaded {len(self.grid_data)} grid data files")
        logging.info(f"Dataset size: {len(self.cif_ids)}")
        
    def __len__(self):
        return len(self.cif_ids)
    
    def get_raw_grid_data(self, cif_id):
        file_grid = self.grid_files[cif_id]
        file_griddata = self.grid16_files[cif_id]

        # get grid
        with open(file_grid, "r") as f:
            lines = f.readlines()
            a, b, c = [float(i) for i in lines[0].split()[1:]]
            angle_a, angle_b, angle_c = [float(i) for i in lines[1].split()[1:]]
            cell = [int(i) for i in lines[2].split()[1:]]

        volume = Dataset.calculate_volume(a, b, c, angle_a, angle_b, angle_c)

        # get grid data
        grid_data = pickle.load(open(file_griddata, "rb"))
        grid_data = Dataset.make_grid_data(grid_data)
        grid_data = torch.FloatTensor(grid_data)

        return cell, volume, grid_data

    def get_grid_data(self, cif_id, draw_false_grid=False):
        cell, volume, grid_data = self.get_raw_grid_data(cif_id)
        ret = {
            "cell": cell,
            "volume": volume,
            "grid_data": grid_data,
        }

        if draw_false_grid:
            random_index = random.randint(0, len(self.cif_ids) - 1)
            cif_id = self.cif_ids[random_index]
            cell, volume, grid_data = self.get_raw_grid_data(cif_id)
            ret.update(
                {
                    "false_cell": cell,
                    "fale_volume": volume,
                    "false_grid_data": grid_data,
                }
            )
        return ret

    def get_graph(self, cif_id):
        file_graph = self.graph_files[cif_id]

        graphdata = pickle.load(open(file_graph, "rb"))
        # graphdata = ["cif_id", "atom_num", "nbr_idx", "nbr_dist", "uni_idx", "uni_count"]
        atom_num = torch.LongTensor(graphdata[1].copy())
        nbr_idx = torch.LongTensor(graphdata[2].copy()).view(len(atom_num), -1)
        nbr_dist = torch.FloatTensor(graphdata[3].copy()).view(len(atom_num), -1)

        nbr_fea = torch.FloatTensor(
            Dataset.get_gaussian_distance(nbr_dist, num_step=self.nbr_fea_len, dmax=8)
        )

        uni_idx = graphdata[4]
        uni_count = graphdata[5]
        cell_params = graphdata[6]

        return {
            "atom_num": atom_num,
            "nbr_idx": nbr_idx,
            "nbr_fea": nbr_fea,
            "uni_idx": uni_idx,
            "uni_count": uni_count,
            "cell_params": cell_params
        }

    @functools.lru_cache(maxsize=1024)  # cache loaded structures
    def __getitem__(self, idx):
        """Get item by index."""
        ret = dict()
        cif_id = self.cif_ids[idx]

        ret.update(copy.deepcopy(self.grid_data[cif_id]))
        ret.update(copy.deepcopy(self.graph_data[cif_id]))

        ret.update({
            "cif_id": cif_id,
            "task_id": self.task_id
        })

        return ret
    
    @staticmethod
    def collate(batch, img_size, task_num):
    
        """
        collate batch
        Args:
            batch (dict): [cif_id, atom_num, nbr_idx, nbr_fea, uni_idx, uni_count,
                            grid_data, cell, (false_grid_data, false_cell)]
            img_size (int): maximum length of img size

        Returns:
            dict_batch (dict): [cif_id, atom_num, nbr_idx, nbr_fea, crystal_atom_idx,
                                uni_idx, uni_count, grid, false_grid_data]
        """
        batch_size = len(batch)
        keys = set([key for b in batch for key in b.keys()])

        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

        # graph
        batch_atom_num = dict_batch["atom_num"]
        batch_nbr_idx = dict_batch["nbr_idx"]
        batch_nbr_fea = dict_batch["nbr_fea"]

        crystal_atom_idx = []
        base_idx = 0
        for i, nbr_idx in enumerate(batch_nbr_idx):
            n_i = nbr_idx.shape[0]
            crystal_atom_idx.append(torch.arange(n_i) + base_idx)
            nbr_idx += base_idx
            base_idx += n_i

        dict_batch["atom_num"] = torch.cat(batch_atom_num, dim=0)
        dict_batch["nbr_idx"] = torch.cat(batch_nbr_idx, dim=0)
        dict_batch["nbr_fea"] = torch.cat(batch_nbr_fea, dim=0)
        dict_batch["crystal_atom_idx"] = crystal_atom_idx

        # grid
        batch_grid_data = dict_batch["grid_data"]
        batch_cell = dict_batch["cell"]
        new_grids = []

        for bi in range(batch_size):
            orig = batch_grid_data[bi].view(batch_cell[bi][::-1]).transpose(0, 2)
            if batch_cell[bi] == [30, 30, 30]:  # version >= 1.1.2
                orig = orig[None, None, :, :, :]
            else:
                orig = interpolate(
                    orig[None, None, :, :, :],
                    size=[img_size, img_size, img_size],
                    mode="trilinear",
                    align_corners=True,
                )
            new_grids.append(orig)
        new_grids = torch.concat(new_grids, axis=0)
        dict_batch["grid"] = new_grids

        if "false_grid_data" in dict_batch.keys():
            batch_false_grid_data = dict_batch["false_grid_data"]
            batch_false_cell = dict_batch["false_cell"]
            new_false_grids = []
            for bi in range(batch_size):
                orig = batch_false_grid_data[bi].view(batch_false_cell[bi])
                if batch_cell[bi] == [30, 30, 30]:  # version >= 1.1.2
                    orig = orig[None, None, :, :, :]
                else:
                    orig = interpolate(
                        orig[None, None, :, :, :],
                        size=[img_size, img_size, img_size],
                        mode="trilinear",
                        align_corners=True,
                    )
                new_false_grids.append(orig)
            new_false_grids = torch.concat(new_false_grids, axis=0)
            dict_batch["false_grid"] = new_false_grids

        if "task_id" in dict_batch.keys():
            dict_batch["task_id"] = torch.IntTensor(dict_batch["task_id"])
        dict_batch["target_mask"] = torch.ones(batch_size, task_num, dtype=torch.bool)

        dict_batch.pop("grid_data", None)
        dict_batch.pop("false_grid_data", None)
        dict_batch.pop("cell", None)
        dict_batch.pop("false_cell", None)

        return dict_batch

def load_config_from_dir(model_dir):
    
    model_dir = Path(model_dir)
    with open(model_dir/'hparams.yaml', 'r') as f:
        hparams = yaml.load(f, Loader=yaml.Loader)
    model_file = [file for file in (model_dir / 'checkpoints/val').glob('*.ckpt') if 'last' not in file.name][0]
    print("##")
    print(hparams)
    print("##")
    config = hparams["config"]
    config["load_path"] = model_file
    config = get_valid_config(config)
    # model = Module.load_from_checkpoint(model_file, **hparams)
    return config

def inference(cif_list, model_dir, saved_dir, uncertainty_trees_file=None, **kwargs):
    """
    Perform inference using trained MOFTransformer model.
    
    Args:
        cif_list (list or str): List of CIF file paths or single CIF file path
        model_dir (str or Path): Path to trained model directory
        saved_dir (str or Path): Directory to save inference results
        uncertainty_trees_file (str, optional): Path to uncertainty trees file
        **kwargs: Additional parameters
        
    Returns:
        dict: Dictionary containing predictions for each task
    """
    # Set up model
    model_dir = Path(model_dir)
    saved_dir = Path(saved_dir)
    saved_dir.mkdir(exist_ok=True, parents=True)
    
    logging.info(f"Loading model from: {model_dir}")
    model, trainer = load_model_from_dir(model_dir)
    model.hparams["config"]["noise_var"] = 0.0 # disable noise during inference
    model_name = f"{model_dir.parent.name}_{model_dir.name}"
    
    # Load uncertainty trees if available
    uncertainty_trees = None
    if uncertainty_trees_file and Path(uncertainty_trees_file).exists() and FAISS_AVAILABLE:
        try:
            with open(uncertainty_trees_file, 'rb') as f:
                uncertainty_trees = pickle.load(f)
            for task in uncertainty_trees.keys():
                uncertainty_trees[task]["tree"] = faiss.index_cpu_to_all_gpus(uncertainty_trees[task]["tree"])
            logging.info(f"Loaded uncertainty trees from {uncertainty_trees_file}")
        except Exception as e:
            logging.warning(f"Failed to load uncertainty trees: {e}")
            uncertainty_trees = None
    elif uncertainty_trees_file and not FAISS_AVAILABLE:
        logging.warning("FAISS not available, skipping uncertainty tree loading")
    
    model.eval()
    
    # Set up dataset
    clean = kwargs.get("clean", True)
    try:
        infer_dataset = InferenceDataset(
            cif_list, saved_dir=saved_dir, clean=clean, 
            **model.hparams["config"]
        )
        infer_dataset.setup()
        
        if len(infer_dataset) == 0:
            logging.error("No valid data found for inference")
            return {}
            
    except Exception as e:
        logging.error(f"Error setting up dataset: {e}")
        return {}
    
    # Create data loader
    batch_size = min(len(infer_dataset), model.hparams["config"].get("per_gpu_batchsize", 8))
    infer_loader = DataLoader(
        infer_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=model.hparams["config"].get("num_workers", 2),
        collate_fn=functools.partial(
            InferenceDataset.collate, 
            img_size=model.hparams["config"].get("img_size", 30),
            task_num=len(model.hparams["config"]["tasks"])
        ),
    )
    
    # Perform inference
    logging.info("Running inference...")
    
    try:
        outputs = trainer.predict(model, infer_loader)
        
        # Organize outputs by task
        final_outputs = {}
        for task, task_type in model.hparams["config"]["tasks"].items():
            task_outputs = {}
            task_outputs["Predicted"] = torch.cat([d[f"{task}_pred"] for d in outputs], dim=0).cpu().numpy().squeeze()
            task_outputs["last_layer_fea"] = torch.cat([d[f"{task}_cls_feats"] for d in outputs], dim=0).cpu().numpy().squeeze()
            task_outputs["CifId"] = np.concatenate([d[f"{task}_cif_id"] for d in outputs], axis=0)
            
            if "classification" in task_type:
                task_outputs["PredictedProb"] = torch.cat([d[f"{task}_logits"] for d in outputs], dim=0).cpu().numpy()
            
            # Add uncertainty quantification if available
            if uncertainty_trees and task in uncertainty_trees:
                try:
                    task_outputs["Uncertainty"] = calculate_lsv_from_tree(
                        uncertainty_trees[task], 
                        task_outputs["last_layer_fea"], 
                        k=uncertainty_trees[task]["k"]
                    )
                except Exception as e:
                    logging.warning(f"Error calculating uncertainty for task {task}: {e}")
            
            # Save results to CSV
            df_columns = {k: v for k, v in task_outputs.items() if k != "last_layer_fea"}
            df_res = pd.DataFrame(df_columns)
            df_fea = pd.DataFrame(task_outputs["last_layer_fea"])
            df_fea.columns = [f"last_layer_fea_{i}" for i in range(df_fea.shape[1])]
            df_fea.insert(0, "CifId", df_res["CifId"])
            
            # Reorder columns for better readability
            output_cols = ["CifId", "Predicted"]
            if "Uncertainty" in task_outputs:
                output_cols.append("Uncertainty")
            if "PredictedProb" in task_outputs:
                output_cols.append("PredictedProb")
            
            # Add any remaining columns
            remaining_cols = [col for col in df_res.columns if col not in output_cols]
            output_cols.extend(remaining_cols)
            
            df_res = df_res.reindex(columns=[col for col in output_cols if col in df_res.columns])
            
            output_file = saved_dir / f"{task}_predictions_{model_name}.csv"
            output_fea_file = saved_dir / f"{task}_features_{model_name}.csv.gz"
            df_res.to_csv(output_file, index=False)
            logging.info(f"Saved {task} predictions to: {output_file}")
            df_fea.to_csv(output_fea_file, index=False, compression='gzip')
            logging.info(f"Saved {task} features to: {output_fea_file}")

            final_outputs[task] = task_outputs
        
        logging.info("Inference completed successfully!")
        return final_outputs
        
    except Exception as e:
        logging.error(f"Error during inference: {e}")
        return {}

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='MOFTransformer Inference Script')
    parser.add_argument('--cif_dir', type=str, required=True,
                       help='Directory containing CIF files or path to single CIF file')
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Directory containing trained model')
    parser.add_argument('--output_dir', type=str, default='inference_results',
                       help='Directory to save inference results')
    parser.add_argument('--uncertainty_trees', type=str,
                       help='Path to uncertainty trees file')
    parser.add_argument('--clean', action='store_true', default=False,
                       help='Clean CIF files before processing')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for inference')
    return parser.parse_args()


def main():
    """Main function for command line usage."""
    args = parse_args()
    
    # Set up paths
    cif_path = Path(args.cif_dir)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    
    # Get CIF files
    if cif_path.is_file() and cif_path.suffix == '.cif':
        cif_list = [cif_path]
    elif cif_path.is_dir():
        cif_list = list(cif_path.glob("*.cif"))
        if not cif_list:
            logging.error(f"No CIF files found in {cif_path}")
            return
    else:
        logging.error(f"Invalid CIF path: {cif_path}")
        return
    
    logging.info(f"Found {len(cif_list)} CIF files")
    
    # Validate model directory
    if not model_dir.exists() or not (model_dir / 'hparams.yaml').exists():
        logging.error(f"Invalid model directory: {model_dir}")
        return
    
    # Run inference
    results = inference(
        cif_list=cif_list,
        model_dir=model_dir,
        saved_dir=output_dir,
        uncertainty_trees_file=args.uncertainty_trees,
        clean=args.clean
    )
    
    if results:
        logging.info(f"Inference completed. Results saved to: {output_dir}")
    else:
        logging.error("Inference failed")


if __name__ == "__main__":
    # Example usage when run as script
    if len(sys.argv) > 1:
        main()
    else:
        # Demo usage - adjust paths as needed
        logging.info("Running in demo mode with example data...")
        
        # Project root directory
        project_root = Path(__file__).parent.parent.parent
        
        # Example paths - update these for your specific case
        cif_dir = project_root / "results/cbm_screening/dup_demo_ATC-Cu"  # Update this path
        # model_dir = project_root / "results/moftransformer_models_opt/ads_qst_ch4_n2_seed42_from_pmtransformer/version_15"
        # model_dir = project_root / "results/moftransformer_models/ads_qst_ch4_n2_seed42_moftransformer_from_pmtransformer/version_8"
        model_dir = project_root / "results/moftransformer_models/api_psa_vsa_seed42_moftransformer_from_pmtransformer/version_8"
        result_dir = project_root / "results" / "inference_demo"
        
        # Check if example paths exist
        if not cif_dir.exists():
            logging.warning(f"Example CIF directory not found: {cif_dir}")
            logging.info("Please provide CIF files and model directory as command line arguments")
            logging.info("Usage: python inference.py --cif_dir /path/to/cifs --model_dir /path/to/model --output_dir /path/to/output")
        elif not model_dir.exists():
            logging.warning(f"Example model directory not found: {model_dir}")
            logging.info("Please train a model first or provide a valid model directory")
        else:
            # Get a few example CIF files
            cif_files = list(cif_dir.glob("*.cif"))[:5]  # Limit to 5 files for demo
            
            if cif_files:
                logging.info(f"Running demo inference on {len(cif_files)} CIF files")
                
                # Find uncertainty trees file if it exists
                uncertainty_file = model_dir / "uncertainty_trees.pkl"
                if not uncertainty_file.exists():
                    uncertainty_file = None
                
                results = inference(
                    cif_list=cif_files,
                    model_dir=model_dir,
                    saved_dir=result_dir,
                    uncertainty_trees_file=uncertainty_file,
                    clean=True
                )
                
                if results:
                    logging.info(f"Demo completed. Results saved to: {result_dir}")
                else:
                    logging.error("Demo inference failed")
            else:
                logging.warning(f"No CIF files found in {cif_dir}")
                logging.info("Please add CIF files to the directory or specify a different path")