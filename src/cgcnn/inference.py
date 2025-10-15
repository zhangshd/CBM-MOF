'''
Author: zhangshd
Date: 2024-08-19 15:59:37
LastEditors: zhangshd
LastEditTime: 2025-06-10 16:00:09
'''
import os
import sys
from pathlib import Path
SCRIPT_DIR = Path(__file__).parent
# Get the root directory of the project
ROOT_DIR = Path(SCRIPT_DIR).parent.parent
sys.path.append(str(SCRIPT_DIR.parent))
from argparse import ArgumentParser
from pymatgen.io.cif import CifParser
from ase.io import read
from pytorch_lightning.accelerators import find_usable_cuda_devices
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pytorch_lightning import Trainer

import pandas as pd
import logging
import matplotlib
import pickle
import os, sys
import functools
import inspect
from tqdm import tqdm
import shutil
import time
import warnings
import tempfile
from functools import partial

# Ignore specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen.io.cif")
warnings.filterwarnings("ignore", category=UserWarning, module="ase.io.cif")

from cgcnn.module.module import Module
from cgcnn.module.att_cgcnn import CrystalGraphConvNet
from cgcnn.datamodule.data_interface import Datamodule
from cgcnn.datamodule.prepare_data import _make_supercell, get_crystal_graph
from cgcnn.datamodule.dataset import AtomCustomJSONInitializer, GaussianDistance
from cgcnn.utils import load_model_from_dir, MODEL_NAME_TO_DATASET_CLS
from cgcnn.datamodule.prepare_data import make_prepared_data
from cgcnn.datamodule.clean_cif import clean_cif
from cgcnn.module.module_utils import calculate_lse_from_tree, calculate_lsv_from_tree

matplotlib.use('Agg')

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

def process_cif(cif, saved_dir, clean=True, **kwargs):

    if isinstance(cif, str):
        cif = Path(cif)
    saved_dir = Path(saved_dir)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(str(saved_dir/__name__))

    graphdata_dir = saved_dir

    cif_id: str = cif.stem
    graphdata_dir.mkdir(exist_ok=True, parents=True)
    clean_cif_file = graphdata_dir / f"{cif_id}.cif"
    p_graphdata = graphdata_dir / f"{cif_id}.graphdata"
    if not clean_cif_file.exists() and clean:
        flag = clean_cif(cif, clean_cif_file)
        if not flag:
            return None
    elif str(clean_cif_file.resolve()) != str(cif.resolve()):
        shutil.copy(cif, clean_cif_file)
    # else:
    #     logger.info(f"Using existing clean cif file: {clean_cif_file}")
    if not p_graphdata.exists():
        p_graphdata = make_prepared_data(clean_cif_file, graphdata_dir, logger, **kwargs)
    return p_graphdata
    

class InferenceDataset(Dataset):
    def __init__(self, cif_list, **kwargs):
        """
        Args:
            cif_list (list or str): list of cif file paths or a single cif file path.
        """
        if isinstance(cif_list, (str, Path)):
            self.cif_list = [Path(cif_list)]
        else:
            self.cif_list = [Path(cif) for cif in cif_list]

        self.split = "infer"
        self.radius = kwargs.get("radius", 8)
        self.max_num_nbr = kwargs.get("max_num_nbr", 12)
        self.dmin = kwargs.get("dmin", 0)
        self.step = kwargs.get("step", 0.2)
        self.use_cell_params = kwargs.get("use_cell_params", False)
        self.use_extra_fea = kwargs.get("use_extra_fea", False)
        # self.task_id = kwargs.get("task_id", 0)
        self.max_sample_size = kwargs.get("max_sample_size", None)
        self.saved_dir = kwargs.get("saved_dir", Path(os.getcwd())/"inference")
        self.clean = kwargs.get("clean", False)
        self.tasks = kwargs.get("tasks", {})

        self.cif_ids = [cif.stem for cif in self.cif_list]
        self.g_data ={}
        
        atom_prop_json = Path(inspect.getfile(AtomCustomJSONInitializer)).parent/'atom_init.json'
        self.ari = AtomCustomJSONInitializer(atom_prop_json)
        self.gdf = GaussianDistance(dmin=self.dmin, dmax=self.radius, step=self.step)
    
    def append(self, new_data: Dataset):
        if hasattr(self, 'datasets'):
            self.datasets.append(new_data)
        else:
            self.datasets = [self, new_data]
        self.g_data.update(new_data.g_data)

    def setup(self, stage=None):

        for cif in self.cif_list:
            graphdata_file = process_cif(cif, self.saved_dir, clean=self.clean, 
                                         max_num_nbr=self.max_num_nbr, 
                                         radius=self.radius)
            if graphdata_file:
                self.g_data[cif.stem] = graphdata_file
            else:
                self.cif_ids.remove(cif.stem)
                print(f"Error: {cif} has been removed from the dataset due to errors during data preparation.")
        
    def __len__(self):
        return len(self.g_data)

    @functools.lru_cache(maxsize=None)  # cache load strcutrue
    def __getitem__(self, idx):

        cif_id = self.cif_ids[idx]
        # print(cif_id, self.g_data[cif_id])
        with open(self.g_data[cif_id], 'rb') as f:
            data = pickle.load(f)

        cif_id, atom_num, nbr_fea_idx, nbr_dist, *_, cell_params = data
        # assert nbr_fea_idx.shape[0] / atom_num.shape[0] == self.max_num_nbr, f"nbr_fea_idx.shape[0] / atom_num.shape[0]!= 12.0 for file: {self.g_data[cif_id]}"


        extra_fea = torch.FloatTensor([])

        atom_fea = np.vstack([self.ari.get_atom_fea(i) for i in atom_num])
        atom_fea = torch.Tensor(atom_fea)

        nbr_fea_idx = torch.LongTensor(nbr_fea_idx).view(len(atom_num), -1)
        nbr_dist = torch.FloatTensor(nbr_dist).view(len(atom_num), -1)
        nbr_fea = self.gdf.expand(nbr_dist).float()
        assert isinstance(nbr_fea, torch.Tensor)

        if self.use_cell_params:
            cell_params = torch.FloatTensor(cell_params)
            extra_fea = torch.cat([extra_fea, cell_params], dim=-1)

        ret_dict = {
            "atom_fea": atom_fea,
            "nbr_fea": nbr_fea,
            "nbr_fea_idx": nbr_fea_idx,
            "extra_fea": extra_fea,
            "cif_id": cif_id,
            # "task_id": self.task_id
        }

        return ret_dict
    
    @staticmethod
    def collate(batch, tasks={}):
    
        keys = set([key for b in batch for key in b.keys()])
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

        batch_atom_fea = dict_batch["atom_fea"]
        batch_nbr_fea_idx = dict_batch["nbr_fea_idx"]
        batch_nbr_fea = dict_batch["nbr_fea"]
        batch_extra_fea = dict_batch["extra_fea"]

        crystal_atom_idx = []
        base_idx = 0
        for i, nbr_fea_idx in enumerate(batch_nbr_fea_idx):
            n_i = nbr_fea_idx.shape[0]
            crystal_atom_idx.append(torch.arange(n_i) + base_idx)
            nbr_fea_idx += base_idx
            base_idx += n_i

        dict_batch["atom_fea"] = torch.cat(batch_atom_fea, dim=0)
        dict_batch["nbr_fea"] = torch.cat(batch_nbr_fea, dim=0)
        dict_batch["nbr_fea_idx"] = torch.cat(batch_nbr_fea_idx, dim=0)
        dict_batch["extra_fea"] = torch.stack(batch_extra_fea, dim=0)
        dict_batch["crystal_atom_idx"] = crystal_atom_idx
        # dict_batch["task_id"] = torch.IntTensor(dict_batch["task_id"])
        dict_batch["target_mask"] = torch.ones((len(batch), len(tasks)), dtype=torch.bool)
        return dict_batch

def inference(cif_list, model_dir, saved_dir, uncertainty_trees_file=None, logger=None, **kwargs):
    """
    Perform inference on a list of CIF files using a trained model.
    
    Args:    
        cif_list (list or str): List of CIF file paths or a single CIF file path.
        model_dir (str or Path): Directory containing the trained model.
        saved_dir (str or Path): Directory for saving temporary and output files.
        uncertainty_trees_file (str or Path, optional): Path to file containing uncertainty trees.
        logger (logging.Logger, optional): Logger for logging progress and results.
        **kwargs: Additional keyword arguments for inference.
            - clean (bool): Whether to clean the CIF files before inference.
            - batch_size (int): Batch size for inference.
            - num_workers (int): Number of workers for data loading.
    
    Returns:
        dict: Dictionary containing inference results.
    """
    # Set up model
    tick = time.time()
    clean = kwargs.get("clean", True)
    if logger:
        logger.info(f"Loading model from {model_dir}")
    else:
        print(f"Loading model from {model_dir}")
    
    model, trainer = load_model_from_dir(model_dir)
    
    if uncertainty_trees_file is not None and os.path.exists(uncertainty_trees_file):
        with open(uncertainty_trees_file, 'rb') as f:
            uncertainty_trees = pickle.load(f)
        if logger:
            logger.info(f"Loaded uncertainty trees from {uncertainty_trees_file}")
        else:
            print(f"Loaded uncertainty trees from {uncertainty_trees_file}")
    else:
        uncertainty_trees = None

    # Set up dataset
    if logger:
        logger.info(f"Setting up inference dataset with {len(cif_list) if isinstance(cif_list, list) else 1} CIF files")
    else:
        print(f"Setting up inference dataset with {len(cif_list) if isinstance(cif_list, list) else 1} CIF files")
    
    batch_size = kwargs.get("batch_size", model.hparams.get("batch_size", 8))
    num_workers = kwargs.get("num_workers", model.hparams.get("num_workers", 2))
    
    infer_dataset = InferenceDataset(cif_list, saved_dir=saved_dir, clean=clean, **model.hparams)
    infer_dataset.setup()
    
    if len(infer_dataset) == 0:
        error_msg = "No valid CIF files found for inference."
        if logger:
            logger.error(error_msg)
        else:
            print(error_msg)
        return None
    
    if logger:
        logger.info(f"Creating data loader with batch size {batch_size} and {num_workers} workers")
    
    infer_loader = DataLoader(infer_dataset, 
                              batch_size=min(len(infer_dataset), batch_size), 
                              shuffle=False, 
                              num_workers=num_workers,
                              collate_fn=partial(infer_dataset.collate, tasks=model.hparams.get("tasks", {}))   
                              )

    if logger:
        logger.info("Starting model prediction")
    
    outputs = trainer.predict(model, infer_loader)
    
    if logger:
        logger.info("Processing prediction outputs")
    
    all_outputs = {}
    all_outputs["cif_ids"] = [d["cif_id"] for d in infer_dataset]
    
    for task, task_tp in model.hparams.get("tasks", {}).items():
        all_outputs[f"{task}_pred"] = torch.cat([d[f"{task}_pred"] for d in outputs], dim=0).cpu().numpy().squeeze().tolist()
        
        if "classification" in task_tp:
            all_outputs[f"{task}_prob"] = torch.cat([d[f"{task}_prob"] for d in outputs], dim=0).cpu().numpy().tolist()

        if uncertainty_trees is None:
            continue

        if logger:
            logger.info(f"Calculating uncertainty for task {task}")
        
        all_outputs[f"{task}_uncertainty"] = []
        for d in tqdm(outputs, desc=f"Calculating uncertainty for {task}"):
            task_fea = d[f'{task}_last_layer_fea'].cpu().numpy().squeeze()
            if task_fea.ndim == 1:
                task_fea = task_fea.reshape(1, -1)
            if "classification" in task_tp:
                all_outputs[f"{task}_uncertainty"].append(calculate_lse_from_tree(uncertainty_trees[task], task_fea, k=uncertainty_trees[task]["k"]))
            else:
                all_outputs[f"{task}_uncertainty"].append(calculate_lsv_from_tree(uncertainty_trees[task], task_fea, k=uncertainty_trees[task]["k"]))
        all_outputs[f"{task}_uncertainty"] = np.concatenate(all_outputs[f"{task}_uncertainty"], axis=0).tolist()
    
    inference_time = time.time() - tick
    if logger:
        logger.info(f"Inference completed in {time_cost(tick)}")
        if len(infer_dataset) > 0:
            logger.info(f"Average time per CIF: {inference_time / len(infer_dataset):.4f}s")
    
    return all_outputs

def process_cif_directory(dir_path: str, model_dir: str, saved_dir: str, 
                    uncertainty_trees_file: str = None, logger: logging.Logger = None, **kwargs) -> pd.DataFrame:
    """
    Process all CIF files in a directory and make predictions.
    
    Args:
        dir_path: Path to the directory containing CIF files
        model_dir: Path to the directory containing the trained model
        saved_dir: Path to directory for saving temporary and output files
        uncertainty_trees_file: Path to file containing uncertainty trees
        logger: Logger for logging progress and results
        **kwargs: Additional keyword arguments
        
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
        error_msg = f"No CIF files found in directory: {dir_path}"
        if logger:
            logger.error(error_msg)
        else:
            print(error_msg)
        return None
    
    cif_names = [os.path.basename(cif).replace('.cif', '') for cif in cif_files]
    if logger:
        logger.info(f"Found {len(cif_files)} CIF files to process")
    else:
        print(f"Found {len(cif_files)} CIF files to process")
    
    results = inference(cif_files, model_dir, saved_dir, uncertainty_trees_file, logger, **kwargs)
    
    if results is None:
        error_msg = "All CIF files failed to process"
        if logger:
            logger.error(error_msg)
        else:
            print(error_msg)
        return None
    
    # Create DataFrame from results
    df_results = pd.DataFrame({k:v for k,v in results.items() if k != "cif_ids"}, index=results["cif_ids"])
    df_results.index.name = "CifId"
    
    failed_cifs = set(cif_names) - set(results["cif_ids"])
    # Log failed files
    if failed_cifs:
        msg = f"Failed to process {len(failed_cifs)} files: {', '.join(failed_cifs)}"
        if logger:
            logger.warning(msg)
        else:
            print(msg)

    return df_results

def main():
    """
    Main function for CGCNN model inference.
    """
    parser = ArgumentParser(description='MOF CGCNN Model Inference Script')
    parser.add_argument('--input_path', required=True, help='Path to CIF file or directory containing CIF files')
    parser.add_argument('--output_path', required=True, help='Path for the output CSV file')
    parser.add_argument('--model_dir', default=None, 
                        help='Path to the model directory')
    parser.add_argument('--uncertainty_trees_file', 
                        default=None, 
                        help='Path to uncertainty trees file')
    parser.add_argument('--uncertainty', action='store_true', default=False, help='Whether to enable uncertainty estimation')
    parser.add_argument('--temp_dir', type=str, default=None, help='Directory for saving temporary files')
    parser.add_argument('--clean', action="store_true", default=False, help='Whether to clean CIF files before inference')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for data loading')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Start timing
    tick = time.time()
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(tick))}")
    
    # Set up logging
    log_dir = os.path.join(ROOT_DIR, 'logs/cgcnn_inference')
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger('cgcnn_inference', os.path.join(log_dir, 'cgcnn_inference.log'))
    logger.info("Starting CGCNN model inference")
    
    # Process arguments
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    model_dir = Path(args.model_dir)
    if args.temp_dir is None:
        temp_dir = tempfile.mkdtemp()
    else:
        temp_dir = Path(args.temp_dir)
        temp_dir.mkdir(exist_ok=True, parents=True)
    
    logger.info(f"Using model: {model_dir}")
    if args.uncertainty and args.uncertainty_trees_file:
        uncertainty_trees_file = args.uncertainty_trees_file
        logger.info("Uncertainty estimation enabled")
        logger.info(f"Using uncertainty trees: {uncertainty_trees_file}")
    else:
        uncertainty_trees_file = None

    logger.info(f"Saving temporary files to: {temp_dir}")
    
    # Process input
    if input_path.is_file() and input_path.suffix == '.cif':
        logger.info(f"Processing single CIF file: {input_path}")
        results = inference(
            input_path, model_dir, saved_dir=temp_dir, 
            uncertainty_trees_file=uncertainty_trees_file, 
            clean=args.clean, batch_size=args.batch_size, 
            num_workers=args.num_workers, logger=logger
        )
        if results:
            df_results = pd.DataFrame({k:v for k,v in results.items() if k != "cif_ids"}, index=results["cif_ids"])
            df_results.index.name = "MofName"
        else:
            logger.error("Failed to process CIF file")
            logger.info(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}")
            logger.info(f"Time cost: {time_cost(tick)}")
            if isinstance(temp_dir, tempfile.TemporaryDirectory):
                temp_dir.cleanup()
            return 1
    elif input_path.is_dir():
        logger.info(f"Processing directory containing CIF files: {input_path}")
        df_results = process_cif_directory(
            input_path, model_dir, temp_dir, 
            uncertainty_trees_file=uncertainty_trees_file, 
            clean=args.clean, batch_size=args.batch_size, 
            num_workers=args.num_workers, logger=logger
        )
        if df_results is None:
            logger.error("No results to save")
            logger.info(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}")
            logger.info(f"Time cost: {time_cost(tick)}")
            if isinstance(temp_dir, tempfile.TemporaryDirectory):
                temp_dir.cleanup()
            return 1
    else:
        logger.error(f"Invalid input path: {input_path}. Must be a CIF file or directory containing CIF files.")
        logger.info(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}")
        logger.info(f"Time cost: {time_cost(tick)}")
        if isinstance(temp_dir, tempfile.TemporaryDirectory):
            temp_dir.cleanup()
        return 1
    
    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_results.to_csv(output_path, float_format='%.6f')
    logger.info(f"Saved prediction results to {output_path}")
    print(f"Processed {len(df_results)} MOFs. Results saved to {output_path}")
    
    logger.info("CGCNN model inference completed successfully")
    logger.info(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}")
    logger.info(f"Time cost: {time_cost(tick)}")
    logger.info(f"Average time per CIF: {(time.time() - tick) / len(df_results):.4f}s")
    if isinstance(temp_dir, tempfile.TemporaryDirectory):
        temp_dir.cleanup()
    return 0

if __name__ == "__main__":
    sys.exit(main())