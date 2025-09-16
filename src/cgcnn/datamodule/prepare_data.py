'''
Author: zhangshd
Date: 2024-08-09 16:49:54
LastEditors: zhangshd
LastEditTime: 2025-05-16 21:51:37
'''
## This script is adapted from MOFTransformer(https://github.com/hspark1212/MOFTransformer) and CGCNN(https://github.com/txie-93/cgcnn)

import os
import math
import logging
import logging.handlers
import pickle
from pathlib import Path

import numpy as np

from tqdm import tqdm
from pymatgen.io.cif import CifParser

from ase.io import read
from ase.neighborlist import natural_cutoffs
from ase import neighborlist
from ase.build import make_supercell
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen.io.cif")
import multiprocessing as mp
from functools import partial
import shutil


def get_logger(filename):
    logger = logging.getLogger(filename)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    file_handler = logging.FileHandler(filename)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def get_unique_atoms(atoms):
    # get graph
    cutoff = natural_cutoffs(atoms)
    neighbor_list = neighborlist.NeighborList(
        cutoff, self_interaction=True, bothways=True
    )
    neighbor_list.update(atoms)
    matrix = neighbor_list.get_connectivity_matrix()

    # Get N, N^2
    numbers = atoms.numbers
    number_sqr = np.multiply(numbers, numbers)

    matrix_sqr = matrix.dot(matrix)
    matrix_cub = matrix_sqr.dot(matrix)
    matrix_sqr.data[:] = 1  # count 1 for atoms

    # calculate
    list_n = [numbers, number_sqr]
    list_m = [matrix, matrix_sqr, matrix_cub]

    arr = [numbers]

    for m in list_m:
        for n in list_n:
            arr.append(m.dot(n))

    arr = np.vstack(arr).transpose()

    uni, unique_idx, unique_count = np.unique(
        arr, axis=0, return_index=True, return_counts=True
    )

    # sort
    final_uni = uni[np.argsort(-unique_count)].tolist()
    final_unique_count = unique_count[np.argsort(-unique_count)].tolist()

    arr = arr.tolist()
    final_unique_idx = []
    for u in final_uni:
        final_unique_idx.append([i for i, a in enumerate(arr) if a == u])

    return final_unique_idx, final_unique_count


def get_crystal_graph(atoms, radius=8, max_num_nbr=12):
    dist_mat = atoms.get_all_distances(mic=True)
    nbr_mat = np.where(dist_mat > 0, dist_mat, 1000)  # 1000 is mamium number
    nbr_idx = []
    nbr_dist = []
    for row in nbr_mat:
        idx = np.argsort(row)[:max_num_nbr]
        nbr_idx.extend(idx)
        nbr_dist.extend(row[idx])

    # get same-topo atoms
    uni_idx, uni_count = get_unique_atoms(atoms)

    # convert to small size
    atom_num = np.array(list(atoms.numbers), dtype=np.int8)
    nbr_idx = np.array(nbr_idx, dtype=np.int16)
    nbr_dist = np.array(nbr_dist, dtype=np.float32)
    uni_count = np.array(uni_count, dtype=np.int16)
    return atom_num, nbr_idx, nbr_dist, uni_idx, uni_count

def _make_supercell(atoms, cutoff):
    """
    make atoms into supercell when cell length is less than cufoff (min_length)
    """
    # when the cell lengths are smaller than radius, make supercell to be longer than the radius
    scale_abc = []
    for l in atoms.cell.cellpar()[:3]:
        if l < cutoff:
            scale_abc.append(math.ceil(cutoff / l))
        else:
            scale_abc.append(1)

    # make supercell
    m = np.zeros([3, 3])
    np.fill_diagonal(m, scale_abc)
    atoms = make_supercell(atoms, m)
    return atoms


def make_prepared_data(
    cif: Path, root_dataset_total: Path, logger=None, **kwargs
):
    if logger is None:
        logger = get_logger(filename="prepare_data.log")

    if isinstance(cif, str):
        cif = Path(cif)
    if isinstance(root_dataset_total, str):
        root_dataset_total = Path(root_dataset_total)

    root_dataset_total.mkdir(exist_ok=True, parents=True)

    max_num_nbr = kwargs.get("max_num_nbr", 10)
    max_num_unique_atoms = kwargs.get("max_num_unique_atoms", 300)
    max_num_atoms = kwargs.get("max_num_atoms", None)
    radius = kwargs.get("radius", 8)

    cif_id: str = cif.stem

    p_graphdata = root_dataset_total / f"{cif_id}.graphdata"

    # Grid data and Graph data already exists
    if p_graphdata.exists():
        logger.info(f"{cif_id} graph data already exists")
        return p_graphdata

    # valid cif check
    # try:
    #     CifParser(cif).parse_structures(primitive=True)
    # except ValueError as e:
    #     logger.info(f"{cif_id} failed : {e} (error when reading cif with pymatgen)")
    #     return False

    # read cif by ASE
    try:
        atoms = read(str(cif))
    except Exception as e:
        logger.error(f"{cif_id} failed : {e}")
        return False

    # 1. get crystal graph
    atoms = _make_supercell(atoms, cutoff=radius)  # radius = 8

    if max_num_atoms and len(atoms) > max_num_atoms:
        logger.error(
            f"{cif_id} failed : number of atoms are larger than `max_num_atoms` ({max_num_atoms})"
        )
        return False

    atom_num, nbr_idx, nbr_dist, uni_idx, uni_count = get_crystal_graph(
        atoms, radius=radius, max_num_nbr=max_num_nbr
    )
    cellpars = atoms.cell.cellpar()
    if len(nbr_idx) < len(atom_num) * max_num_nbr:
        logger.error(
            f"{cif_id} failed : num_nbr is smaller than max_num_nbr. please make radius larger"
        )
        return False

    # if len(uni_idx) > max_num_unique_atoms:
    #     logger.error(
    #         f"{cif_id} failed : The number of topologically unique atoms is larget than `max_num_unique_atoms` ({max_num_unique_atoms})"
    #     )
    #     return False
    
    # save graphdata file
    data = [cif_id, atom_num, nbr_idx, nbr_dist, uni_idx, uni_count, cellpars]
    with open(str(p_graphdata), "wb") as f:
        pickle.dump(data, f)
    return p_graphdata

def prepare_data(root_cifs, root_dataset, **kwargs):
    """
    Args:
        root_cifs (str): root for cif files,
                        it should contains "train" and "test" directory in root_cifs
                        ("val" directory is optional)
        root_dataset (str): root for generated datasets

    kwargs:
        - seed : (int) random seed for split data. (default : 42)
        - train_fraction : (float) fraction for train dataset. train_fraction + test_fraction must be smaller than 1 (default : 0.8)
        - test_fraction : (float) fraction for test dataset. train_fraction + test_fraction must be smaller than 1 (default : 0.1)

        - get_primitive (bool) : If True, use primitive cell in graph embedding
        - max_num_unique_atoms (int): max number unique atoms in primitive cells (default: 300)
        - max_num_supercell_atoms (int or None): max number atoms in super cell atoms (default: None)
        - max_length (float) : maximum length of supercell
        - min_length (float) : minimum length of supercell
        - max_num_nbr (int) : maximum number of neighbors when calculating graph
    """

    

    # directory to "Path"
    root_cifs = Path(root_cifs)
    root_dataset = Path(root_dataset)

    # set logger
    logger = get_logger(filename=str(root_dataset.parent/"prepare_data.log"))

    if not root_cifs.exists():
        raise ValueError(f"{root_cifs} does not exists.")

    # make prepare_data in 'total' directory
    root_dataset_total = Path(root_dataset)
    root_dataset_total.mkdir(exist_ok=True, parents=True)

    # make *.grid, *.griddata16, and *.graphdata file
    for cif in tqdm(
        root_cifs.glob("*.cif"), total=sum(1 for _ in root_cifs.glob("*.cif"))
    ):
        make_prepared_data(cif, root_dataset_total, logger, **kwargs)

def main(cif_dir, out_dir=None, radius=8, max_num_nbr=10, n_cpus=32):
    cif_dir = Path(cif_dir)
    if out_dir is None:
        out_dir = cif_dir
    else:
        out_dir = Path(out_dir)
    dataset = cif_dir.parent.name
    logger = get_logger(filename=str(out_dir.parent/f"prepare_data_{dataset}.log"))
    if n_cpus > 1:
        with mp.Pool(processes=n_cpus) as pool:
            pool.map(partial(make_prepared_data, root_dataset_total=out_dir, 
                            radius=radius, max_num_nbr=max_num_nbr, logger=logger), cif_dir.glob("*.cif"))
    for cif_file in tqdm(list(cif_dir.glob("*.cif"))):
        g_file_name = cif_file.stem + ".graphdata"
        if (out_dir/g_file_name).exists():
            continue
        else:
            flag = make_prepared_data(cif_file, out_dir, radius=radius, max_num_nbr=max_num_nbr, logger=logger)
            if not flag:
                print(f"Failed to generate graph data for {cif_file}")
                continue

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cif_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--radius", type=int, default=8)
    parser.add_argument("--max_num_nbr", type=int, default=12)
    parser.add_argument("--n_cpus", type=int, default=1)

    args = parser.parse_args()
    main(**vars(args))