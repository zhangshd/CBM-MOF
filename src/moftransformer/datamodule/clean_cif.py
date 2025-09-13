'''
Author: zhangshd
Date: 2024-08-09 16:49:54
LastEditors: zhangshd
LastEditTime: 2024-08-27 17:23:25
'''

## This script is adapted from MOFTransformer(https://github.com/hspark1212/MOFTransformer) and CGCNN(https://github.com/txie-93/cgcnn)

# This script removes overlapping atoms
# and floating (unbound) solvent from CIFs.

from molSimplify.Informatics.MOF.PBC_functions import solvent_removal, overlap_removal
from pathlib import Path
import multiprocessing as mp
import argparse
from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from ase import io
import os


def standardized_cif(cif_file, out_file, primitive=False, spacegroup=False):
    cif_file = str(cif_file)
    print("*"*50)
    print(cif_file)

    # preprocessed by ase
    atoms = io.read(cif_file)
    atoms.write(out_file, format='cif')

    parser = CifParser(out_file)
    struct = parser.parse_structures(primitive=False)[0]
    if not spacegroup:
        struct.to(out_file, fmt='cif')
        return out_file
    # Create SpacegroupAnalyzer object
    spacegroup_analyzer = SpacegroupAnalyzer(struct)
    if primitive:
        standardized_structure = spacegroup_analyzer.get_primitive_standard_structure()
        standardized_structure = standardized_structure.get_primitive_structure()
    else:
        # Get space group symbol
        space_group_symbol = spacegroup_analyzer.get_space_group_symbol()
        # print("Space Group Symbol:", space_group_symbol)
        # Get space group number
        space_group_number = spacegroup_analyzer.get_space_group_number()
        # print("Space Group Number:", space_group_number)
        # Get space group operations
        space_group_operations = spacegroup_analyzer.get_space_group_operations()
        # print("Space Group Operations:", space_group_operations)
        # Get standardized crystal structure (according to International Union of Crystallography standard)
        standardized_structure = spacegroup_analyzer.get_refined_structure()
        # print("Standardized Structure:", standardized_structure)
    standardized_structure.to(out_file, fmt='cif')
    print("standardized structure saved to: ", out_file)
    return out_file

def log_error(log_file, message, log_lock=None):
    """Helper function to log errors with optional lock to avoid race conditions."""
    if log_file is not None:
        if log_lock is not None:
            with log_lock:
                with open(log_file, 'a') as f:
                    f.write(message + '\n')
        else:
            with open(log_file, 'a') as f:
                f.write(message + '\n')

def clean_cif(cif_file, out_file, log_file=None, sanitize=True, log_lock=None, **kwargs):
    """Clean a CIF file by standardizing it, removing overlapping atoms and floating solvent."""
    current_cif = str(cif_file)
    out_file = str(out_file)
    if sanitize:
        try:
            standardized_cif(current_cif, out_file, **kwargs)
            current_cif = out_file
        except Exception as e:
            error_message = f"Error in standardizing {current_cif}: {e}"
            print(error_message)
            log_error(log_file, f'{current_cif} failed in standardized_cif', log_lock)
            return None
    
    try:
        overlap_removal(current_cif, out_file)
    except Exception as e:
        error_message = f"Error in removing overlapping atoms from {current_cif}: {e}"
        print(error_message)
        log_error(log_file, f'{current_cif} failed in overlap_removal', log_lock)
        return None
    
    try:
        solvent_removal(out_file, out_file)
    except Exception as e:
        error_message = f"Error in removing floating solvent from {out_file}: {e}"
        print(error_message)
        log_error(log_file, f'{out_file} failed in solvent_removal', log_lock)
        return None
    
    return out_file

def main(cif_dir, output_dir, log_file=None, sanitize=True, n_cpus=None, **kwargs):
    cif_dir = Path(cif_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    if n_cpus is None:
        n_cpus = min(int(mp.cpu_count()*0.8), len(os.listdir(cif_dir)))

    # Create a Lock for logging to avoid conflicts
    log_lock = mp.Lock()
    
    with mp.Pool(processes=n_cpus) as pool:
        results = []
        for cif_path in cif_dir.glob('*.cif'):
            output_path = str(output_dir / cif_path.name)
            cif_path = str(cif_path)
            result = pool.apply_async(
                clean_cif, 
                args=(cif_path, output_path, log_file, sanitize), 
                kwds={'log_lock': log_lock, **kwargs}
            )
            results.append(result)
        
        # Ensure all processes complete
        for result in results:
            result.wait()

    # process_dir = cif_dir
    # if sanitize:
    #     i = 0
    #     # pool = mp.Pool(processes=n_cpus)
    #     print("Start cleaning CIFs..")
    #     for cif_path in Path(process_dir).glob('*.cif'):
    #         output_path = str(output_dir / cif_path.name)
    #         cif_path = str(cif_path)
    #         try:
    #             # pool.apply_async(standardized_cif, args=(cif_path, output_path))
    #             standardized_cif(cif_path, output_path)
    #             print(f"cif cleaned: {cif_path} -> {output_path}")
    #         except Exception as e:
    #             print("-"*50)
    #             print(f'{cif_path} failed in standardized_cif:\n{e}')
    #             if log_file is not None:
    #                 with open(log_file, 'a') as f:
    #                     f.write(f'{cif_path} failed in standardized_cif\n')
    #         if i % 100 == 0:
    #             print(f'{i} CIFs standardized')
    #         i += 1
    #     # pool.close()
    #     # pool.join()
    #     process_dir = output_dir
    #     print("#"*100)

    # # Remove overlapping atoms and floating solvent
    # i = 0
    # # pool = mp.Pool(processes=n_cpus)
    # print("Start removing overlapping atoms and floating solvent..")
    # for cif_path in Path(process_dir).glob('*.cif'):
    #     output_path = str(output_dir / cif_path.name)
    #     cif_path = str(cif_path)
    #     try:
    #         # pool.apply_async(overlap_removal, args=(cif_path, output_path)) # Input CIF should have P1 symmetry.
    #         overlap_removal(cif_path, output_path) # Input CIF should have P1 symmetry.
    #     except Exception as e:
    #         print("-"*50)
    #         print(f'{cif_path} failed in overlap_removal:\n{e}')
    #         if log_file is not None:
    #             with open(log_file, 'a') as f:
    #                 f.write(f'{cif_path} failed in overlap_removal\n')
    #     if i % 100 == 0:
    #         print(f'{i} CIFs removed overlapping atoms')
    #     i += 1
    # # pool.close()
    # # pool.join()
    # process_dir = output_dir
    # print("#"*100)

    # # Remove floating solvent
    # # pool = mp.Pool(processes=n_cpus)
    # i = 0
    # print("Start removing floating solvent..")
    # for cif_path in Path(process_dir).glob('*.cif'):
    #     output_path = str(output_dir / cif_path.name)
    #     cif_path = str(cif_path)
    #     try:
    #         # pool.apply_async(solvent_removal, args=(cif_path, cif_path)) # Output CIF will have floating solvent removed.
    #         solvent_removal(cif_path, output_path)
    #     except Exception as e:
    #         print("-"*50)
    #         print(f'{cif_path} failed in solvent_removal:\n{e}')
    #         if log_file is not None:
    #             with open(log_file, 'a') as f:
    #                 f.write(f'{cif_path} failed in solvent_removal\n')
    #     if i % 100 == 0:
    #         print(f'{i} CIFs removed floating solvent')
    #     i += 1
    # # pool.close()
    # # pool.join()
    # print("#"*100)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cif_dir', type=str, help='Directory containing CIFs to clean')
    parser.add_argument('--output_dir', type=str, help='Directory to save cleaned CIFs')
    parser.add_argument('--sanitize', type=bool, default=True, help='Whether to standardize CIFs')
    parser.add_argument('--log_file', type=str, default="cleaning_log.txt", help='File to log errors')
    parser.add_argument('--n_cpus', type=int, default=None, help='Number of parallel processes')
    args = parser.parse_args()
    main(**vars(args))