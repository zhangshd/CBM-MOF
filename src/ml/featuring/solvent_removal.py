'''
Author: zhangshd
Date: 2024-08-17 19:43:39
LastEditors: zhangshd
LastEditTime: 2024-08-17 19:43:40
'''
# This script removes overlapping atoms
# and floating (unbound) solvent from CIFs.

from molSimplify.Informatics.MOF.PBC_functions import solvent_removal, overlap_removal
from pathlib import Path
import multiprocessing as mp
import argparse


def main(cif_dir, output_dir, log_file=None, **kwargs):
    # Remove overlapping atoms and floating solvent
    cif_dir = Path(cif_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    # n_jobs = 96
    # pool = mp.Pool(processes=n_jobs)
    i = 0
    for cif_path in Path(cif_dir).glob('*.cif'):
        output_path = str(output_dir / cif_path.name)
        cif_path = str(cif_path)
        try:
            # pool.apply_async(overlap_removal, args=(cif_path, output_path)) # Input CIF should have P1 symmetry.
            overlap_removal(cif_path, output_path) # Input CIF should have P1 symmetry.
        except Exception as e:
            print("-"*50)
            print(f'{cif_path} failed in overlap_removal:\n{e}')
            if log_file is not None:
                with open(log_file, 'a') as f:
                    f.write(f'{cif_path} failed in overlap_removal\n')
        if i % 100 == 0:
            print(f'{i} CIFs removed overlapping atoms')
        i += 1
    # pool.close()
    # pool.join()

    # Remove floating solvent
    # pool = mp.Pool(processes=n_jobs)
    i = 0
    for cif_path in Path(output_dir).glob('*.cif'):
        cif_path = str(cif_path)
        try:
            # pool.apply_async(solvent_removal, args=(cif_path, cif_path)) # Output CIF will have floating solvent removed.
            solvent_removal(cif_path, cif_path)
        except Exception as e:
            print("-"*50)
            print(f'{cif_path} failed in solvent_removal:\n{e}')
            if log_file is not None:
                with open(log_file, 'a') as f:
                    f.write(f'{cif_path} failed in solvent_removal\n')
        if i % 100 == 0:
            print(f'{i} CIFs removed floating solvent')
        i += 1
    # pool.close()
    # pool.join()

# Example usage
# cleaned_path = 'cleaned_example_2.cif'

# # Remove overlapping atoms and floating solvent
# overlap_removal('example_2.cif', cleaned_path) # Input CIF should have P1 symmetry.
# solvent_removal(cleaned_path, cleaned_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cif_dir', type=str, help='Directory containing CIFs to clean')
    parser.add_argument('--output_dir', type=str, help='Directory to save cleaned CIFs')
    parser.add_argument('--log_file', type=str, default="cleaning_log.txt", help='File to log errors')
    args = parser.parse_args()
    main(**vars(args))