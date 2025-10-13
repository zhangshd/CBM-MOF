import os
import glob
import shutil
import subprocess
import numpy as np
import pandas as pd
from molSimplify.Informatics.MOF.MOF_descriptors import get_primitive
import multiprocessing as mp
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent

ZEOPP_PATH = ROOT_DIR.parent/"zeo++-0.3/network"  # Change this path as appropriate
RAC_GETTER_PATH = ROOT_DIR/"src/ml/featuring/RAC_getter.py"

## This script is adapted from the original example_feature_generation.py script provided by Terrones et al. in their published work:
## Terrones G G, Huang S P, Rivera M P, et al. Journal of the American Chemical Society, 2024, 146(29): 20333–20348.

# Use molSimplify version 1.7.3 and pymatgen=2020.10.20
# Use Zeo++-0.3
# This script generates RAC and Zeo++ features for the specified CIFs.
# It then writes the features to a CSV.

### Start of functions ###

def delete_and_remake_folders(folder_names):
    """
    Deletes the folder specified by each item in folder_names if it exists, then remakes it.

    :param folder_names: list of str, the folders to remake.
    :return None:
    
    """
    for folder_name in folder_names:
        if os.path.isdir(folder_name):
            shutil.rmtree(folder_name)
        os.mkdir(folder_name) # A new folder, with nothing in it. 

def descriptor_generator(name, structure_path, wiggle_room, prob_radius):
    """
    descriptor_generator generates RAC and Zeo++ descriptors.

    :param name: str, the name of the MOF being analyzed.
    :param structure_path: str, path to the CIF of the MOF being analyzed.
    :param wiggle_room: float, the wiggle room used in determining the MOF's adjacency matrix.
    :return: None
    """ 

    # Make the required folders for the current MOF.
    current_MOF_folder = f'feature_folders/{name}'
    cif_folder = f'{current_MOF_folder}/cifs'
    RACs_folder = f'{current_MOF_folder}/RACs'
    zeo_folder = f'{current_MOF_folder}/zeo++'
    merged_descriptors_folder = f'{current_MOF_folder}/merged_descriptors'
    delete_and_remake_folders([current_MOF_folder, cif_folder, RACs_folder, zeo_folder, merged_descriptors_folder])

    # Next, running MOF featurization.
    get_primitive_success = True

    try:
        # get_primitive also removes symmetry (enforces P1 symmetry).
        get_primitive(structure_path, f'{cif_folder}/{name}_primitive.cif')
    except ValueError:
        print(f'The primitive cell of {name} could not be found.')
        get_primitive_success = False

    if get_primitive_success:
        structure_path = f'{cif_folder}/{name}_primitive.cif'

    # get_MOF_descriptors is used in RAC_getter.py to get RAC features.
        # The files that are generated from RAC_getter.py: lc_descriptors.csv, sbu_descriptors.csv, linker_descriptors.csv

    # Zeo++ should be installed
    # Change the zeo++-0.3/network path as appropriate
    cmd1 = f'{ZEOPP_PATH} -ha -res {zeo_folder}/{name}_pd.txt {structure_path} > /dev/null 2>&1' # > /dev/null 2>&1 mutes terminal printing
    cmd2 = f'{ZEOPP_PATH} -sa {prob_radius} {prob_radius} 10000 {zeo_folder}/{name}_sa.txt {structure_path} > /dev/null 2>&1'
    cmd3 = f'{ZEOPP_PATH} -volpo {prob_radius} {prob_radius} 10000 {zeo_folder}/{name}_pov.txt {structure_path} > /dev/null 2>&1'
    cmd4 = f'python {RAC_GETTER_PATH} %s %s %s %f' %(structure_path, name, RACs_folder, wiggle_room)

    # four parallelized Zeo++ and RAC commands
    process1 = subprocess.Popen(cmd1, stdout=subprocess.PIPE, stderr=None, shell=True)
    process2 = subprocess.Popen(cmd2, stdout=subprocess.PIPE, stderr=None, shell=True)
    process3 = subprocess.Popen(cmd3, stdout=subprocess.PIPE, stderr=None, shell=True)
    process4 = subprocess.Popen(cmd4, stdout=subprocess.PIPE, stderr=None, shell=True)

    output1 = process1.communicate()[0]
    output2 = process2.communicate()[0]
    output3 = process3.communicate()[0]
    output4 = process4.communicate()[0]

    # Commands above write Zeo++ output to files. Now, code below extracts information from those files.
    # The Zeo++ calculations use a probe radius of 1.4 angstrom, and zeo++ is called by subprocess.

    dict_list = []
    cif_file = name + '.cif'
    largest_included_sphere, largest_free_sphere, largest_included_sphere_along_free_sphere_path  = np.nan, np.nan, np.nan
    unit_cell_volume, crystal_density, VSA, GSA  = np.nan, np.nan, np.nan, np.nan
    VPOV, GPOV = np.nan, np.nan
    POAV, PONAV, GPOAV, GPONAV, POAV_volume_fraction, PONAV_volume_fraction = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    if (os.path.exists(f'{zeo_folder}/{name}_pd.txt') & os.path.exists(f'{zeo_folder}/{name}_sa.txt') &
        os.path.exists(f'{zeo_folder}/{name}_pov.txt')):
        with open(f'{zeo_folder}/{name}_pd.txt') as f:
            pore_diameter_data = f.readlines()
            for row in pore_diameter_data:
                largest_included_sphere = float(row.split()[1]) # largest included sphere
                largest_free_sphere = float(row.split()[2]) # largest free sphere
                largest_included_sphere_along_free_sphere_path = float(row.split()[3]) # largest included sphere along free sphere path
        with open(f'{zeo_folder}/{name}_sa.txt') as f:
            surface_area_data = f.readlines()
            for i, row in enumerate(surface_area_data):
                if i == 0:
                    unit_cell_volume = float(row.split('Unitcell_volume:')[1].split()[0]) # unit cell volume
                    crystal_density = float(row.split('Density:')[1].split()[0]) # crystal density
                    VSA = float(row.split('ASA_m^2/cm^3:')[1].split()[0]) # volumetric surface area
                    GSA = float(row.split('ASA_m^2/g:')[1].split()[0]) # gravimetric surface area
        with open(f'{zeo_folder}/{name}_pov.txt') as f:
            pore_volume_data = f.readlines()
            for i, row in enumerate(pore_volume_data):
                if i == 0:
                    density = float(row.split('Density:')[1].split()[0])
                    POAV = float(row.split('POAV_A^3:')[1].split()[0]) # probe-occupiable accessible volume
                    PONAV = float(row.split('PONAV_A^3:')[1].split()[0]) # probe-occupiable non-accessible volume
                    GPOAV = float(row.split('POAV_cm^3/g:')[1].split()[0])
                    GPONAV = float(row.split('PONAV_cm^3/g:')[1].split()[0])
                    POAV_volume_fraction = float(row.split('POAV_Volume_fraction:')[1].split()[0])
                    PONAV_volume_fraction = float(row.split('PONAV_Volume_fraction:')[1].split()[0])
                    VPOV = POAV_volume_fraction+PONAV_volume_fraction
                    GPOV = VPOV/density
    else:
        print(f'Not all 3 files exist for {name}, so at least one Zeo++ call failed!', 'sa: ', os.path.exists(f'{zeo_folder}/{name}_sa.txt'), 
              '; pd: ', os.path.exists(f'{zeo_folder}/{name}_pd.txt'), '; pov: ', os.path.exists(f'{zeo_folder}/{name}_pov.txt'))
    geo_dict = {'name': name, 'cif_file': cif_file, 'Di': largest_included_sphere, 'Df': largest_free_sphere, 'Dif': largest_included_sphere_along_free_sphere_path,
                'rho': crystal_density, 'VSA': VSA, 'GSA': GSA, 'VPOV': VPOV, 'GPOV': GPOV, 'POAV_vol_frac': POAV_volume_fraction, 
                'PONAV_vol_frac': PONAV_volume_fraction, 'GPOAV': GPOAV, 'GPONAV': GPONAV, 'POAV': POAV, 'PONAV': PONAV}
    dict_list.append(geo_dict)
    geo_df = pd.DataFrame(dict_list)
    geo_df.to_csv(f'{zeo_folder}/geometric_parameters.csv', index=False)

    # error handling for cmd4
    with open(f'{RACs_folder}/RAC_getter_log.txt', 'r') as f:
        if 'FAILED' in f.readline():
            # Do not continue considering this MOF.
            # Check the FailedStructures.log file for this MOF.
            print(f'RAC generation failed for {name}. Continuing to next MOF.')
            return 

    # Merging geometric information with the RAC information that is in the get_MOF_descriptors-generated files (lc_descriptors.csv, sbu_descriptors.csv, linker_descriptors.csv)
    lc_df = pd.read_csv(f"{RACs_folder}/lc_descriptors.csv") 
    sbu_df = pd.read_csv(f"{RACs_folder}/sbu_descriptors.csv")
    linker_df = pd.read_csv(f"{RACs_folder}/linker_descriptors.csv")
    lc_df.drop(columns=['name'], inplace=True)
    sbu_df.drop(columns=['name'], inplace=True)
    linker_df.drop(columns=['name'], inplace=True)  # remove the name column, as it is not needed here.

    lc_df = lc_df.mean().to_frame().transpose() # Averaging over all rows. Convert resulting Series into a DataFrame, then transpose.
    sbu_df = sbu_df.mean().to_frame().transpose()
    linker_df = linker_df.mean().to_frame().transpose()

    merged_df = pd.concat([geo_df, lc_df, sbu_df, linker_df], axis=1)

    merged_df.to_csv(f'{merged_descriptors_folder}/{name}_descriptors.csv', index=False)
### End of functions ###

if __name__ == '__main__':
    # Deleting the previous feature_folders folder, and all folders within it
    import argparse
    parser = argparse.ArgumentParser(description='Generate RAC and Zeo++ features for the specified CIFs.')
    parser.add_argument('--cif_dir', type=str, help='Path to the folder containing the CIFs.')
    parser.add_argument('--wiggle_room', type=float, default=1, help='The wiggle room used in determining the MOF\'s adjacency matrix.')
    parser.add_argument('--prob_radius', type=float, default=0, help='The probe radius used in Zeo++ calculations.')
    parser.add_argument('--n_jobs', type=int, default=2, help='Number of jobs to run in parallel.')

    args = parser.parse_args()

    cif_dir = args.cif_dir
    if not os.path.exists('feature_folders'): 
        os.mkdir('feature_folders')

    cif_paths = glob.glob(f'{cif_dir}/*.cif')
    cif_paths.sort()

    '''
    Generating features for all MOFs.
    '''
    pool = mp.Pool(processes=args.n_jobs)
    print(f"Generating features for all MOFs using {args.n_jobs} cores.")
    for i, cp in enumerate(cif_paths):
        MOF_name = os.path.basename(cp).replace('.cif', '')

        if os.path.exists(f'feature_folders/{MOF_name}'):
            rac_log_file = f'feature_folders/{MOF_name}/RACs/RAC_getter_log.txt'
            if os.path.exists(rac_log_file):
                with open(rac_log_file, 'r') as f:
                    line = f.readline()
                if 'FAILED' in line:
                    # remove the feature_folders/{MOF_name} folder if RAC generation failed.
                    shutil.rmtree(f'feature_folders/{MOF_name}')
                    print(f'RAC generation failed for {MOF_name}. Regenerating features for this MOF.')
                else:
                    print(f'RAC generation already complete for {MOF_name}. Continuing to next MOF.')
                
        print(f'The current MOF is {MOF_name}. Number {i+1} of {len(cif_paths)}.')
        # wiggle_room = 1 # Using the default wiggle_room for all MOFs.
        # descriptor_generator(MOF_name, cp, args.wiggle_room, args.prob_radius)
        pool.apply_async(descriptor_generator, args=(MOF_name, cp, args.wiggle_room, args.prob_radius))
    pool.close()
    pool.join()

    '''
    Collecting MOF features together across all MOFs.
    '''

    # This dictionary will contain the merged descriptors for every MOF that was featurized successfully.
    # This is for quick construction of the final Pandas DataFrame.
    # Keys will be the feature name, and values will be lists of features for each MOF.
    final_df_content_dict = {}

    merged_column_names = ['name', 'cif_file', 'Di', 'Df', 'Dif', 'rho', 'VSA', 'GSA', 'VPOV', 'GPOV', 'POAV_vol_frac', 'PONAV_vol_frac', 'GPOAV', 'GPONAV', 'POAV', 'PONAV', 'D_func-I-0-all', 'D_func-I-1-all', 'D_func-I-2-all', 'D_func-I-3-all', 'D_func-S-0-all', 'D_func-S-1-all', 'D_func-S-2-all', 'D_func-S-3-all', 'D_func-T-0-all', 'D_func-T-1-all', 'D_func-T-2-all', 'D_func-T-3-all', 'D_func-Z-0-all', 'D_func-Z-1-all', 'D_func-Z-2-all', 'D_func-Z-3-all', 'D_func-alpha-0-all', 'D_func-alpha-1-all', 'D_func-alpha-2-all', 'D_func-alpha-3-all', 'D_func-chi-0-all', 'D_func-chi-1-all', 'D_func-chi-2-all', 'D_func-chi-3-all', 'D_lc-I-0-all', 'D_lc-I-1-all', 'D_lc-I-2-all', 'D_lc-I-3-all', 'D_lc-S-0-all', 'D_lc-S-1-all', 'D_lc-S-2-all', 'D_lc-S-3-all', 'D_lc-T-0-all', 'D_lc-T-1-all', 'D_lc-T-2-all', 'D_lc-T-3-all', 'D_lc-Z-0-all', 'D_lc-Z-1-all', 'D_lc-Z-2-all', 'D_lc-Z-3-all', 'D_lc-alpha-0-all', 'D_lc-alpha-1-all', 'D_lc-alpha-2-all', 'D_lc-alpha-3-all', 'D_lc-chi-0-all', 'D_lc-chi-1-all', 'D_lc-chi-2-all', 'D_lc-chi-3-all', 'func-I-0-all', 'func-I-1-all', 'func-I-2-all', 'func-I-3-all', 'func-S-0-all', 'func-S-1-all', 'func-S-2-all', 'func-S-3-all', 'func-T-0-all', 'func-T-1-all', 'func-T-2-all', 'func-T-3-all', 'func-Z-0-all', 'func-Z-1-all', 'func-Z-2-all', 'func-Z-3-all', 'func-alpha-0-all', 'func-alpha-1-all', 'func-alpha-2-all', 'func-alpha-3-all', 'func-chi-0-all', 'func-chi-1-all', 'func-chi-2-all', 'func-chi-3-all', 'lc-I-0-all', 'lc-I-1-all', 'lc-I-2-all', 'lc-I-3-all', 'lc-S-0-all', 'lc-S-1-all', 'lc-S-2-all', 'lc-S-3-all', 'lc-T-0-all', 'lc-T-1-all', 'lc-T-2-all', 'lc-T-3-all', 'lc-Z-0-all', 'lc-Z-1-all', 'lc-Z-2-all', 'lc-Z-3-all', 'lc-alpha-0-all', 'lc-alpha-1-all', 'lc-alpha-2-all', 'lc-alpha-3-all', 'lc-chi-0-all', 'lc-chi-1-all', 'lc-chi-2-all', 'lc-chi-3-all', 'D_mc-I-0-all', 'D_mc-I-1-all', 'D_mc-I-2-all', 'D_mc-I-3-all', 'D_mc-S-0-all', 'D_mc-S-1-all', 'D_mc-S-2-all', 'D_mc-S-3-all', 'D_mc-T-0-all', 'D_mc-T-1-all', 'D_mc-T-2-all', 'D_mc-T-3-all', 'D_mc-Z-0-all', 'D_mc-Z-1-all', 'D_mc-Z-2-all', 'D_mc-Z-3-all', 'D_mc-chi-0-all', 'D_mc-chi-1-all', 'D_mc-chi-2-all', 'D_mc-chi-3-all', 'f-I-0-all', 'f-I-1-all', 'f-I-2-all', 'f-I-3-all', 'f-S-0-all', 'f-S-1-all', 'f-S-2-all', 'f-S-3-all', 'f-T-0-all', 'f-T-1-all', 'f-T-2-all', 'f-T-3-all', 'f-Z-0-all', 'f-Z-1-all', 'f-Z-2-all', 'f-Z-3-all', 'f-chi-0-all', 'f-chi-1-all', 'f-chi-2-all', 'f-chi-3-all', 'mc-I-0-all', 'mc-I-1-all', 'mc-I-2-all', 'mc-I-3-all', 'mc-S-0-all', 'mc-S-1-all', 'mc-S-2-all', 'mc-S-3-all', 'mc-T-0-all', 'mc-T-1-all', 'mc-T-2-all', 'mc-T-3-all', 'mc-Z-0-all', 'mc-Z-1-all', 'mc-Z-2-all', 'mc-Z-3-all', 'mc-chi-0-all', 'mc-chi-1-all', 'mc-chi-2-all', 'mc-chi-3-all', 'f-lig-I-0', 'f-lig-I-1', 'f-lig-I-2', 'f-lig-I-3', 'f-lig-S-0', 'f-lig-S-1', 'f-lig-S-2', 'f-lig-S-3', 'f-lig-T-0', 'f-lig-T-1', 'f-lig-T-2', 'f-lig-T-3', 'f-lig-Z-0', 'f-lig-Z-1', 'f-lig-Z-2', 'f-lig-Z-3', 'f-lig-chi-0', 'f-lig-chi-1', 'f-lig-chi-2', 'f-lig-chi-3']
    for _i in merged_column_names:
        final_df_content_dict[_i] = []

    # Combining all features together for all MOFs.

    MOF_names = [os.path.basename(i).replace('.cif', '') for i in cif_paths]
    unsuccessful_featurizations = []
    for MOF_name in MOF_names: # Iterating through all MOFs.
        descriptors_folder = f'feature_folders/{MOF_name}/merged_descriptors'

        csv_path = f'{descriptors_folder}/{MOF_name}_descriptors.csv'
        if os.path.exists(csv_path): # RAC and Zeo++ feature generation was successful.
            df = pd.read_csv(csv_path)
            
            # Adding to the dictionary
            for _i in merged_column_names: # Iterating through all columns
                value = df.iloc[0][_i]
                final_df_content_dict[_i].append(value)

        else: # RAC and Zeo++ feature generation was *not* successful.
            unsuccessful_featurizations.append(MOF_name)
            continue
    
    final_csv_path = os.path.join(os.path.dirname(cif_dir), 'RAC_and_zeo_features.csv')
    print(f'Writing the final CSV to {final_csv_path}')
    final_df = pd.DataFrame.from_dict(final_df_content_dict)
    final_df = final_df.sort_values(by=['name']) # Sort names alphabetically.
    final_df.to_csv(final_csv_path, index=False)

    print(f'unsuccessful_featurizations is {unsuccessful_featurizations}') 
