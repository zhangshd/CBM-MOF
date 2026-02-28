#!/bin/bash
#SBATCH --job-name=ml_train_round2_VSA_API_CH4_XGB
#SBATCH --output=slurm_logs/%A_%x.out
#SBATCH --error=slurm_logs/%A_%x.err
#SBATCH --partition=C9654 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
export PATH=/opt/share/miniconda3/envs/mofmthnn/bin/:$PATH
export LD_LIBRARY_PATH=/opt/share/miniconda3/envs/mofmthnn/lib/:$LD_LIBRARY_PATH

srun python -u main.py --model_type regression --model_list XGB  --search_max_evals 100 --search_metric val_MAE  --label_column VSA_API_CH4 --name_column MofName  --feature_selector_list RFE f1 mutual_info  --data_dir /home/zhangsd/repos/CBM-MOF/src/ml/data/round2 --in_file_name RAC_and_zeo_features_with_id_prop.csv --target_transform_method log10
