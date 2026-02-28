#!/bin/bash
#SBATCH --job-name=inference_4
#SBATCH --output=slurm_logs/%x_%A.out
#SBATCH --error=slurm_logs/%x_%A.err
#SBATCH --partition=C9654 
#SBATCH --nodelist=c3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-gpu=100G
#SBATCH --gres=gpu:1
export PATH=/opt/share/miniconda3/envs/mofnn/bin/:$PATH
export LD_LIBRARY_PATH=/opt/share/miniconda3/envs/mofnn/lib/:$LD_LIBRARY_PATH

srun python -u /home/zhangsd/repos/CBM-MOF/src/cgcnn/inference.py --input_path /home/zhangsd/repos/CBM-MOF/results/cbm_screening/inference_cgcnn/batch_4/graphs_grids --model_dir /home/zhangsd/repos/CBM-MOF/results/cgcnn_models/ads_qst_ch4_n2_org_seed42_att_cgcnn/version_1 --output_path /home/zhangsd/repos/CBM-MOF/results/cbm_screening/inference_cgcnn/batch_4/infer_results_cgcnn.csv --temp_dir /home/zhangsd/repos/CBM-MOF/results/cbm_screening/inference_cgcnn/batch_4/graphs_grids --batch_size 32