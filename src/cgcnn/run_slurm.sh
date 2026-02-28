#!/bin/bash
#SBATCH --job-name=opt_tsd_ssd_ws24_cgcnn_raw
#SBATCH --output=slurm_logs/%x_%A.out
#SBATCH --error=slurm_logs/%x_%A.err
#SBATCH --partition=C9654
#SBATCH --nodelist=c3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-gpu=100G
#SBATCH --gres=gpu:1
export PATH=/opt/share/miniconda3/envs/mofmthnn/bin/:$PATH
export LD_LIBRARY_PATH=/opt/share/miniconda3/envs/mofmthnn/lib/:$LD_LIBRARY_PATH

srun python -u main.py --progress_bar --task_cfg ads_qst_ch4_n2 --model_cfg att_cgcnn --batch_size 32 --max_epochs 500 --max_graph_len 200 --atom_fea_len 256 --extra_fea_len 16 --h_fea_len 128 --n_conv 6 --n_h 4 --dropout_prob 0.5 --loss_aggregation fixed_weight_sum --dl_sampler random --task_att_type none --lr 0.001 --lr_mult 10 --optim_config fine --patience 50 --log_dir logs --optuna_name optuna