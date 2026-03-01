'''
Author: zhangshd
Date: 2024-08-09 16:49:54
LastEditors: zhangshd
LastEditTime: 2025-05-10 17:06:59
'''
## This script is adapted from MOFTransformer(https://github.com/hspark1212/MOFTransformer)

import os
from sacred import Experiment
from pathlib import Path

# 确定项目根目录路径
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent

ex = Experiment("cbm-mof", save_git_info=False)


@ex.config
def cfg():
    # Basic Training Control
    num_workers = 2  # Number of worker processes for data loading
    random_seed = 42  # Random seed
    accelerator = "gpu"  # Accelerator type
    devices = 1  # Number of devices
    max_epochs = 1000  # Maximum number of training epochs
    limit_train_batches = None  # Limit on training batches
    limit_val_batches = None  # Limit on validation batches
    auto_lr_bs_find = False  # Auto learning rate and batch size finder flag
    progress_bar = True  # Progress bar display flag

    # Optimizer
    optim = 'Adam'  # Optimizer type
    lr = 1e-3  # Learning rate
    weight_decay = 1e-5  # Weight decay
    momentum = 0.9  # Momentum parameter
    group_lr = True  # Group learning rate flag
    lr_mult = 10  # Learning rate multiplier for multi-task learning heads

    # LR Scheduler
    lr_scheduler = 'polynomial'  # Learning rate scheduler type: multi_step, cosine, reduce_on_plateau
    decay_power = (
        1  # Power of polynomial decay function
                   ) 
    warmup_steps = 2
    

    # Restart Control
    load_best = False  # Load best model flag
    load_dir = None  # Directory to load the model from
    load_ver = None  # Version of the model to load
    load_v_num = None  # Number of the model to load

    # Training Info
    log_dir = os.path.join(ROOT_DIR, 'results/cgcnn_models')  # Log directory
    patience = 50  # Patience
    min_delta = 0.001  # Minimum change
    monitor = 'val/the_metric'  # Monitoring metric
    mode = 'max'  # Mode

    # Data Module Hyperparameters
    radius = 8  # Radius
    dmin = 0  # Minimum distance
    step = 0.2  # Step
    use_cell_params = False  # Use cell parameters flag
    use_extra_fea = False  # Use extra features flag
    task_weights = None
    augment = False  # Data augmentation flag

    # Model Hyperparameters
    model_name = 'att_cgcnn'  # Model name
    atom_fea_len = 128  # Atom feature length
    extra_fea_len = 128  # Extra feature length
    h_fea_len = 256  # Hidden feature length
    n_conv = 3  # Number of convolutional layers
    n_h = 4  # Number of hidden layers
    att_S = 64  # S parameter
    dropout_prob = 0.0  # Dropout probability
    att_pooling = True # Attention pooling flag
    task_norm = True  # Task normalization flag
    dwa_temp = 2.0  # DWA temperature parameter
    dwa_alpha = 0.8  # DWA alpha parameter



@ex.named_config
def cgcnn():
    model_name = 'cgcnn'  # Model name
    atom_fea_len = 64  # Atom feature length
    extra_fea_len = 128  # Extra feature length
    h_fea_len = 128  # Hidden feature length
    n_conv = 3  # Number of convolutional layers
    n_h = 2  # Number of hidden layers
    dropout_prob = 0.0  # Dropout probability
    use_extra_fea = False  # Use extra features flag
    use_cell_params = False  # Use cell parameters flag
    atom_layer_norm = True  # Atom layer normalization flag

@ex.named_config
def cgcnn_raw():
    model_name = 'cgcnn_raw'  # Model name
    atom_fea_len = 64  # Atom feature length
    extra_fea_len = 128  # Extra feature length
    h_fea_len = 128  # Hidden feature length
    n_conv = 3  # Number of convolutional layers
    n_h = 2  # Number of hidden layers
    dropout_prob = 0.0  # Dropout probability
    use_extra_fea = False  # Use extra features flag
    use_cell_params = False  # Use cell parameters flag
    atom_layer_norm = True  # Atom layer normalization flag

@ex.named_config
def att_cgcnn():
    model_name = 'att_cgcnn'  # Model name
    atom_fea_len = 64  # Atom feature length
    extra_fea_len = 128  # Extra feature length
    h_fea_len = 128  # Hidden feature length
    n_conv = 3  # Number of convolutional layers
    n_h = 2  # Number of hidden layers
    dropout_prob = 0.0  # Dropout probability
    use_extra_fea = False  # Use extra features flag
    use_cell_params = False  # Use cell parameters flag
    atom_layer_norm = True  # Atom layer normalization flag
    task_att_type = 'self'  # Attention type: self or external
    att_S = 64  # S parameter of external attention

@ex.named_config
def cgcnn_uni_atom():
    model_name = 'cgcnn_uni_atom'  # Model name
    atom_fea_len = 64  # Atom feature length
    extra_fea_len = 128  # Extra feature length
    max_graph_len = 300  # Maximum number of atoms in a graph
    h_fea_len = 128  # Hidden feature length
    n_conv = 3  # Number of convolutional layers
    n_h = 2  # Number of hidden layers
    dropout_prob = 0.0  # Dropout probability
    use_extra_fea = False  # Use extra features flag
    use_cell_params = False  # Use cell parameters flag
    atom_layer_norm = True  # Atom layer normalization flag
    task_att_type = 'self'  # Attention type: self or external
    att_S = 64  # S parameter of external attention
    reconstruct = False  # Reconstruct atom features into fixed length gragph representation flag


@ex.named_config
def ads_qst_ch4_n2_mini():
    exp_name = "ads_qst_ch4_n2_mini"
    root_dataset = 'src/cgcnn/data/round1/mof_split_val500_test0_seed0'  # Data directory
    root_dataset = str(Path(__file__).parent.parent.parent/root_dataset)
    tasks = {
        'logAdsCH4_10kPa': "regression",
        'logAdsCH4_100kPa': "regression",
        'logAdsCH4_1000kPa': "regression",
        'logAdsN2_10kPa': "regression",
        'logAdsN2_100kPa': "regression",
        'logAdsN2_1000kPa': "regression",
        'QstCH4': "regression",
        'QstN2': "regression",
    }
    max_epochs = 200
    per_gpu_batchsize = 32
    lr = 1e-3
    loss_aggregation = 'fixed_weight_sum'  # Loss aggregation type: sum, trainable_weight_sum, sample_weight_sum, fixed_weight_sum
    task_weights = None

@ex.named_config
def ads_qst_ch4_n2():
    exp_name = "ads_qst_ch4_n2"
    root_dataset = 'src/cgcnn/data/round2'  # Data directory
    root_dataset = str(Path(__file__).parent.parent.parent/root_dataset)
    tasks = {
        'logAdsCH4_10kPa': "regression",
        'logAdsCH4_100kPa': "regression",
        'logAdsCH4_1000kPa': "regression",
        'logAdsN2_10kPa': "regression",
        'logAdsN2_100kPa': "regression",
        'logAdsN2_1000kPa': "regression",
        'QstCH4': "regression",
        'QstN2': "regression",
    }
    max_epochs = 200
    per_gpu_batchsize = 32
    lr = 1e-3
    loss_aggregation = 'fixed_weight_sum'  # Loss aggregation type: sum, trainable_weight_sum, sample_weight_sum, fixed_weight_sum
    task_weights = None

@ex.named_config
def ads_qst_ch4_n2_org():
    exp_name = "ads_qst_ch4_n2_org"
    root_dataset = 'src/cgcnn/data/round2'  # Data directory
    root_dataset = str(Path(__file__).parent.parent.parent/root_dataset)
    tasks = {
        'AdsCH4_10kPa': "regression_log",
        'AdsCH4_100kPa': "regression_log",
        'AdsCH4_1000kPa': "regression_log",
        'AdsN2_10kPa': "regression_log",
        'AdsN2_100kPa': "regression_log",
        'AdsN2_1000kPa': "regression_log",
        'QstCH4': "regression",
        'QstN2': "regression",
    }
    max_epochs = 200
    per_gpu_batchsize = 32
    lr = 1e-3
    loss_aggregation = 'fixed_weight_sum'  # Loss aggregation type: sum, trainable_weight_sum, sample_weight_sum, fixed_weight_sum
    task_weights = None

@ex.named_config
def ads_qst_ch4_n2_symlog_1e3():
    exp_name = "ads_qst_ch4_n2_symlog_1e3"
    root_dataset = 'src/cgcnn/data/round2'  # Data directory
    root_dataset = str(Path(__file__).parent.parent.parent/root_dataset)
    tasks = {
        'symlogAdsCH4_10kPa_1e3':   "regression",
        'symlogAdsCH4_100kPa_1e3':  "regression",
        'symlogAdsCH4_1000kPa_1e3': "regression",
        'symlogAdsN2_10kPa_1e3':    "regression",
        'symlogAdsN2_100kPa_1e3':   "regression",
        'symlogAdsN2_1000kPa_1e3':  "regression",
        'QstCH4': "regression",
        'QstN2':  "regression",
    }
    max_epochs = 500
    per_gpu_batchsize = 32
    lr = 1e-3
    loss_aggregation = 'fixed_weight_sum'
    task_weights = None