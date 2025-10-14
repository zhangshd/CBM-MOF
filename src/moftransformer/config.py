# MOFTransformer version 2.1.3
import os
from sacred import Experiment
from moftransformer import __root_dir__
from moftransformer.utils.validation import _set_load_path
from pathlib import Path

ex = Experiment("pretrained_mof", save_git_info=False)

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent

@ex.config
def config():
    """
    # prepare_data
    max_num_atoms = 300
    min_length = 30
    max_length = 60
    radius = 8
    max_nbr_atoms = 12
    """

    # model
    exp_name = "pretrained_mof"
    model_name = "moftransformer" # 
    seed = 42
    noise_var = 0
    limit_train_batches = None

    # graph seeting
    #max_supercell_atoms = None  # number of maximum atoms in supercell atoms
    atom_fea_len = 64
    nbr_fea_len = 64
    max_graph_len = 300  # number of maximum nodes in graph
    max_nbr_atoms = 12

    # grid setting
    img_size = 30
    patch_size = 5  # length of patch
    in_chans = 1  # channels of grid image
    max_grid_len = -1  # when -1, max_image_len is set to maximum ph*pw of batch images
    draw_false_grid = False

    # transformer setting
    hid_dim = 768
    num_heads = 12
    num_layers = 12
    mlp_ratio = 4
    drop_rate = 0.1
    mpp_ratio = 0.15

    # downstream
    # downstream = ""
    # n_classes = 0

    # Optimizer Setting
    optim_type = "adamw"  # adamw, adam, sgd (momentum=0.9)
    learning_rate = 1e-6
    weight_decay = 1e-2
    decay_power = (
        1  # default polynomial decay, [cosine, constant, constant_with_warmup]
    )
    max_epochs = 20
    max_steps = -1  # num_data * max_epoch // batch_size (accumulate_grad_batches)
    warmup_steps = 0.05  # int or float ( max_steps * warmup_steps)
    end_lr = 0
    lr_mult = 100  # multiply lr for downstream heads

    # PL Trainer Setting
    resume_from = None
    val_check_interval = 1.0
    test_only = False
    progress_bar = False

    # callbacks
    patience = 50
    min_delta = 0.001

    # below params varies with the environment
    root_dataset = os.path.join(__root_dir__, "examples/dataset")
    log_dir = os.path.join(ROOT_DIR, 'results/moftransformer_models')  # Log directory
    # batch_size = 64  # desired batch size; for gradient accumulation
    per_gpu_batchsize = 8  # you should define this manually with per_gpu_batch_size
    accelerator = "auto"
    devices = "auto"
    num_nodes = 1

    # load_path = _set_load_path('models/pmtransformer.ckpt')
    load_path = None

    num_workers = 2  # the number of cpu's core
    precision = "16-mixed"  #  "32-true", "16-mixed"

    # visualize
    visualize = False  # return attention map


@ex.named_config
def test():
    exp_name = "test"
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
    root_dataset = "src/moftransformer/data/round1/mof_split_val10_test0_seed0"
    root_dataset = str(Path(__file__).parent.parent.parent/root_dataset)
    max_epochs = 2
    # batch_size = 16
    per_gpu_batchsize = 16

@ex.named_config
def ads_qst_ch4_n2_mini():
    exp_name = "ads_qst_ch4_n2_mini"
    root_dataset = 'src/moftransformer/data/round1/mof_split_val500_test0_seed0'  # Data directory
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

@ex.named_config
def ads_qst_ch4_n2():
    exp_name = "ads_qst_ch4_n2"
    root_dataset = 'src/moftransformer/data/round2'  # Data directory
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

@ex.named_config
def api_psa_vsa():
    exp_name = "api_psa_vsa"
    root_dataset = 'src/moftransformer/data/round2'  # Data directory
    root_dataset = str(Path(__file__).parent.parent.parent/root_dataset)
    tasks = {
        'PSA_API_CH4': "regression_mae",
        'VSA_API_CH4': "regression_mae",
    }
    max_epochs = 200
    per_gpu_batchsize = 32

@ex.named_config
def ads_qst_ch4_n2_org():
    exp_name = "ads_qst_ch4_n2_org"
    root_dataset = 'src/moftransformer/data/round2'  # Data directory
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