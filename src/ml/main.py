'''
Author: zhangshd
Date: 2024-08-16 11:09:28
LastEditors: zhangshd
LastEditTime: 2025-05-16 13:38:43
'''
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from interface import MainRegression, MainClassification
import pandas as pd
import os
import time
from module import sec_to_time
import argparse
import shutil

def balance_neg_samples(df, label_column, random_seed=42):
    """Balance the negative samples to match positive samples."""
    pos_df = df[df[label_column] == 1]
    neg_df = df[df[label_column] == 0].sample(n=pos_df.shape[0], random_state=random_seed)
    return pd.concat([pos_df, neg_df], axis=0)

def prepare_data(in_file_path, label_column, not_feat_cols):
    """Load data and prepare feature columns."""
    df = pd.read_csv(in_file_path)
    feat_cols = [x for x in df.columns if x not in not_feat_cols]
    train_df = df[df["Partition"] == "train"].copy()
    test_df = df[df["Partition"] == "test"].copy()
    valid_df = df[df["Partition"] == "val"].copy()
    return train_df, test_df, valid_df, feat_cols

def main(data_dir, in_file_name, name_column, label_column, saved_dir=None, overwrite=True, **kwargs):
    
    """Main function to handle model training and evaluation."""

    model_type = kwargs.get('model_type', 'classification')
    
    t0 = time.time()
    in_file_path = os.path.join(data_dir, in_file_name)
    if saved_dir is None:
        saved_dir = os.path.join(ROOT_DIR, "results/ml_models", os.path.basename(data_dir), f"{in_file_name[:-4]}", label_column)
    print(f"saved_dir: {saved_dir}")
    if overwrite and os.path.exists(saved_dir):
        print(f"Directory {saved_dir} already exists. Overwriting...")
        try:
            shutil.rmtree(saved_dir)
        except Exception as e:
            print(f"Error occurred while deleting directory {saved_dir}: {e}")

    not_feat_cols_regression = ['MofName', 'Partition', 'AdsCH4_10kPa', 'AdsCH4_100kPa',
       'AdsCH4_1000kPa', 'AdsN2_10kPa', 'AdsN2_100kPa', 'AdsN2_1000kPa',
       'QstCH4', 'QstN2', 'PSA_WC_CH4', 'PSA_WC_N2', 'PSA_alpha_CH4_N2',
       'PSA_API_CH4', 'VSA_WC_CH4', 'VSA_WC_N2', 'VSA_alpha_CH4_N2',
       'VSA_API_CH4']
    not_feat_cols_classification = ["MOF_name", "data_set", "split", "MofName", "Label", "Partition"]

    if model_type == 'regression':
        train_df, test_df, valid_df, feat_cols = prepare_data(in_file_path, label_column, not_feat_cols_regression)
        MainRegression(train_df, saved_dir, test_df=test_df, valid_df=valid_df, feat_cols=feat_cols,
                       show_progressbar=False, name_column=name_column, label_column=label_column, **kwargs)
    
    elif model_type == 'classification':
        train_df, test_df, valid_df, feat_cols = prepare_data(in_file_path, label_column, not_feat_cols_classification)
        if label_column in ["acid_label", "base_label", "boiling_label"]:
            train_df = balance_neg_samples(train_df, label_column)
            test_df = balance_neg_samples(test_df, label_column)
            valid_df = balance_neg_samples(valid_df, label_column)
        MainClassification(train_df, saved_dir, test_df=test_df, valid_df=valid_df, feat_cols=feat_cols,
                           show_progressbar=False, name_column=name_column, label_column=label_column, **kwargs)
    
    else:
        print("Invalid model type!")

    print(f"Time cost: {sec_to_time(time.time() - t0)}")

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='classification', help='Model type: regression or classification')
    parser.add_argument('--data_dir', type=str, default=os.path.join(ROOT_DIR, 'data/ml_data'), help='Data directory')
    parser.add_argument('--saved_dir', type=str, default=None, help='Directory to save results')
    parser.add_argument('--in_file_name', type=str, default='id_prop_feat.csv', help='Input file name')
    parser.add_argument('--random_state_list', type=int, default=[0], nargs="+", help='List of random states')
    parser.add_argument('--feature_selector_list', type=str, default=['RFE'], nargs="+", help='List of feature selectors')
    parser.add_argument('--select_des_num_list', type=int, default=[50], nargs="+", help='List of number of features to select')
    parser.add_argument('--model_list', type=str, default=['RF'], nargs="+", help='List of models')
    parser.add_argument('--search_max_evals', type=int, default=10, help='Maximum number of evaluations for search')
    parser.add_argument('--name_column', type=str, default='MofName', help='Name column')
    parser.add_argument('--label_column', type=str, default='Label', help='Label column')
    parser.add_argument('--group_column', type=str, default='', help='Group column')
    parser.add_argument('--test_size', type=float, default=0.2, help='Size of the test set')
    parser.add_argument('--kfold_type', type=str, default='none', help='Type of k-fold cross-validation')
    parser.add_argument('--k', type=int, default=1, help='Number of folds')
    parser.add_argument('--search_metric', type=str, default='val_AUC', help='Metric for search')
    parser.add_argument('--scaler_name', type=str, default='StandardScaler', help='Name of the scaler')
    # parser.add_argument('--overwrite', action='store_true', help='Whether to overwrite existing results')
    parser.add_argument('--use_target_transform', action='store_true', help='Whether to use target transformation (for regression)')
    parser.add_argument('--target_transform_method', type=str, default='yeo-johnson', help='Method for target transformation (for regression)')
    #############
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
