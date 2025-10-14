'''
Author: zhangshd
Date: 2024-08-16 11:09:58
LastEditors: zhangshd
LastEditTime: 2024-08-17 19:19:31
'''
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import hyperopt
import pandas as pd
import numpy as np
from ml.module import *
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.svm import SVR, SVC
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from pathlib import Path
from typing import Union


def prepare_data(in_df: pd.DataFrame, test_df, valid_df, feat_cols, label_column, group_column, test_size, random_state):

    np.random.seed(random_state)
    print("Init data shape: ", in_df.shape)
    # balance the data
    in_df = in_df.dropna(subset=[label_column]).reset_index(drop=True).copy()
    if label_column in ["acid_label", "base_label", "boiling_label"] and test_df is None:
        positive_df = in_df[in_df[label_column] == 1]
        negative_df = in_df[in_df[label_column] == 0]
        negative_df = negative_df.sample(n=positive_df.shape[0], random_state=random_state)
        in_df = pd.concat([positive_df, negative_df], axis=0).reset_index(drop=True)
    print("Data shape:", in_df.shape)
    if test_df is not None:
        print("Using provided test data.")
        train_X = in_df[feat_cols]
        train_y = in_df[label_column]
        test_X = test_df[feat_cols].copy()
        test_y = test_df[label_column]
    elif (isinstance(test_size, float) and (0.0 < test_size < 1.0)) or (isinstance(test_size, int) and (0 < test_size < len(in_df))):
        print("Splitting data into train and test sets.")
        train_index, test_index = split_train_test(in_df, test_size=test_size, group_column=group_column, random_state=random_state)
        train_X = in_df.loc[train_index, feat_cols]
        train_y = in_df.loc[train_index, label_column]
        test_X = in_df.loc[test_index, feat_cols]
        test_y = in_df.loc[test_index, label_column]
    else:
        print("All samples are used as training data.")
        train_X = in_df[feat_cols]
        train_y = in_df[label_column]
        test_X = None
        test_y = None
    if valid_df is not None:
        valid_x = valid_df[feat_cols]
        valid_y = valid_df[label_column]

    else:
        valid_x = None
        valid_y = None
    train_groups = in_df.loc[train_X.index, group_column] if group_column else None
    return train_X, train_y, valid_x, valid_y, test_X, test_y, train_groups

def get_model_params(model_list, model_type, random_state, n_jobs):
    if model_type == "regression":
        model_param_dict = {
            "LR": {"estimator": LinearRegression(), "params": {}},
            "RF": {"estimator": RandomForestRegressor(random_state=random_state, n_jobs=n_jobs), "params": {
                'n_estimators': hyperopt.hp.uniformint('n_estimators', 10, 500),
                'max_leaf_nodes': hyperopt.hp.uniformint('max_leaf_nodes', 10, 150),
                'min_samples_split': hyperopt.hp.uniformint('min_samples_split', 2, 30),
                'min_samples_leaf': hyperopt.hp.uniformint('min_samples_leaf', 1, 30),
            }},
            "SVM": {"estimator": SVR(), "params": {
                'C': hyperopt.hp.uniform("C", 1e-5, 1e2),
                'gamma': hyperopt.hp.uniform("gamma", 1e-5, 1e2),
                'epsilon': hyperopt.hp.uniform("epsilon", 1e-5, 1),
            }},
            "GP": {"estimator": GaussianProcessRegressor(random_state=random_state), "params": {
                'constant_value': hyperopt.hp.uniform("constant_value", 1e-5, 1e2),
                'length_scale': hyperopt.hp.uniform("length_scale", 1e-5, 1e2),
                'noise_level': hyperopt.hp.uniform("noise_level", 1e-5, 1),
            }},
            "XGB": {"estimator": XGBRegressor(objective='reg:squarederror', random_state=random_state, n_jobs=n_jobs), "params": {
                'gamma': hyperopt.hp.uniform("gamma", 0, 0.5),
                'max_depth': hyperopt.hp.uniformint('max_depth', 3, 30),
                'min_child_weight': hyperopt.hp.uniformint('min_child_weight', 1, 50),
                'colsample_bytree': hyperopt.hp.uniform('colsample_bytree', 0.1, 1),
                'subsample': hyperopt.hp.uniform('subsample', 0.1, 1),
                'learning_rate': hyperopt.hp.uniform('learning_rate', 0.0001, 0.2),
                'n_estimators': hyperopt.hp.uniformint('n_estimators', 10, 500),
            }},
        }
    elif model_type == "classification":
        model_param_dict = {
            "LR": {"estimator": LogisticRegression(random_state=random_state, n_jobs=n_jobs, max_iter=1000), "params": {
                'C': hyperopt.hp.uniform("C", 1e-5, 1e2),
                'class_weight': hyperopt.hp.choice('class_weight', ["balanced"]),
            }},
            "RF": {"estimator": RandomForestClassifier(random_state=random_state, n_jobs=n_jobs), "params": {
                'n_estimators': hyperopt.hp.uniformint('n_estimators', 10, 500),
                'max_leaf_nodes': hyperopt.hp.uniformint('max_leaf_nodes', 10, 150),
                'min_samples_split': hyperopt.hp.uniformint('min_samples_split', 2, 30),
                'min_samples_leaf': hyperopt.hp.uniformint('min_samples_leaf', 1, 30),
                'class_weight': hyperopt.hp.choice('class_weight', ["balanced", "balanced_subsample", None])
            }},
            "SVM": {"estimator": SVC(random_state=random_state, probability=True), "params": {
                'C': hyperopt.hp.uniform("C", 1e-5, 1e2),
                'gamma': hyperopt.hp.uniform("gamma", 1e-5, 1e2),
                'class_weight': hyperopt.hp.choice('class_weight', ["balanced", None]),
            }},
            "GP": {"estimator": GaussianProcessClassifier(random_state=random_state, n_jobs=n_jobs), "params": {
                'constant_value': hyperopt.hp.uniform("constant_value", 1e-5, 1e2),
                'length_scale': hyperopt.hp.uniform("length_scale", 1e-5, 1e2),
                'noise_level': hyperopt.hp.uniform("noise_level", 1e-5, 1),
            }},
            "XGB": {"estimator": XGBClassifier(objective='binary:logistic', random_state=random_state, n_jobs=n_jobs), "params": {
                'gamma': hyperopt.hp.uniform("gamma", 0, 0.9),
                'max_depth': hyperopt.hp.uniformint('max_depth', 3, 30),
                'min_child_weight': hyperopt.hp.uniformint('min_child_weight', 1, 50),
                'colsample_bytree': hyperopt.hp.uniform('colsample_bytree', 0.1, 1),
                'subsample': hyperopt.hp.uniform('subsample', 0.1, 1),
                'learning_rate': hyperopt.hp.uniform('learning_rate', 0.001, 0.2),
                'n_estimators': hyperopt.hp.uniformint('n_estimators', 10, 500),
                # 'scale_pos_weight': hyperopt.hp.uniform('scale_pos_weight', 0.01, 2),
            }},
        }
    return {m: model_param_dict[m] for m in model_list}


def search_best_params(model: Union[RegressionModel, ClassificationModel], estimator, params_space, search_metric, search_max_evals, random_state, show_progressbar):
    def search(params):
        feature_selector = params.pop("feature_selector", "f1")
        select_des_num = int(params.pop("select_des_num", 50))
        if model.feature_selector_name != feature_selector or model.feature_select_num != select_des_num:
            model.select_feature(feature_selector=feature_selector, select_des_num=select_des_num)
        model.train(estimator, params=params)
        val_metric = model.all_metrics_df.loc["mean", search_metric]
        return val_metric if search_metric.split("_")[-1] in ["MAE", "RMSE", "CE"] else -val_metric

    best_params = hyperopt.fmin(search, space=params_space, algo=hyperopt.tpe.suggest,
                                max_evals=search_max_evals, rstate=np.random.default_rng(random_state),
                                show_progressbar=show_progressbar)
    best_params = hyperopt.space_eval(params_space, best_params)
    for key, value in params_space.items():
        if value.name == "int":
            best_params[key] = int(best_params[key])
    return best_params

def train_and_save_model(model: Union[RegressionModel, ClassificationModel], estimator, best_params, saved_dir, model_name, random_state):
    for k, v in best_params.items():
        print(f"{k}: {v}")
    select_des_num = best_params.pop("select_des_num")
    feature_selector = best_params.pop("feature_selector")
    model.select_feature(feature_selector=feature_selector, select_des_num=select_des_num, saved_dir=saved_dir, saved_file_note=model_name)
    model.train(estimator, params=best_params, saved_dir=saved_dir)
    model.fulltrain(estimator, params=best_params, saved_dir=saved_dir)
    model.all_metrics_df["model_name"] = model_name
    model.all_metrics_df["feature_selector"] = feature_selector
    model.all_metrics_df["feature_num"] = select_des_num
    model.all_metrics_df["random_state"] = random_state
    model.all_metrics_df["hyperparams"] = best_params
    metrics_out_file = os.path.join(saved_dir, "model_metrics.csv")
    model.all_metrics_df.to_csv(metrics_out_file, mode="a")
    model.save_total_model(saved_dir=saved_dir, saved_file_note=random_state)
    # model.generate_ball_tree(p=1, saved_dir=saved_dir)
    model.draw_predictions(model.val_y_all, model.val_pred_all, saved_dir=saved_dir, saved_file_note=model_name, data_group="validation")
    if model.test_y is not None:
        test_y_for_plot = model.test_y_original if hasattr(model, 'test_y_original') else model.test_y
        model.draw_predictions(test_y_for_plot, model.test_pred, saved_dir=saved_dir, saved_file_note=model_name, data_group="test")

def MainRegression(in_df, saved_dir, feature_selector_list, select_des_num_list=(50,), test_df=None, valid_df=None, model_list=("LR",),
                   search_max_evals=30, name_column="Name", label_column="", group_column="", feat_cols=None, k=5,
                   test_size=0.2, kfold_type="normal", random_state=0, search_metric="val_RMSE", scaler_name="StandardScaler",
                   show_progressbar=True, use_target_transform=False, target_transform_method="yeo-johnson", **kwargs):
    """
    Main function for regression model training and evaluation.
    
    Parameters:
    -----------
    use_target_transform : bool, optional (default=False)
        Whether to apply target transformation
    target_transform_method : str, optional (default='yeo-johnson')
        Target transformation method. Options: 'yeo-johnson', 'box-cox', 'log10'
        - 'yeo-johnson': Works with both positive and negative values
        - 'box-cox': Requires all positive values
        - 'log10': Requires all positive values, applies log10 transformation
    """
    n_jobs = min(int(0.8 * os.cpu_count()), 64)
    saved_dir = Path(saved_dir)
    saved_dir.mkdir(exist_ok=True, parents=True)

    id_lb_cols = [name_column, label_column, group_column] if group_column not in ["", None, "None", label_column] else [name_column, label_column]
    feat_cols = [x for x in in_df.columns if x not in id_lb_cols] if feat_cols is None else [x for x in feat_cols if x not in id_lb_cols]

    train_X, train_y, valid_x, valid_y, test_X, test_y, train_groups = prepare_data(in_df, test_df, valid_df, feat_cols, label_column, group_column, test_size, random_state)

    model = RegressionModel(random_state=random_state)
    model.load_data(train_X, train_y, test_X, test_y, valid_x, valid_y, train_groups=train_groups)
    if use_target_transform:
        model.transform_target(method=target_transform_method, saved_dir=saved_dir)
    model.scale_feature(saved_dir=saved_dir, scaler_name=scaler_name)
    model.kfold_split(k=k, kfold_type=kfold_type)

    model_param_dict = get_model_params(model_list, "regression", random_state, n_jobs)

    for m in model_list:
        estimator = model_param_dict[m]["estimator"]
        model_name = estimator.__class__.__name__
        params_space = {"feature_selector": hyperopt.hp.choice('feature_selector', feature_selector_list),
                        "select_des_num": hyperopt.hp.uniformint("select_des_num", 5, train_X.shape[1])}
        params_space.update(model_param_dict[m]["params"])
        best_params = search_best_params(model, estimator, params_space, search_metric, search_max_evals, random_state, show_progressbar)
        train_and_save_model(model, estimator, best_params, saved_dir, model_name, random_state)


def MainClassification(in_df, saved_dir, feature_selector_list, select_des_num_list=(50,), test_df=None, valid_df=None, model_list=("LR",),
                       search_max_evals=30, name_column="Name", label_column="", group_column="", feat_cols=None, k=5,
                       test_size=0.2, kfold_type="normal", random_state=0, search_metric="val_CE", scaler_name="StandardScaler",
                       show_progressbar=True, **kwargs):
    n_jobs = min(int(0.8 * os.cpu_count()), 64)
    saved_dir = Path(saved_dir)
    saved_dir.mkdir(exist_ok=True, parents=True)

    id_lb_cols = [name_column, label_column, group_column] if group_column not in ["", None, "None", label_column] else [name_column, label_column]
    feat_cols = [x for x in in_df.columns if x not in id_lb_cols] if feat_cols is None else [x for x in feat_cols if x not in id_lb_cols]

    train_X, train_y, valid_x, valid_y, test_X, test_y, train_groups = prepare_data(in_df, test_df, valid_df, feat_cols, label_column, group_column, test_size, random_state)

    model = ClassificationModel(random_state=random_state, n_class=len(np.unique(train_y)))
    model.load_data(train_X, train_y, test_X, test_y, valid_x, valid_y, train_groups=train_groups)
    model.label_encode(saved_dir=saved_dir, saved_file_note=label_column)
    model.scale_feature(saved_dir=saved_dir, saved_file_note=label_column, scaler_name=scaler_name)
    model.kfold_split(k=k, kfold_type=kfold_type)

    model_param_dict = get_model_params(model_list, "classification", random_state, n_jobs)

    for m in model_list:
        estimator = model_param_dict[m]["estimator"]
        model_name = estimator.__class__.__name__
        params_space = {"feature_selector": hyperopt.hp.choice('feature_selector', feature_selector_list),
                        "select_des_num": hyperopt.hp.uniformint("select_des_num", 5, train_X.shape[1])}
        params_space.update(model_param_dict[m]["params"])
        best_params = search_best_params(model, estimator, params_space, search_metric, search_max_evals, random_state, show_progressbar)
        print("*"*50)
        print("best params: \n", best_params)
        print("*"*50)
        train_and_save_model(model, estimator, best_params, saved_dir, model_name, random_state)


if __name__ == "__main__":
    t0 = time.time()
    data_dir = os.path.join(ROOT_DIR, "data/ml_data/WS24")
    in_file_name = "features_and_labels_full.csv"
    in_file_path = os.path.join(data_dir, in_file_name)

    random_state_list = (0,)
    feature_selector_list = ("f1",)
    select_des_num_list = (10, 15, 20, 30,)
    model_list = ("RF", )
    search_max_evals = 5
    name_column = "MOF_name"
    label_column = "base_label"
    group_column = "base_label"
    test_size = 0.2
    kfold_type = "stratified"
    search_metric = "val_AUC"

    saved_dir = os.path.join(ROOT_DIR, "results/ml_models", f"{in_file_name[:-4]}", label_column)
    print("saved_dir:", saved_dir)

    not_feat_cols = ["MOF_name", "data_set", ]

    in_df = pd.read_csv(in_file_path)
    feat_cols = [x for x in in_df.columns if x not in not_feat_cols]

    for random_state in random_state_list:
        MainClassification(in_df, saved_dir, feature_selector_list, select_des_num_list, model_list=model_list, k=5,
                           search_max_evals=search_max_evals, name_column=name_column, label_column=label_column,
                           group_column=group_column, feat_cols=feat_cols, test_size=test_size, kfold_type=kfold_type,
                           random_state=random_state, search_metric=search_metric, show_progressbar=True)

    print("Time cost:", sec_to_time(time.time() - t0))