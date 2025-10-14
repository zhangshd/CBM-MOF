'''
Author: zhangshd
Date: 2024-08-15 15:51:31
LastEditors: zhangshd
LastEditTime: 2025-05-16 04:08:28
'''
import os
import time
import copy
import numpy as np
import pandas as pd
import joblib
import warnings
import matplotlib.pyplot as plt
from matplotlib import colors
plt.switch_backend('agg')
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, RFE, VarianceThreshold, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, LeaveOneOut, LeaveOneGroupOut, train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, matthews_corrcoef, roc_auc_score, f1_score, balanced_accuracy_score, confusion_matrix
from sklearn.neighbors import BallTree
import copy


def sec_to_time(seconds):
    """Convert seconds to time format."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}h:{m:02d}m:{s:02d}s"


def weighted_cross_entropy(y_true, y_pred_proba, n_class):
    """Compute the weighted cross-entropy loss for multi-class classification."""
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    class_counts = np.bincount(y_true, minlength=n_class)
    total_samples = len(y_true)
    class_weights = total_samples / (n_class * class_counts)
    y_pred_proba = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
    # print(y_true)
    y_true_one_hot = np.eye(n_class)[y_true]
    weighted_ce = - (y_true_one_hot * np.log(y_pred_proba)) * class_weights
    return weighted_ce.sum(axis=1).mean()


def split_train_test(df, test_size=0.1, group_column="", random_state=0):
    """Split the dataset into train and test sets."""
    if group_column:
        train_index, test_index = train_test_split(df.index, test_size=test_size, stratify=df[group_column], random_state=random_state)
    else:
        train_index, test_index = train_test_split(df.index, test_size=test_size, random_state=random_state)
    print(f'Train test split successfully: train/test = {len(train_index)}/{len(test_index)}')
    return train_index, test_index

def plot_roc_curve(fpr, tpr, roc_auc, title=None, outfile=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    ax.plot([0, 1], [0, 1], 'r--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    if title is not None:
        ax.set_title(title)
    ax.legend(loc="lower right")
    if outfile is not None:
        plt.savefig(outfile, dpi=300, bbox_inches='tight', format='png')
    return fig, ax

def plot_scatter(targets, predictions, title: str=None, metrics: dict=None, outfile: str=None):

    targets = np.array(targets)
    predictions = np.array(predictions)
    max_value = max(targets.max(), predictions.max())
    min_value = min(targets.min(), predictions.min())
    offset = (max_value-min_value)*0.06
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(targets, predictions, alpha=0.5)
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel(f"Groud Truth")
    ax.set_ylabel(f"Predictions")

    ax.set_xlim(min_value - offset, max_value + offset)
    ax.set_ylim(min_value - offset, max_value + offset)

    ax.plot([min_value, max_value], [min_value, max_value], 'r--')  # 'r--' means red dashed line

    if metrics:
        text_content = ""
        for k, v in metrics.items():
            text_content += f"{k}: {v:.4f}\n"
        ax.text(max_value - offset*6, min_value + offset, 
            text_content, 
            fontsize=12, color='red')
    if outfile is not None:
        plt.savefig(outfile, dpi=300, bbox_inches='tight', format='png')
    return fig, ax

def plot_confusion_matrix(targets, predictions, title=None, outfile=None):
    
    cm = confusion_matrix(targets, predictions)
    if title is None:
        title = f"Confusion Matrix"
    num_classes = len(cm)
    acc = (cm.diagonal().sum()/cm.sum())*100
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm_norm, cmap='Blues')
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xlabel('Groud Truth')
    ax.set_ylabel('Predictions')
    ax.set_title(title+f'(ACC={acc:.2f}%)')
    ax.set_aspect('equal')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='black')
    if outfile is not None:
        plt.savefig(outfile, dpi=300, bbox_inches='tight', format='png')
    return fig, ax


class BaseModel:
    """Base class for Regression and Classification models."""

    def __init__(self, random_state=0):
        self.random_state = random_state
        self.color_list = list(colors.XKCD_COLORS.values())
        self.n_jobs = min(int(0.8 * os.cpu_count()), 64)
        self.feature_selector_name = "null"
        self.feature_select_num = 0
        self.model_type = None
        self.full_trained = False
        self.target_scale_factor = 1.0

    def load_data(self, train_X, train_y, test_X=None, test_y=None, valid_X=None, valid_y=None, train_groups=None):
        self.train_X = np.array(train_X)
        self.train_y = np.array(train_y)
        if test_X is not None and test_y is not None:
            self.test_X = np.array(test_X)
            self.test_y = np.array(test_y)
        else:
            self.test_X = None
            self.test_y = None
        if valid_X is not None and valid_y is not None:
            self.valid_X = np.array(valid_X)
            self.valid_y = np.array(valid_y)
        else:
            self.valid_X = None
            self.valid_y = None
        self.train_groups = np.array(train_groups)
        print("=" * 50)
        print(f"Train_X shape: {self.train_X.shape}")
        print(f"Train_y shape: {self.train_y.shape}")
        if self.test_X is not None:
            print(f"Test_X shape: {self.test_X.shape}")
            print(f"Test_y shape: {self.test_y.shape}")
        if self.valid_X is not None:
            print(f"Valid_X shape: {self.valid_X.shape}")
            print(f"Valid_y shape: {self.valid_y.shape}")
        print("=" * 50)

    def scale_feature(self, feature_range=(0, 1), saved_dir="", saved_file_note="", scaler_name="MinMaxScaler"):
        print(f"Scaled feature range: {feature_range}")
        self.scaler = MinMaxScaler(feature_range=feature_range) if scaler_name == "MinMaxScaler" else StandardScaler()
        self.train_X_init = self.train_X.copy()
        self.train_X = self.scaler.fit_transform(self.train_X)
        if self.test_X is not None and self.test_y is not None:
            self.test_X_init = self.test_X
            self.test_X = self.scaler.transform(self.test_X)
        if self.valid_X is not None and self.valid_y is not None:
            self.valid_X_init = self.valid_X
            self.valid_X = self.scaler.transform(self.valid_X)
        if saved_dir:
            scaler_x_file = os.path.join(saved_dir, f"scaler_{saved_file_note}.pkl")
            with open(scaler_x_file, 'wb') as f:
                joblib.dump(self.scaler, f)

    def select_feature(self, saved_dir="", feature_selector='f1', select_des_num=100, saved_file_note=""):
        t1 = time.time()
        self.variance_filter = VarianceThreshold(threshold=0)
        self.train_X_filtered = self.variance_filter.fit_transform(self.train_X)
        print(f"Features shape after variance filter: {self.train_X_filtered.shape}")
        print(f'Executing feature selection on features by {feature_selector}.')
        select_des_num = min(select_des_num, self.train_X_filtered.shape[1])
        if self.model_type == "regression":
            if feature_selector == 'RFE':
                base_model = RandomForestRegressor(n_estimators=20, random_state=self.random_state, n_jobs=self.n_jobs)
                self.selector = RFE(base_model, n_features_to_select=select_des_num, step=0.01)
            elif feature_selector in ['f1', 'f_regression']:
                self.selector = SelectKBest(score_func=f_regression, k=select_des_num)
            elif feature_selector in ['mutual_info','mutual_info_regression']:
                self.selector = SelectKBest(score_func=mutual_info_regression, k=select_des_num)
            else:
                raise NotImplementedError(f"Feature selector choice: REF/f1/mutual_info, {feature_selector} is not implemented.")
        elif self.model_type == "classification":
            if feature_selector == 'RFE':
                base_model = RandomForestClassifier(n_estimators=20, random_state=self.random_state, n_jobs=self.n_jobs)
                self.selector = RFE(base_model, n_features_to_select=select_des_num, step=0.01)
            elif feature_selector in ['f1', 'f_classif']:
                self.selector = SelectKBest(score_func=f_classif, k=select_des_num)
            elif feature_selector in ['mutual_info','mutual_info_classif']:
                self.selector = SelectKBest(score_func=mutual_info_classif, k=select_des_num)
            else:
                raise NotImplementedError(f"Feature selector choice: REF/f1/mutual_info, {feature_selector} is not implemented.")
        else:
            raise NotImplementedError(f"model_type choice: regression/classification, {self.model_type} is not implemented.")
        self.selector.fit(self.train_X_filtered, self.train_y)
        self.train_X_selected = self.selector.transform(self.train_X_filtered)
        if self.test_X is not None and self.test_y is not None:
            self.test_X_filtered = self.variance_filter.transform(self.test_X)
            self.test_X_selected = self.selector.transform(self.test_X_filtered)
        if self.valid_X is not None and self.valid_y is not None:
            self.valid_X_filtered = self.variance_filter.transform(self.valid_X)
            self.valid_X_selected = self.selector.transform(self.valid_X_filtered)
        self.feature_selector_name = feature_selector
        self.feature_select_num = select_des_num
        if saved_dir:
            variance_file = os.path.join(saved_dir, f"variance_{saved_file_note}.pkl")
            with open(variance_file, 'wb') as f:
                joblib.dump(self.variance_filter, f)
            selector_file = os.path.join(saved_dir, f'selector_{saved_file_note}.pkl')
            with open(selector_file, 'wb') as f:
                joblib.dump(self.selector, f)
        print(f"Selected feature num: {self.train_X_selected.shape[1]}")
        print(f'Time cost for selection: {sec_to_time(time.time() - t1)}')
        print('')

    def kfold_split(self, k=5, kfold_type="normal"):
        np.random.seed(self.random_state)
        train_val_idxs = []
        self.k = k
        if kfold_type in [None, "none"] or k == 1:
            print("Using none KFold, all training.")
            train_idx = np.random.permutation(len(self.train_X))
            val_idx = np.random.permutation(len(self.train_X))
            train_val_idxs.append([train_idx, val_idx])
        elif kfold_type == 'group' and k != 1:
            print("Using GroupKFold.")
            kf = GroupKFold(k)
            for train_idx, val_idx in kf.split(self.train_X, groups=self.train_groups):
                train_val_idxs.append([np.random.permutation(train_idx), np.random.permutation(val_idx)])
        elif kfold_type == 'stratified' and k != 1:
            print("Using StratifiedKFold.")
            kf = StratifiedKFold(k, shuffle=True, random_state=self.random_state)
            for train_idx, val_idx in kf.split(self.train_X, y=self.train_groups):
                train_val_idxs.append([np.random.permutation(train_idx), np.random.permutation(val_idx)])
        elif kfold_type == 'loo':
            print("Using LeaveOneOut.")
            kf = LeaveOneOut()
            for train_idx, val_idx in kf.split(self.train_X):
                train_val_idxs.append([np.random.permutation(train_idx), np.random.permutation(val_idx)])
        elif kfold_type == 'logo':
            print("Using LeaveOneGroupOut.")
            kf = LeaveOneGroupOut()
            for train_idx, val_idx in kf.split(self.train_X, groups=self.train_groups):
                train_val_idxs.append([np.random.permutation(train_idx), np.random.permutation(val_idx)])
        else:
            print("Using normal KFold.")
            kf = KFold(k, shuffle=True, random_state=self.random_state)
            for train_idx, val_idx in kf.split(self.train_X):
                train_val_idxs.append([np.random.permutation(train_idx), np.random.permutation(val_idx)])
        self.kfold_type = kfold_type
        self.train_val_idxs = train_val_idxs

    def save_total_model(self, saved_dir, saved_file_note=""):
        total_model_file = os.path.join(saved_dir, f"total_model_{self.feature_selector_name}{self.feature_select_num}_{self.kfold_type}{self.k}_{self.model_name}_{saved_file_note}.model")
        with open(total_model_file, "wb+") as f:
            joblib.dump(self, f)
        print(f"The total model file has been saved: {total_model_file}")

    def load_total_model(self, model_file):
        with open(model_file, 'rb+') as f:
            new_model = joblib.load(f)
        for key, value in new_model.__dict__.items():
            self.__setattr__(key, value)

    def draw_predictions(self, y_true, y_pred, saved_dir, saved_file_note="", data_group="test"):
        
        fig_file = os.path.join(saved_dir, f"{data_group}_predicted_{saved_file_note}.png")
        if self.model_type == "regression":
            metrics = {"R2": r2_score(y_true, y_pred), 
                       "MSE": mean_squared_error(y_true, y_pred), 
                       "MAE": mean_absolute_error(y_true, y_pred)}
            plot_scatter(y_true, y_pred, title=f"{data_group} predicted", metrics=metrics, outfile=fig_file)
        elif self.model_type == "classification":
            plot_confusion_matrix(y_true, y_pred, title=f"{data_group} predicted", outfile=fig_file)
        else:
            raise NotImplementedError(f"model_type choice: regression/classification, {self.model_type} is not implemented.")

    def generate_ball_tree(self, p=1, neighbors_num=1, saved_dir=""):
        self.balltrees = []
        self.ref_dist_values = []
        self.feature_weights_list = []
        if self.full_trained:
            model = self.model
            if hasattr(model, "coef_"):
                feature_weights = model.coef_
            elif hasattr(model, "feature_importances_"):
                feature_weights = model.feature_importances_
            else:
                warnings.warn("The trained models have no attribute like 'coef_' or 'feature_importances_'")
                feature_weights = np.array([1] * self.train_X_selected.shape[1])
            scaler_w = MinMaxScaler(feature_range=(0.1, 0.9))
            feature_weights = scaler_w.fit_transform(feature_weights.reshape(-1, 1)).squeeze()
            self.feature_weights_list.append(feature_weights)
            balltree = BallTree(self.train_X_selected, metric='minkowski', p=p)
            if saved_dir:
                balltree_file = os.path.join(saved_dir, f"balltree_{self.feature_selector_name}{self.feature_select_num}_full_{self.model_name}.pkl")
                with open(balltree_file, 'wb') as f:
                    joblib.dump(balltree, f)
            self.balltrees.append(balltree)
            dist_mean = balltree.query(self.train_X_selected, k=neighbors_num, dualtree=True)[0][:, -1].mean()
            ref_dist_value = 2.5 * np.quantile(dist_mean, 0.75) - 1.5 * np.quantile(dist_mean, 0.25)  # Q3+1.5IQR
            self.ref_dist_values.append(ref_dist_value)
        else:
            for i, (train_idx, val_idx) in enumerate(self.train_val_idxs):
                model = self.models[i]
                if hasattr(model, "coef_"):
                    feature_weights = model.coef_
                elif hasattr(model, "feature_importances_"):
                    feature_weights = model.feature_importances_
                else:
                    warnings.warn("The trained models have no attribute like 'coef_' or 'feature_importances_'")
                    feature_weights = np.array([1] * self.train_X_selected.shape[1])
                scaler_w = MinMaxScaler(feature_range=(0.1, 0.9))
                feature_weights = scaler_w.fit_transform(feature_weights.reshape(-1, 1)).squeeze()
                self.feature_weights_list.append(feature_weights)
                kf_train_X = self.train_X_selected[train_idx]
                kf_val_X = self.train_X_selected[val_idx]
                balltree = BallTree(kf_train_X, metric='minkowski', p=p)
                if saved_dir:
                    with open(os.path.join(saved_dir, f"balltree_p{p}_fold_{i + 1}.pkl"), "wb+") as f:
                        joblib.dump(balltree, f)
                dist, ind = balltree.query(kf_val_X, k=neighbors_num, dualtree=True)
                dist_mean = dist.mean(axis=1)
                ref_dist_value = 2.5 * np.quantile(dist_mean, 0.75) - 1.5 * np.quantile(dist_mean, 0.25)  # Q3+1.5IQR
                self.ref_dist_values.append(ref_dist_value)
                print("*" * 50)
                print(f"Get reference distance value from validation set of fold {i + 1}: \033[36;1m{ref_dist_value}\033[0m")
                self.balltrees.append(balltree)

    def visualize_chem_space(self, train_X, test_X, saved_dir, method="UMAP", notes=""):
        """Visualize chemical space using UMAP, tSNE or PCA."""
        all_X = np.concatenate((train_X, test_X), axis=0)
        if method == "UMAP":
            dimension_model = UMAP(n_components=2, random_state=self.random_state, n_neighbors=15, min_dist=0.1)
            reduction_X = dimension_model.fit_transform(all_X)
        elif method == "tSNE":
            dimension_model = TSNE(n_components=2, perplexity=30, random_state=self.random_state)
            reduction_X = dimension_model.fit_transform(all_X)
        elif method == "PCA":
            dimension_model = PCA(n_components=2)
            reduction_X = dimension_model.fit_transform(all_X)
        else:
            raise NotImplementedError
        dimension_model_name = dimension_model.__class__.__name__
        reduction_X_tr = reduction_X[:len(train_X)]
        reduction_X_te = reduction_X[len(train_X):]
        plt.clf()
        plt.figure(figsize=(6, 6))
        plt.plot(reduction_X_tr[:, 0], reduction_X_tr[:, 1], linestyle='', marker='+',
                 color=self.color_list[-1], markerfacecolor='w', markersize=8, label="training set")
        plt.plot(reduction_X_te[:, 0], reduction_X_te[:, 1], linestyle='', marker='o',
                 color=self.color_list[-2], markerfacecolor='w', markersize=6, label="test set")
        plt.xlabel(f"{dimension_model_name}1", fontdict={'fontsize': 15})
        plt.ylabel(f"{dimension_model_name}2", fontdict={'fontsize': 15})
        plt.legend(loc="best")
        plt.savefig(os.path.join(saved_dir, f"train_test_distribution_{dimension_model_name}_{notes}.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()


class RegressionModel(BaseModel):
    """Regression Model class with methods for training and evaluation."""

    def __init__(self, random_state=0):
        super().__init__(random_state)
        self.metrics_list = ["R2", "RMSE", "MAE", "Pearson", "Spearman"]
        self.model_type = "regression"
        self.target_transformer = None
        self.use_target_transform = False

    def transform_target(self, method="yeo-johnson", saved_dir="", saved_file_note=""):
        """
        Transform target variable to improve model performance.
        
        Parameters:
        -----------
        method : str, optional (default='yeo-johnson')
            Transformation method. Options: 'yeo-johnson', 'box-cox', 'log10'
        saved_dir : str, optional
            Directory to save the transformer object
        saved_file_note : str, optional
            Note to append to the saved file name
            
        Notes:
        ------
        - Yeo-Johnson works with both positive and negative values
        - Box-Cox requires all positive values
        - log10 requires all positive values and applies log10 transformation
        - The transformer will be saved and used for inverse transformation during prediction
        """
        print(f"Applying {method} transformation to target variable.")
        self.use_target_transform = True
        self.target_transform_method = method
        
        # Store original target values
        self.train_y_original = self.train_y.copy()

        if method == 'log10':
            # For log10 transformation
            self.target_transformer = None
            
            # Check if all values are positive
            if np.any(self.train_y <= 0):
                raise ValueError("log10 transformation requires all positive values in target variable.")
            
            # Transform training targets: scale then log10
            self.train_y = self.train_y * self.target_scale_factor
            self.train_y = np.log10(self.train_y)
            
            # Transform test targets if they exist: scale then log10
            if self.test_y is not None:
                self.test_y_original = self.test_y.copy()
                if np.any(self.test_y <= 0):
                    raise ValueError("log10 transformation requires all positive values in test target variable.")
                self.test_y = self.test_y * self.target_scale_factor
                self.test_y = np.log10(self.test_y)
            
            # Transform validation targets if they exist: scale then log10
            if self.valid_y is not None:
                self.valid_y_original = self.valid_y.copy()
                if np.any(self.valid_y <= 0):
                    raise ValueError("log10 transformation requires all positive values in validation target variable.")
                self.valid_y = self.valid_y * self.target_scale_factor
                self.valid_y = np.log10(self.valid_y)
        else:

            # Compute target scale factor
            self.target_scale_factor = float(1.0 / np.median(self.train_y))
            print(f"Target scale factor: {self.target_scale_factor}")
            
            # For PowerTransformer (yeo-johnson, box-cox)
            self.target_transformer = PowerTransformer(method=method, standardize=True)
            
            # Transform training targets: scale then power transform
            self.train_y = self.train_y * self.target_scale_factor
            self.train_y = self.target_transformer.fit_transform(self.train_y.reshape(-1, 1)).ravel()
            if hasattr(self.target_transformer, 'lambdas_'):
                print(f"Transformer's lambda: {self.target_transformer.lambdas_}")
            
            # Transform test targets if they exist: scale then power transform
            if self.test_y is not None:
                self.test_y_original = self.test_y.copy()
                self.test_y = self.test_y * self.target_scale_factor
                self.test_y = self.target_transformer.transform(self.test_y.reshape(-1, 1)).ravel()
            
            # Transform validation targets if they exist: scale then power transform
            if self.valid_y is not None:
                self.valid_y_original = self.valid_y.copy()
                self.valid_y = self.valid_y * self.target_scale_factor
                self.valid_y = self.target_transformer.transform(self.valid_y.reshape(-1, 1)).ravel()
        
        # Save transformer if directory is provided
        if saved_dir:
            transformer_file = os.path.join(saved_dir, f"target_transformer_{saved_file_note}.pkl")
            with open(transformer_file, 'wb') as f:
                if method == 'log10':
                    joblib.dump({'method': 'log10', 'scale_factor': self.target_scale_factor}, f)
                else:
                    joblib.dump(self.target_transformer, f)
            print(f"Target transformer saved to: {transformer_file}")
        
        print(f"Target transformation completed. Method: {method}")
        
    def inverse_transform_target(self, y_transformed):
        """
        Inverse transform the target variable back to original scale.
        
        Parameters:
        -----------
        y_transformed : array-like
            Transformed target values
            
        Returns:
        --------
        y_original : array-like
            Original scale target values
        """
        if not self.use_target_transform:
            return y_transformed
        
        if self.target_transform_method == 'log10':
            # Inverse log10: 10^y then divide by scale factor
            y_original = np.power(10, y_transformed)
            y_original = y_original / self.target_scale_factor
        else:
            # Inverse power transform then divide by scale factor
            y_original = self.target_transformer.inverse_transform(y_transformed.reshape(-1, 1)).ravel()
            y_original = y_original / self.target_scale_factor
        
        return y_original

    def cal_metrics(self, y_true, y_pred):

        ## fill out NaN values
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        if (np.isnan(y_true).sum() > 0) or (np.isnan(y_pred).sum() > 0):
            warnings.warn("There are NaN values in y_true or y_pred, which will be filled with the mean of non-NaN values.")
            print(f"NaN num in y_true: {np.isnan(y_true).sum()}, NaN num in y_pred: {np.isnan(y_pred).sum()}")
        y_true = np.nan_to_num(y_true, nan=np.nanmean(y_true))
        y_pred = np.nan_to_num(y_pred, nan=np.nanmean(y_pred))
        ## calculate metrics
        r2 = r2_score(y_true=y_true, y_pred=y_pred)
        rmse = mean_squared_error(y_true=y_true, y_pred=y_pred) ** 0.5
        mae = mean_absolute_error(y_true=y_true, y_pred=y_pred)
        df_y = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
        pearson_corr = df_y.corr().iloc[0, 1]
        spearman_corr = df_y.corr("spearman").iloc[0, 1]
        return r2, rmse, mae, pearson_corr, spearman_corr

    def train(self, estimator, params, saved_dir=""):
        self.model = estimator
        self.params = copy.deepcopy(params)
        self.model_name = self.model.__class__.__name__
        tick = time.time()
        if isinstance(self.model, GaussianProcessRegressor) and "kernel" not in self.params:
            kernel = ConstantKernel(constant_value=self.params.pop("constant_value")) \
                     * RBF(length_scale=self.params.pop("length_scale")) \
                     + WhiteKernel(noise_level=self.params.pop("noise_level"))
            self.params.update(kernel=kernel)
        self.model.set_params(**self.params)
        train_metrics_all, val_metrics_all, test_metrics_all = [], [], []
        test_pred_all, val_pred_all, val_y_all = [], [], []
        self.models = []
        if hasattr(self, 'valid_X_selected') and self.valid_X_selected is not None:
            ## if validation set is provided, train model without cross-validation
            print("=" * 50)
            print(f"Train/validation num: {len(self.train_y)}/{len(self.valid_y)}")
            self.model.fit(self.train_X_selected, self.train_y)
            self.models.append(copy.copy(self.model))
            if saved_dir:
                with open(os.path.join(saved_dir, f'{self.model_name}.pkl'), 'wb+') as f:
                    joblib.dump(self.model, f)
                self.visualize_chem_space(self.train_X_selected, self.valid_X_selected, saved_dir=saved_dir, method="UMAP", notes="valid")
            train_pred = self.model.predict(self.train_X_selected)
            
            valid_pred = self.model.predict(self.valid_X_selected)
            
            # Inverse transform predictions if target transformation was used
            if self.use_target_transform:
                train_pred_original = self.inverse_transform_target(train_pred)
                valid_pred_original = self.inverse_transform_target(valid_pred)
                train_y_for_metrics = self.train_y_original
                valid_y_for_metrics = self.valid_y_original
            else:
                train_pred_original = train_pred
                valid_pred_original = valid_pred
                train_y_for_metrics = self.train_y
                valid_y_for_metrics = self.valid_y
            
            train_metrics = self.cal_metrics(train_y_for_metrics, train_pred_original)
            valid_metrics = self.cal_metrics(valid_y_for_metrics, valid_pred_original)
            train_metrics_all.append(train_metrics)
            val_metrics_all.append(valid_metrics)
            val_pred_all.extend(valid_pred_original)
            
            val_y_all.extend(self.valid_y_original if self.use_target_transform else self.valid_y)
            if hasattr(self, "test_X_selected") and self.test_y is not None:
                
                test_pred = self.model.predict(self.test_X_selected)
                
                # Inverse transform test predictions if target transformation was used
                if self.use_target_transform:
                    test_pred_original = self.inverse_transform_target(test_pred)
                    test_y_for_metrics = self.test_y_original
                else:
                    test_pred_original = test_pred
                    test_y_for_metrics = self.test_y
                
                test_pred_all.append(test_pred_original)
                
                test_metrics = self.cal_metrics(test_y_for_metrics, test_pred_original)
                test_metrics_all.append(test_metrics)
                if saved_dir:
                    self.visualize_chem_space(self.train_X_selected, self.test_X_selected, saved_dir=saved_dir, method="UMAP", notes="test")
            self._aggregate_metrics(train_metrics_all, val_metrics_all, test_metrics_all, val_pred_all, val_y_all, test_pred_all, saved_dir)
            print('Total run time:', sec_to_time(time.time() - tick))
            return
        for i, (train_idx, val_idx) in enumerate(self.train_val_idxs):
            print("=" * 50)
            print(f"Train/validation num: {len(train_idx)}/{len(val_idx)}")
            kf_train_X = self.train_X_selected[train_idx]
            kf_train_y = self.train_y[train_idx]
            kf_val_X = self.train_X_selected[val_idx]
            kf_val_y = self.train_y[val_idx]
            
            # Get original target values for metrics calculation
            if self.use_target_transform:
                kf_train_y_original = self.train_y_original[train_idx]
                kf_val_y_original = self.train_y_original[val_idx]
            else:
                kf_train_y_original = kf_train_y
                kf_val_y_original = kf_val_y
            
            self.model.fit(kf_train_X, kf_train_y)
            self.models.append(copy.copy(self.model))
            if saved_dir:
                with open(os.path.join(saved_dir, f'{self.model_name}_{i + 1}.pkl'), 'wb+') as f:
                    joblib.dump(self.model, f)
                self.visualize_chem_space(kf_train_X, kf_val_X, saved_dir=saved_dir, method="UMAP", notes=f"fold{i + 1}")
            kf_train_pred = self.model.predict(kf_train_X)
            kf_val_pred = self.model.predict(kf_val_X)
            
            # Inverse transform predictions if target transformation was used
            if self.use_target_transform:
                kf_train_pred_original = self.inverse_transform_target(kf_train_pred)
                kf_val_pred_original = self.inverse_transform_target(kf_val_pred)
            else:
                kf_train_pred_original = kf_train_pred
                kf_val_pred_original = kf_val_pred
            
            train_metrics_all.append(self.cal_metrics(kf_train_y_original, kf_train_pred_original))
            val_metrics_all.append(self.cal_metrics(kf_val_y_original, kf_val_pred_original) if self.kfold_type != "loo" else [None] * len(self.metrics_list))
            val_pred_all.extend(kf_val_pred_original)
            val_y_all.extend(kf_val_y_original)
            if hasattr(self, "test_X_selected") and self.test_y is not None:
                test_pred = self.model.predict(self.test_X_selected)
                
                # Inverse transform test predictions if target transformation was used
                if self.use_target_transform:
                    test_pred_original = self.inverse_transform_target(test_pred)
                    test_y_for_metrics = self.test_y_original
                else:
                    test_pred_original = test_pred
                    test_y_for_metrics = self.test_y
                
                test_pred_all.append(test_pred_original)
                test_metrics_all.append(self.cal_metrics(y_true=test_y_for_metrics, y_pred=test_pred_original))
        self._aggregate_metrics(train_metrics_all, val_metrics_all, test_metrics_all, val_pred_all, val_y_all, test_pred_all, saved_dir)
        print('Total run time:', sec_to_time(time.time() - tick))

    def _aggregate_metrics(self, train_metrics_all, val_metrics_all, test_metrics_all, val_pred_all, val_y_all, test_pred_all, saved_dir):
        metrics_list = self.metrics_list
        self.train_metrics_df = pd.DataFrame(train_metrics_all, columns=["tr_" + s for s in metrics_list],
                                             index=[f'fold_{i + 1}' for i in range(len(self.train_val_idxs))])
        self.val_metrics_df = pd.DataFrame(val_metrics_all, columns=["val_" + s for s in metrics_list],
                                           index=[f'fold_{i + 1}' for i in range(len(self.train_val_idxs))])
        metrics_dfs = [self.train_metrics_df, self.val_metrics_df]

        self.val_pred_all = np.array(val_pred_all, dtype=np.float32)
        self.val_y_all = np.array(val_y_all, dtype=np.float32)

        if hasattr(self, "test_X_selected") and self.test_y is not None:
            self.test_metrics_df = pd.DataFrame(test_metrics_all, columns=["te_" + s for s in metrics_list],
                                                index=[f'fold_{i + 1}' for i in range(len(self.train_val_idxs))])
            metrics_dfs.append(self.test_metrics_df)                # Save original prediction values
            original_pred = test_pred_all[0].squeeze() if len(test_pred_all) == 1 else None
            
            self.test_pred = np.mean(test_pred_all, axis=0)
            
            # Calculate the difference between original predictions and average predictions
            if original_pred is not None:
                saved_pred = self.test_pred.squeeze()
                diff = np.abs(original_pred - saved_pred).max()
                print(f"Maximum prediction difference: {diff}")
                
                # Print original metrics and recalculated metrics
                original_metrics = test_metrics_all[0]
                test_y_original = self.test_y_original if self.use_target_transform else self.test_y
                recalculated_metrics = self.cal_metrics(test_y_original.squeeze(), self.test_pred.squeeze())
                
                print(f"Original metrics: R2={original_metrics[0]}, RMSE={original_metrics[1]}, MAE={original_metrics[2]}")
                print(f"Recalculated metrics: R2={recalculated_metrics[0]}, RMSE={recalculated_metrics[1]}, MAE={recalculated_metrics[2]}")
                
                # Calculate metric differences
                metrics_diff = [abs(original_metrics[i] - recalculated_metrics[i]) for i in range(3)]
                print(f"Metric differences: R2 diff={metrics_diff[0]}, RMSE diff={metrics_diff[1]}, MAE diff={metrics_diff[2]}")
            
            test_y_original = self.test_y_original if self.use_target_transform else self.test_y
            df_te_pred = pd.DataFrame({
                "GroundTruth": test_y_original.squeeze(),
                "Predicted": self.test_pred.squeeze()
            })
            if saved_dir:
                test_pred_file = os.path.join(saved_dir, f"test_predicted_{self.model_name}.csv")
                df_te_pred.to_csv(test_pred_file, index=False)
                
                # Also save original predictions for comparison
                if original_pred is not None:
                    df_original_pred = pd.DataFrame({
                        "GroundTruth": test_y_original.squeeze(),
                        "Predicted": original_pred
                    })
                    original_test_pred_file = os.path.join(saved_dir, f"original_test_predicted_{self.model_name}.csv")
                    df_original_pred.to_csv(original_test_pred_file, index=False)
                    print(f"Original prediction values saved to {original_test_pred_file}")

        all_metrics_df = pd.concat(metrics_dfs, axis=1).T
        all_metrics_df['mean'] = all_metrics_df.mean(axis=1)
        if self.kfold_type == "loo":
            val_mean_metric = self.cal_metrics(self.val_y_all, self.val_pred_all)
            for c_idx, col in enumerate(["val_" + s for s in metrics_list]):
                all_metrics_df.loc[col, 'mean'] = val_mean_metric[c_idx]
        self.all_metrics_df = all_metrics_df.T
        print('*' * 50)
        print("All results for k-fold cross validation: ")
        print(self.all_metrics_df[[col for col in self.all_metrics_df.columns if (("R2" in col) or ("RMSE" in col))]])

    def predict(self, X, cal_feature_distance=False, neighbors_num=1):
        X = np.array(X)
        print(f"Predicting {X.shape[0]} samples with {X.shape[1]} features.")
        if len(X.shape) == 1:
            X = X.reshape((1, -1))
        if hasattr(self, "scaler"):
            X = self.scaler.transform(X)
            print("Features scaled.")
        if hasattr(self, "variance_filter"):
            X = self.variance_filter.transform(X)
            print("Low variance features removed. Current shape:", X.shape)
        if hasattr(self, "selector"):
            X = self.selector.transform(X)
            print("Features selected. Current shape:", X.shape)
        all_y_pred = []
        for model in self.models:
            y_pred = model.predict(X)
            all_y_pred.append(y_pred)
        y_pred_mean = np.mean(all_y_pred, axis=0)
        
        # Inverse transform predictions if target transformation was used
        if self.use_target_transform:
            y_pred_mean = self.inverse_transform_target(y_pred_mean)
        
        y_pred_mean = y_pred_mean.reshape((-1, 1))
        
        if cal_feature_distance and hasattr(self, "balltrees"):
            dist_means = []
            confident_indexes = []
            for i, balltree in enumerate(self.balltrees):
                dist, ind = balltree.query(X, k=neighbors_num, dualtree=True)
                dist_mean = dist.mean(axis=1).reshape((-1, 1))
                dist_means.append(dist_mean)
                confident_index = self.ref_dist_values[i] / (dist_mean + 0.00001)
                confident_indexes.append(confident_index)
            dist_mean = np.mean(dist_means, axis=0)
            confident_index = np.mean(confident_indexes, axis=0)
            y_pred_mean = np.hstack([y_pred_mean, dist_mean, confident_index])
        return y_pred_mean

    def fulltrain(self, estimator, params, saved_dir=""):
        self.model = estimator
        self.params = params
        self.model_name = self.model.__class__.__name__
        tick = time.time()
        if isinstance(self.model, GaussianProcessRegressor):
            kernel = ConstantKernel(constant_value=self.params.pop("constant_value")) \
                     * RBF(length_scale=self.params.pop("length_scale")) \
                     + WhiteKernel(noise_level=self.params.pop("noise_level"))
            self.params.update(kernel=kernel)
        self.model.set_params(**self.params)
        if hasattr(self, 'valid_X_selected') and self.valid_X_selected is not None:
            ## concatenate train and valid data for training
            train_X_selected = np.vstack([self.train_X_selected, self.valid_X_selected])
            train_y = np.hstack([self.train_y, self.valid_y])
        else:
            train_X_selected = self.train_X_selected
            train_y = self.train_y
        self.model.fit(train_X_selected, train_y)
        if saved_dir:
            with open(os.path.join(saved_dir, f'{self.model_name}_full.pkl'), 'wb+') as f:
                joblib.dump(self.model, f)
        train_pred = self.model.predict(train_X_selected)
        
        # Inverse transform predictions if target transformation was used
        if self.use_target_transform:
            train_pred_original = self.inverse_transform_target(train_pred)
            train_y_for_metrics = self.train_y_original if hasattr(self, 'train_y_original') else train_y
            if self.valid_y is not None:
                train_y_for_metrics = np.hstack([self.train_y_original, self.valid_y_original])
        else:
            train_pred_original = train_pred
            train_y_for_metrics = train_y
        
        train_metrics = self.cal_metrics(train_y_for_metrics, train_pred_original)
        self.all_metrics_df.loc['full', ["tr_" + s for s in self.metrics_list]] = train_metrics
        if hasattr(self, "test_X_selected") and self.test_y is not None:
            self.test_pred = self.model.predict(self.test_X_selected)
            
            # Inverse transform test predictions if target transformation was used
            if self.use_target_transform:
                self.test_pred = self.inverse_transform_target(self.test_pred)
                test_y_for_metrics = self.test_y_original if hasattr(self, 'test_y_original') else self.test_y
            else:
                test_y_for_metrics = self.test_y
            
            test_metrics = self.cal_metrics(test_y_for_metrics, self.test_pred)
            self.all_metrics_df.loc['full', ["te_" + s for s in self.metrics_list]] = test_metrics
        # print(self.all_metrics_df.columns)
        print(self.all_metrics_df[[col for col in self.all_metrics_df.columns if (("R2" in col) or ("RMSE" in col))]])
        print('Total run time:', sec_to_time(time.time() - tick))
        self.full_trained = True


class ClassificationModel(BaseModel):
    """Classification Model class with methods for training and evaluation."""

    def __init__(self, random_state=0, n_class=2):
        super().__init__(random_state)
        self.prob_thd = 0.5
        self.metrics_list = ["ACC", "BACC", "MCC", "AUC", "F1", "CE"]
        self.n_class = n_class
        self.model_type = "classification"
        self.full_trained = False
        self.test_pred_prob_all = []

    def cal_metrics(self, y_true, y_pred, y_pred_prob=None, n_class=2):
        y_true = np.array(y_true, dtype=np.int8)
        y_pred = np.array(y_pred)
        acc = round(accuracy_score(y_true, y_pred), 4)
        mcc = round(matthews_corrcoef(y_true, y_pred), 4)
        bacc = round(balanced_accuracy_score(y_true, y_pred), 4)
        f1 = round(f1_score(y_true, y_pred, average='macro'), 4)
        if y_pred_prob is not None:
            y_pred_prob = np.array(y_pred_prob, dtype=np.float32)
            ce = weighted_cross_entropy(y_true, y_pred_prob, n_class)
            if n_class == 2 and len(y_pred_prob.shape) == 2:
                y_pred_prob = y_pred_prob[:, 1]
            if n_class > 2:
                auc = round(roc_auc_score(y_true, y_pred_prob, multi_class='ovo', average='macro'), 4)
            else:
                auc = round(roc_auc_score(y_true, y_pred_prob), 4)
        else:
            ce = 0
            auc = 0
        return acc, bacc, mcc, auc, f1, ce
    def label_encode(self, saved_dir=None, saved_file_note=""):
        self.label_encoder = LabelEncoder()
        self.train_y_init = copy.deepcopy(self.train_y)
        self.train_y = self.label_encoder.fit_transform(self.train_y)
        if hasattr(self, "test_y") and self.test_y is not None:
            self.test_y_init = copy.deepcopy(self.test_y)
            self.test_y = self.label_encoder.transform(self.test_y)
        if hasattr(self, "valid_y") and self.valid_y is not None:
            self.valid_y_init = copy.deepcopy(self.valid_y)
            self.valid_y = self.label_encoder.transform(self.valid_y)
        if saved_dir:
            with open(os.path.join(saved_dir, f"label_encoder_{saved_file_note}.pkl"), 'wb+') as f:
                joblib.dump(self.label_encoder, f)

    def train(self, estimator, params, saved_dir=""):
        self.model = estimator
        self.params = copy.deepcopy(params)
        self.model_name = self.model.__class__.__name__
        tick = time.time()
        if isinstance(self.model, GaussianProcessClassifier) and "kernel" not in self.params:
            kernel = ConstantKernel(constant_value=self.params.pop("constant_value")) \
                     * RBF(length_scale=self.params.pop("length_scale")) \
                     + WhiteKernel(noise_level=self.params.pop("noise_level"))
            self.params.update(kernel=kernel)
        self.model.set_params(**self.params)
        train_metrics_all, val_metrics_all, test_metrics_all = [], [], []
        test_pred_all, test_pred_prob_all, val_pred_all, val_pred_prob_all, val_y_all = [], [], [], [], []
        self.models = []

        if hasattr(self, 'valid_X_selected') and self.valid_X_selected is not None:
            ## if validation set is provided, train model without cross-validation
            print("=" * 50)
            print(f"Train/validation num: {len(self.train_y)}/{len(self.valid_y)}")
            self.model.fit(self.train_X_selected, self.train_y)
            self.models.append(copy.copy(self.model))
            if saved_dir:
                with open(os.path.join(saved_dir, f'{self.model_name}.pkl'), 'wb+') as f:
                    joblib.dump(self.model, f)
                self.visualize_chem_space(self.train_X_selected, self.valid_X_selected, saved_dir=saved_dir, method="UMAP", notes="valid")
            train_pred = self.model.predict(self.train_X_selected)
            train_pred_prob = self.model.predict_proba(self.train_X_selected)
            valid_pred = self.model.predict(self.valid_X_selected)
            valid_pred_prob = self.model.predict_proba(self.valid_X_selected)
            train_metrics = self.cal_metrics(self.train_y, train_pred, train_pred_prob, n_class=self.n_class)
            valid_metrics = self.cal_metrics(self.valid_y, valid_pred, valid_pred_prob, n_class=self.n_class)
            train_metrics_all.append(train_metrics)
            val_metrics_all.append(valid_metrics)
            val_pred_all.extend(valid_pred)
            val_pred_prob_all.extend(valid_pred_prob)
            val_y_all.extend(self.valid_y)
            if hasattr(self, "test_X_selected") and self.test_y is not None:
                test_pred_prob = self.model.predict_proba(self.test_X_selected)
                test_pred = self.model.predict(self.test_X_selected)
                test_pred_all.append(test_pred)
                test_pred_prob_all.append(test_pred_prob)
                test_metrics = self.cal_metrics(self.test_y, test_pred, test_pred_prob, n_class=self.n_class)
                test_metrics_all.append(test_metrics)
                if saved_dir:
                    self.visualize_chem_space(self.train_X_selected, self.test_X_selected, saved_dir=saved_dir, method="UMAP", notes="test")
            self._aggregate_metrics(train_metrics_all, val_metrics_all, test_metrics_all, val_pred_all, val_pred_prob_all, val_y_all, test_pred_all, saved_dir, test_pred_prob_all)
            print('Total run time:', sec_to_time(time.time() - tick))
            return
        
        for i, (train_idx, val_idx) in enumerate(self.train_val_idxs):
            print("=" * 50)
            print(f"Train/validation num: {len(train_idx)}/{len(val_idx)}")
            kf_train_X = self.train_X_selected[train_idx]
            kf_train_y = self.train_y[train_idx]
            kf_val_X = self.train_X_selected[val_idx]
            kf_val_y = self.train_y[val_idx]
            self.model.fit(kf_train_X, kf_train_y)
            self.models.append(copy.copy(self.model))
            if saved_dir:
                with open(os.path.join(saved_dir, f'{self.model_name}_{i + 1}.pkl'), 'wb+') as f:
                    joblib.dump(self.model, f)
                self.visualize_chem_space(kf_train_X, kf_val_X, saved_dir=saved_dir, method="UMAP", notes=f"fold{i + 1}")
            if self.model_name == "SVC":
                kf_train_pred_prob = self.model.decision_function(kf_train_X)
                kf_train_pred = self.model.predict(kf_train_X)
                kf_val_pred_prob = self.model.decision_function(kf_val_X)
                if len(kf_train_pred_prob.shape) == 1:
                    kf_train_pred_prob = np.stack((-kf_train_pred_prob, kf_train_pred_prob), axis=1)
                    kf_val_pred_prob = np.stack((-kf_val_pred_prob, kf_val_pred_prob), axis=1)
                kf_val_pred = self.model.predict(kf_val_X)
            else:
                kf_train_pred_prob = self.model.predict_proba(kf_train_X)
                kf_train_pred = self.model.predict(kf_train_X)
                kf_val_pred_prob = self.model.predict_proba(kf_val_X)
                kf_val_pred = self.model.predict(kf_val_X)

            train_metrics_all.append(self.cal_metrics(kf_train_y, kf_train_pred, kf_train_pred_prob, n_class=self.n_class))
            val_metrics_all.append(self.cal_metrics(kf_val_y, kf_val_pred, kf_val_pred_prob, n_class=self.n_class) if self.kfold_type != "loo" else [None] * len(self.metrics_list))
            val_pred_all.extend(kf_val_pred)
            val_pred_prob_all.extend(kf_val_pred_prob)
            val_y_all.extend(kf_val_y)

            if hasattr(self, "test_X_selected") and self.test_y is not None:
                if self.model_name == "SVC":
                    test_pred_prob = self.model.decision_function(self.test_X_selected)
                    test_pred = self.model.predict(self.test_X_selected)
                    if len(test_pred_prob.shape) == 1:
                        test_pred_prob = np.stack((-test_pred_prob, test_pred_prob), axis=1)
                else:
                    test_pred_prob = self.model.predict_proba(self.test_X_selected)
                    test_pred = self.model.predict(self.test_X_selected)
                test_pred_all.append(test_pred)
                test_pred_prob_all.append(test_pred_prob)
                test_metrics_all.append(self.cal_metrics(y_true=self.test_y, y_pred=test_pred, y_pred_prob=test_pred_prob, n_class=self.n_class))
        self._aggregate_metrics(train_metrics_all, val_metrics_all, test_metrics_all, val_pred_all, val_pred_prob_all, val_y_all, test_pred_all, saved_dir, test_pred_prob_all)
        print('Total run time:', sec_to_time(time.time() - tick))

    def _aggregate_metrics(self, train_metrics_all, val_metrics_all, test_metrics_all, val_pred_all, val_pred_prob_all, val_y_all, test_pred_all, saved_dir, test_pred_prob_all=None):
        metrics_list = self.metrics_list
        self.train_metrics_df = pd.DataFrame(train_metrics_all, columns=["tr_" + s for s in metrics_list],
                                             index=[f'fold_{i + 1}' for i in range(len(self.train_val_idxs))])
        self.val_metrics_df = pd.DataFrame(val_metrics_all, columns=["val_" + s for s in metrics_list],
                                           index=[f'fold_{i + 1}' for i in range(len(self.train_val_idxs))])
        metrics_dfs = [self.train_metrics_df, self.val_metrics_df]

        self.val_pred_all = np.array(val_pred_all, dtype=np.float32)
        self.val_y_all = np.array(val_y_all, dtype=np.float32)
        self.val_pred_prob_all = np.array(val_pred_prob_all, dtype=np.float32)

        if hasattr(self, "test_X_selected") and self.test_y is not None:
            self.test_metrics_df = pd.DataFrame(test_metrics_all, columns=["te_" + s for s in metrics_list],
                                                index=[f'fold_{i + 1}' for i in range(len(self.train_val_idxs))])
            metrics_dfs.append(self.test_metrics_df)
            ## for classification, test_pred is the predicted class label which appears most frequently in all predictions from different models
            self.test_pred_all = np.array(test_pred_all, dtype=np.int8)
            self.test_pred = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=self.test_pred_all)
            df_te_pred = pd.DataFrame({
                "GroundTruth": self.test_y.squeeze(),
                "Predicted": self.test_pred.squeeze(),
                "Prob": np.mean(test_pred_prob_all, axis=0).tolist()
            })
            if saved_dir:
                test_pred_file = os.path.join(saved_dir, f"test_predicted_{self.model_name}.csv")
                df_te_pred.to_csv(test_pred_file, index=False)

        all_metrics_df = pd.concat(metrics_dfs, axis=1).T
        all_metrics_df['mean'] = all_metrics_df.mean(axis=1)
        if self.kfold_type == "loo":
            val_mean_metric = self.cal_metrics(self.val_y_all, self.val_pred_all, self.val_pred_prob_all, n_class=self.n_class)
            for c_idx, col in enumerate(["val_" + s for s in metrics_list]):
                all_metrics_df.loc[col, 'mean'] = val_mean_metric[c_idx]
        self.all_metrics_df = all_metrics_df.T
        print('*' * 50)
        print("All results for k-fold cross validation: ")
        print(self.all_metrics_df[[col for col in self.all_metrics_df.columns if ("tr" not in col and col.split("_")[-1] in ["MCC", "ACC", "BACC", "AUC", "F1"])]])
    
    def predict(self, X, return_prob=False, cal_feature_distance=False, neighbors_num=1):
        X = np.array(X)
        if len(X.shape) == 1:
            X = X.reshape((1, -1))
        if hasattr(self, "scaler"):
            X = self.scaler.transform(X)
            print("Scaled X shape:", X.shape)
        if hasattr(self, "variance_filter"):
            X = self.variance_filter.transform(X)
            print("Variance filtered X shape:", X.shape)
        if hasattr(self, "selector"):
            X = self.selector.transform(X)
            print("Selected X shape:", X.shape)
        all_y_pred = []
        for model in self.models:
            if self.model_name == "SVC":
                y_pred = model.decision_function(X)
                y_pred = np.stack((-y_pred, y_pred), axis=1)
            else:
                y_pred = model.predict_proba(X)
            all_y_pred.append(y_pred)
        # print(all_y_pred)
        y_pred_mean = np.mean(all_y_pred, axis=0)
        y_pred_mean = y_pred_mean/np.sum(y_pred_mean, axis=1, keepdims=True)  # normalize
        if not return_prob:
            y_pred_mean = y_pred_mean.argmax(axis=1)  # prob -> class_label
            y_pred_mean = y_pred_mean.reshape((-1, 1))
        if cal_feature_distance and hasattr(self, "balltrees"):
            dist, ind = self.balltrees[0].query(X, k=neighbors_num, dualtree=True)
            dist_mean = dist.mean(axis=1).reshape((-1, 1))
            y_pred_mean = y_pred_mean.reshape((-1, 1))
            y_pred_mean = np.hstack([y_pred_mean, dist_mean])
            if hasattr(self, "ref_dist_values"):
                confident_index = self.ref_dist_values[0] / (dist_mean + 0.00001)
                y_pred_mean = np.hstack([y_pred_mean, confident_index])
        return y_pred_mean

    def fulltrain(self, estimator, params, saved_dir=""):
        self.model = estimator
        self.params = copy.deepcopy(params)
        self.model_name = self.model.__class__.__name__
        tick = time.time()
        if isinstance(self.model, GaussianProcessClassifier):
            kernel = ConstantKernel(constant_value=self.params.pop("constant_value")) \
                     * RBF(length_scale=self.params.pop("length_scale")) \
                     + WhiteKernel(noise_level=self.params.pop("noise_level"))
            self.params.update(kernel=kernel)
        self.model.set_params(**self.params)
        if self.valid_X_selected is not None:
            ## concatenate train and valid data for training
            train_X_selected = np.vstack([self.train_X_selected, self.valid_X_selected])
            train_y = np.hstack([self.train_y, self.valid_y])
        print(f"Full training with {len(train_X_selected)} samples")
        self.model.fit(train_X_selected, train_y)
        # self.models = [copy.copy(self.model)]
        if saved_dir:
            with open(os.path.join(saved_dir, f'{self.model_name}_full.pkl'), 'wb+') as f:
                joblib.dump(self.model, f)
        train_pred = self.model.predict(train_X_selected)
        if self.model_name == "SVC":
            train_pred_prob = self.model.decision_function(train_X_selected)
            if len(train_pred_prob.shape) == 1:
                train_pred_prob = np.stack((-train_pred_prob, train_pred_prob), axis=1)
        else:
            train_pred_prob = self.model.predict_proba(train_X_selected)
        train_metrics = self.cal_metrics(train_y, train_pred, train_pred_prob, n_class=self.n_class)
        self.all_metrics_df.loc['full', ["tr_" + s for s in self.metrics_list]] = train_metrics
        if hasattr(self, "test_X_selected") and self.test_y is not None:
            self.test_pred = self.model.predict(self.test_X_selected)
            if self.model_name == "SVC":
                test_pred_prob = self.model.decision_function(self.test_X_selected)
                if len(test_pred_prob.shape) == 1:
                    test_pred_prob = np.stack((-test_pred_prob, test_pred_prob), axis=1)
            else:
                test_pred_prob = self.model.predict_proba(self.test_X_selected)
            self.test_pred_prob = test_pred_prob
            test_metrics = self.cal_metrics(self.test_y, self.test_pred, self.test_pred_prob, n_class=self.n_class)
            self.all_metrics_df.loc['full', ["te_" + s for s in self.metrics_list]] = test_metrics
        print('*' * 50)
        print("All results for full training: ")
        print(self.all_metrics_df[[col for col in self.all_metrics_df.columns if ("tr" not in col and col.split("_")[-1] in ["MCC", "ACC", "BACC", "AUC", "F1"])]])
        print('Total run time:', sec_to_time(time.time() - tick))
        self.full_trained = True

if __name__ == "__main__":
    pass