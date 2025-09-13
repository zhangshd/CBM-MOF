'''
Author: zhangshd
Date: 2024-11-04 17:29:19
LastEditors: zhangshd
LastEditTime: 2024-11-04 17:33:13
'''
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import BallTree
import pickle
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from matplotlib.gridspec import GridSpec
# import umap
import faiss

def dist_penalty(d, max_value=87.3):
    """
    Calculate the distance penalty with numerical stability for float32 precision.
    
    Parameters:
    d (float): Distance.
    max_value (float): Maximum exponent to prevent underflow in float32.
    
    Returns:
    float: Penalty value.
    """
    return np.exp(-d ** 2)

def weighted_average(values, weights):
    """
    Calculate the weighted average of a set of values.

    Parameters:
    values (np.ndarray): Values to average.
    weights (np.ndarray): Corresponding weights.

    Returns:
    float: Weighted average.
    """
    return np.sum(values * weights) / np.sum(weights)

def calculate_lsv_from_tree(tree_dic, latent_vectors_test, k=5):
    """
    Calculate Latent Space Variance (LSV) for regression tasks by considering the labels of the nearest neighbors.

    Parameters:
    tree_dic: dictory object which contains ball tree of training set features, traning set labels, and average distance of k nearest neighbors for training set.
    latent_vectors_test (np.ndarray): Latent vectors of the test data.
    k (int): Number of nearest neighbors to consider.

    Returns:
    np.ndarray: Variance values for each test point.
    """
    # Calculate distances between test points and training points

    tree = tree_dic["tree"]
    labels_train= tree_dic["labels_train"]
    print("searching nearest neighbors..")
    nearest_dists, nearest_neighbors = tree.search(latent_vectors_test, k=k)
    avg_traintrain = tree_dic["avg_dist_traintrian"]
    nearest_dists /= avg_traintrain
    print("calculating variance..")
    variances = []
    for i, neighbors in enumerate(nearest_neighbors):
        # Get the labels of the nearest neighbors
        neighbor_labels = labels_train[neighbors]
        
        # Calculate weights based on distances
        weights = dist_penalty(nearest_dists[i])
        
        # Calculate weighted average of the neighbor labels
        weighted_avg = weighted_average(neighbor_labels, weights)
        try:
            # Calculate variance of neighbor labels as a measure of uncertainty
            variance = np.average((neighbor_labels - weighted_avg) ** 2, weights=weights)
        except ZeroDivisionError:
            print(f"ZeroDivisionError: {i}")
            print(f"Nearest dists: {nearest_dists[i]}")
            raise ZeroDivisionError
        
        variances.append(variance)
    variances = np.array(variances)
    if "scaler" in tree_dic:
        variances = tree_dic["scaler"].transform(variances.reshape(-1, 1)).reshape(-1)
    return variances

def plot_tsne_for_task_by_targets(task, latent_feas, targets, ax, targets_map=None, size=5, **kwargs):
    all_latent_feas = []
    all_targets = []
    all_splits = []
    x_label = kwargs.get('x_label', True)
    y_label = kwargs.get('y_label', True)
    cb_pad = kwargs.get('cb_pad', 0.05)
    cb_orientation = kwargs.get('cb_orientation', 'vertical')
    cb_aspect = kwargs.get('cb_aspect', 10)
    cb_title = kwargs.get('cb_title', 'Label Value')
    tick_font_size = kwargs.get('tick_font_size', 12)
    label_font_size = kwargs.get('label_font_size', 12)
    title = kwargs.get('title', f"Last-Layer Features\n{task}")

    # Concatenate all latent features and targets for the task
    for split in ['train', 'val', 'test']:
        key = f"{task}_{split}"
        if key in latent_feas:
            all_latent_feas.append(latent_feas[key])
            all_targets.append(targets[key])
            all_splits.extend([split] * len(latent_feas[key]))

    all_latent_feas = np.concatenate(all_latent_feas, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # # Perform TSNE
    # tsne = TSNE(n_components=2, random_state=42, n_jobs=-1)
    # decompose_results = tsne.fit_transform(all_latent_feas)
    # decompose_cols = ['TSNE1', 'TSNE2']

    pca = PCA(n_components=2, random_state=42)
    decompose_results = pca.fit_transform(all_latent_feas)
    decompose_cols = ['PC1', 'PC2']

    # umap_model = umap.UMAP(n_components=2, n_jobs=-1)  # 将数据降到2维
    # decompose_results = umap_model.fit_transform(all_latent_feas)
    # decompose_cols = ['UMAP1', 'UMAP2']

    # Create a DataFrame for visualization
    decompose_df = pd.DataFrame(decompose_results, columns=decompose_cols)
    decompose_df['Label'] = all_targets
    decompose_df['Split'] = all_splits
    decompose_df['Size'] = [size] * len(decompose_df)

    # Determine if the task is classification or regression
    unique_targets = np.unique(all_targets)
    is_classification = len(unique_targets) < 20

    # Plot the results
    print (f"Number of samples: {len(decompose_df)}")
    if is_classification:
        if targets_map is not None:
            decompose_df['Label'] = decompose_df['Label'].apply(lambda x: targets_map[task][int(x)])
        sns.scatterplot(
            x=decompose_cols[0], y=decompose_cols[1],
            hue='Label',
            style='Label',
            sizes="Size",
            palette=sns.color_palette("hsv", len(unique_targets), desat=0.6),
            data=decompose_df,
            alpha=kwargs.get('alpha', 0.5),
            markers=True,
            ax=ax
        )
        if x_label:
            ax.set_xlabel(decompose_cols[0], fontsize=label_font_size)
        if y_label:
            ax.set_ylabel(decompose_cols[1], fontsize=label_font_size)
        ax.legend(loc='best', fontsize=tick_font_size)
        
    else:
        norm = Normalize(vmin=all_targets.min(), vmax=all_targets.max())
        cmap = plt.get_cmap('rainbow')
        scatter = ax.scatter(
            decompose_df[decompose_cols[0]], decompose_df[decompose_cols[1]],
            c=all_targets, cmap=cmap, norm=norm,
            alpha=kwargs.get('alpha', 0.5),
            s=kwargs.get('size', 5)
        )
        if x_label:
            ax.set_xlabel(decompose_cols[0], fontsize=label_font_size)
        if y_label:
            ax.set_ylabel(decompose_cols[1], fontsize=label_font_size)
        cbar = plt.colorbar(scatter, orientation=cb_orientation, pad=cb_pad, ax=ax, aspect=cb_aspect)
        cbar.set_label(cb_title, fontsize=label_font_size+1)

    ax.set_title(title, weight='bold', fontsize=label_font_size+2)
    ax.tick_params(axis='both', labelsize=tick_font_size)
    ax.xaxis.label.set_size(label_font_size)
    ax.yaxis.label.set_size(label_font_size)
    
    return ax

def lsv_analysis(task, latent_feas, uncertainty_trees, targets, predictions, k=10, **kwargs):

    avg_distances = calculate_lsv_from_tree(uncertainty_trees[task], 
                                latent_feas[f"{task}_test"], k=k)
    
    alpha = kwargs.get('alpha', 0.8)
    xmax = kwargs.get('xmax', None)
    ax = kwargs.get('ax', None)
    frac_cutoff = kwargs.get('frac_cutoff', 0.8)
    x_label = kwargs.get('x_label', True)
    y_label_left = kwargs.get('y_label_left', True)
    y_label_right = kwargs.get('y_label_right', True)
    legend = kwargs.get('legend', False)
    tick_font_size = kwargs.get('tick_font_size', 12)
    label_font_size = kwargs.get('label_font_size', 12)
    title = kwargs.get('title', f'MAE & Fraction vs LSV Cutoff ({task})')

    cmap = plt.get_cmap("rainbow")
    colors = [cmap(i / 9) for i in range(10)]

    if ax is None:
        fig, ax = plt.subplots()

    # num_classes = len(np.unique(targets[f"{task}_train"]))
    scaler = MinMaxScaler()
    scaled_avg_knn_dist = scaler.fit_transform(avg_distances.reshape(-1, 1)).reshape(-1)

    cutoffs = []
    performances_in = []
    performances_out = []
    fracs = []
    r2_scores_in = []
    step = 0.001
    for i in range(1, 996, 1):
        sub_targets = targets[f"{task}_test"][np.where(scaled_avg_knn_dist <= step*i)[0]]
        sub_preds = predictions[f"{task}_test"][np.where(scaled_avg_knn_dist <= step*i)[0]]
        sub_targets_out = targets[f"{task}_test"][np.where(scaled_avg_knn_dist > step*i)[0]]
        sub_preds_out = predictions[f"{task}_test"][np.where(scaled_avg_knn_dist > step*i)[0]]
        cutoffs.append(step*i)
        performances_in.append(metrics.mean_absolute_error(sub_targets, sub_preds))
        performances_out.append(metrics.mean_absolute_error(sub_targets_out, sub_preds_out))
        r2_scores_in.append(metrics.r2_score(sub_targets, sub_preds))
        frac = len(sub_targets)/len(targets[f"{task}_test"])
        fracs.append(frac)
        if frac > frac_cutoff:
            print (f"LSV cutoff={step*i}, MAE={metrics.mean_absolute_error(sub_targets, sub_preds)}, R2={metrics.r2_score(sub_targets, sub_preds)}")
            break

    metric_full = metrics.mean_absolute_error(targets[f"{task}_test"], predictions[f"{task}_test"])
    sns.scatterplot(x=cutoffs, y=performances_in, ax=ax, marker='D', label='MAE of samples inside cutoff', alpha=alpha, color=colors[0])
    sns.scatterplot(x=cutoffs, y=performances_out, ax=ax, marker='^', label='MAE of samples outside cutoff', alpha=alpha, color=colors[1])
    if x_label:
        ax.set_xlabel('Uncertainty Cutoff')
    if y_label_left:
        ax.set_ylabel('Mean Absolute Error')
    ax.set_ylim(-0.01, 0.51)
    

    ax2 = ax.twinx()
    sns.scatterplot(x=cutoffs, y=fracs, ax=ax2, color=colors[2], marker='o', label='Retained data fraction', alpha=alpha)
    # ax2.vlines(x=cutoff, ymin=0, ymax=1.2, colors='gray', linestyles='dashed', label=f'LSD Cutoff={cutoff:.2f}', alpha=alpha)
    if y_label_right:
        ax2.set_ylabel('Fraction')
    ax2.set_ylim(-0.01, 1.05)

    if xmax is None:
        xmax = cutoffs[-1]
    ax.set_xlim(0, xmax)
    ax.hlines(y=metric_full, xmin=0, xmax=xmax, colors=colors[-3], linestyles='dashed', label=f'MAE of full test set', alpha=alpha)
    ax2.text(xmax*0.65, 0.6, "\n".join([  
                                          
                                          f'MAE={performances_in[-1]:.3f}',
                                          "R$^2$" + f'={r2_scores_in[-1]:.3f}',
                                          'when:',
                                          'LSV$_{cutoff}$=' + f'{cutoffs[-1]:.3f}',
                                          ]), 
                                         fontsize=tick_font_size, ha='left', va='center', color=colors[-1])
    ax.text(xmax*0.65, metric_full+0.02, "MAE$_{full}$" + f"={metric_full:.3f}", fontsize=tick_font_size, ha='left', va='center', color=colors[-3])
    if legend:
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        # ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        # ax2.legend(loc='lower right', bbox_to_anchor=(1, 1))
        ax.legend(handles1 + handles2, labels1 + labels2, loc='upper left')
        ax2.legend().set_visible(False)
    else:
        ax.legend().set_visible(False)
        ax2.legend().set_visible(False)
    # fig.suptitle(f'MAE & Fraction vs Cutoff ({task})')
    ax.set_title(title, weight='bold', fontsize=label_font_size)
    ax.tick_params(axis='both', labelsize=tick_font_size)
    ax.xaxis.label.set_size(label_font_size)
    ax.yaxis.label.set_size(label_font_size)
    ax2.tick_params(axis='both', labelsize=tick_font_size)
    ax2.yaxis.label.set_size(label_font_size)
    return ax