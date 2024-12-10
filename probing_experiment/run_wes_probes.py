#%%
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import glob
import itertools as it
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
import os
import probing_experiment.wes_regression_probes as regression_probes
import probing_experiment.wes_classification_probes as classification_probes

from importlib import reload
reload(regression_probes)
reload(classification_probes)

#%%
activations_dir = 'cached_activations/final/'

probe_labels_logistical = np.load('probe_labels_logistical.npy', allow_pickle=True).item() # Shape: (1000,)
probe_labels_linear = np.load('probe_labels_linear.npy', allow_pickle=True).item() # Shape: (1000,)
probe_labels_log = np.load('probe_labels_log.npy', allow_pickle=True).item() # Shape: (1000,)
probe_labels_is_digit_x = np.load('probe_labels_is_digit_x.npy', allow_pickle=True) # Shape: (1000, 8, 10)
suffix = "_pred_token_2"

all_activations = [np.load(os.path.join(activations_dir, f"embedding{suffix}.npy"))]
for i in range(12):
    activations = np.load(os.path.join(activations_dir, f"transformer_layer_{i}{suffix}.npy"))  # Shape: (1000, 768)
    all_activations.append(activations)

all_activations = np.array(all_activations)

print(all_activations.shape)
layer_names = ['embed'] + list(it.chain(*[
    [f"layer_{i}_resid"]
    for i in range(12)
]))
layer_idxs = list(range(len(layer_names)))

#%% 

# layerwise_type_scores = {
#     score_type: [] for score_type in probe_labels_linear
# }
# probes = { 
#     score_type: [] for score_type in probe_labels_linear
# }
layer_results = { 
    score_type: [] for score_type in probe_labels_linear
}
for layer in tqdm(layer_idxs):
    for score_type in probe_labels_linear:
        results = regression_probes.heuristic_sparse_regression_sweep(
            all_activations[layer], probe_labels_linear[score_type], 
        )
        layer_results[score_type].append(results)
#%% 
# Plot R2 scores across layers for different k values
# Get list of k values from first layer/score type results
def plot_results(layer_results):
    k_values = list(layer_results[list(layer_results.keys())[0]][0].keys())

    for score_type in layer_results.keys():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Extract R2 scores and MSE for each k value across layers
        r2_by_k = {k: [] for k in k_values}
        mse_by_k = {k: [] for k in k_values}
        
        for layer_result in layer_results[score_type]:
            for k in k_values:
                r2_by_k[k].append(layer_result[k]['r2'])
                mse_by_k[k].append(layer_result[k]['mean_squared_error'])
        
        # Plot R2 scores
        for k in k_values:
            ax1.plot(layer_idxs, r2_by_k[k], label=f'{score_type} (k={k})', alpha=0.7)

        ax1.set_xlabel('Layer')
        ax1.set_ylabel('R² Score')
        ax1.set_title('R² Score vs Layer')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)

        # Plot MSE
        for k in k_values:
            ax2.plot(layer_idxs, mse_by_k[k], label=f'{score_type} (k={k})', alpha=0.7)

        ax2.set_xlabel('Layer')
        ax2.set_ylabel('Mean Squared Error')
        ax2.set_title('Mean Squared Error vs Layer')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)

        plt.suptitle('Probe Performance vs Layer for Different k Values')
        plt.tight_layout()
        plt.show()

plot_results(layer_results)
#%% 
layer_results = { 
    score_type: [] for score_type in probe_labels_log
}

for layer in tqdm(layer_idxs):
    for score_type in probe_labels_log:
        results = regression_probes.heuristic_sparse_regression_sweep(
            all_activations[layer], probe_labels_log[score_type], 
        )
        layer_results[score_type].append(results)
plot_results(layer_results)


#%% 
layer_results = { 
    score_type: [] for score_type in probe_labels_linear
}
for layer in tqdm(layer_idxs):
    for score_type in probe_labels_linear:
        results = regression_probes.optimal_sparse_regression_probe(
            all_activations[layer], probe_labels_linear[score_type], 
        )
        layer_results[score_type].append(results)
    #     break
    # break
#%% 

layer_results
#%% 
layer_results = {
    score_type: [] for score_type in probe_labels_logistical
}

for layer in tqdm(layer_idxs):
    for score_type in probe_labels_logistical:
        results = classification_probes.optimal_sparse_probing(
            all_activations[layer], probe_labels_logistical[score_type], 
        )
        layer_results[score_type].append(results)
        break
    break

print(layer_results)
# %%
layer_results.keys()
# %%
for key in layer_results.keys():
    print(key)
    print(layer_results[key])
# %%

all_activations[0].shape, probe_labels_linear['Output'].shape
# %%
