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
def get_linear_probe_scores(reps, labels, split=0.8, fit_type='linear'):
    if fit_type == 'logistic': 
        state_probe = LogisticRegression(max_iter=1000)
    elif fit_type == 'lasso': 
        state_probe = Lasso(alpha=0.1)
    elif fit_type == 'ridge': 
        state_probe = Ridge(alpha=0.1)
    elif fit_type == 'linear': 
        state_probe = LinearRegression()
    shuffle_idx = np.random.permutation(len(reps))
    reps = reps[shuffle_idx]
    labels = labels[shuffle_idx]
    num_train = int(split * len(reps))
    train_reps, test_reps = reps[:num_train,:], reps[num_train:,:]  
    # # Check for infinite or very large values in labels
    # mask = np.isfinite(labels) & (np.abs(labels) < 1e10)
    # if not np.all(mask):
    #     print("Invalid values found:")
    #     print("Labels with inf:", np.sum(~np.isfinite(labels)))
    #     print("Labels with large values:", np.sum(np.abs(labels) >= 1e10))
    #     print("Sample of invalid labels:", labels[~mask][:5])
    #     print(f"Warning: Found {np.sum(~mask)} invalid values in labels. Filtering them out.")
    #     reps = reps[mask]
    #     labels = labels[mask]
    #     num_train = int(split * len(reps))

    train_states, test_states = labels[:num_train], labels[num_train:]
    state_probe.fit(train_reps, train_states)
    return state_probe.score(test_reps, test_states), state_probe

#%% 
layerwise_type_scores = {
    score_type: [] for score_type in probe_labels_linear
}
probes = { 
    score_type: [] for score_type in probe_labels_linear
}
for layer in tqdm(layer_idxs):
    for score_type in probe_labels_linear:
        score, probe = get_linear_probe_scores(
            all_activations[layer], probe_labels_linear[score_type], fit_type='lasso'
        )
        probes[score_type].append(probe)
        layerwise_type_scores[score_type].append(score)
        print(f"Layer {layer} {score_type} score: {score}")
#%% 
for score_type in probe_labels_linear:
    print(score_type)
    for activation, label in zip(all_activations[-1][:10], probe_labels_linear[score_type][:10]):
        print(probes[score_type][-1].predict(activation.reshape(1, -1)), label)
#%% 
for score_type in layerwise_type_scores:
    print(score_type)
    print(layerwise_type_scores[score_type][-5:])
#%%
plt.figure(figsize=(15,8))
plt.rcParams.update({'font.size': 24})
plt.rcParams['axes.titlesize'] = 28
plt.rcParams['axes.labelsize'] = 24
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 20

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for score_type in layerwise_type_scores:
    plt.plot(layerwise_type_scores[score_type], label=score_type)
plt.xticks(range(len(layer_idxs)), [layer_names[idx] for idx in layer_idxs], rotation=45)
ax.grid(linestyle='--', alpha=0.4)
plt.ylim(0, 1)  # Set y-axis limits from 0 to 1
plt.title(r'Linear Probe $R^2$ Scores Across Model Layers', pad=20)
plt.legend()
plt.tight_layout()
plt.show()
#%% 
layerwise_type_scores_logistical = {
    score_type: [] for score_type in probe_labels_logistical
} 
for layer in tqdm(layer_idxs):
    for score_type in probe_labels_logistical:
        score, _ = get_linear_probe_scores(
            all_activations[layer], probe_labels_logistical[score_type], fit_type='logistic'
        )
        layerwise_type_scores_logistical[score_type].append(score)
        print(f"Layer {layer} {score_type} score: {score}")
#%% 
plt.figure(figsize=(15,10))
for score_type in layerwise_type_scores_logistical:
    plt.plot(layerwise_type_scores_logistical[score_type], label=score_type)
plt.xticks(range(len(layer_idxs)), [layer_names[idx] for idx in layer_idxs], rotation=90)
plt.gca().xaxis.grid(True)
plt.ylim(0, 1)  # Set y-axis limits from 0 to 1
plt.legend()
plt.show()

#%% 
layerwise_type_scores_log = {
    score_type: [] for score_type in probe_labels_log
}
for layer in tqdm(layer_idxs):
    for score_type in probe_labels_log:
        score, _ = get_linear_probe_scores(
            all_activations[layer], probe_labels_log[score_type], fit_type='ridge'
        )
        layerwise_type_scores_log[score_type].append(score)
        print(f"Layer {layer} {score_type} score: {score}")

plt.figure(figsize=(15,10))
for score_type in layerwise_type_scores_log:
    plt.plot(layerwise_type_scores_log[score_type], label=score_type)
plt.xticks(range(len(layer_idxs)), [layer_names[idx] for idx in layer_idxs], rotation=90)
plt.gca().xaxis.grid(True)
plt.ylim(0, 1)  # Set y-axis limits from 0 to 1
plt.legend()
plt.show()
#%% 
layerwise_type_scores_is_digit_x = {
}
for layer in tqdm(layer_idxs):
    for xth_digit in range(len(probe_labels_is_digit_x)):
        for num in range(10):
            score, _ = get_linear_probe_scores(
                all_activations[layer], probe_labels_is_digit_x[xth_digit, num], fit_type='logistical'
            )
            if f"{xth_digit}th Digit {num}" not in layerwise_type_scores_is_digit_x:
                layerwise_type_scores_is_digit_x[f"{xth_digit}th Digit {num}"] = []
            layerwise_type_scores_is_digit_x[f"{xth_digit}th Digit {num}"].append(score)
            print(f"Layer {layer} {score_type} score: {score}")
#%% 
plt.figure(figsize=(15,10))
for score_type in layerwise_type_scores_is_digit_x:
    plt.plot(layerwise_type_scores_is_digit_x[score_type], label=score_type)
plt.xticks(range(len(layer_idxs)), [layer_names[idx] for idx in layer_idxs], rotation=90)
plt.gca().xaxis.grid(True)
plt.ylim(0, 1)  # Set y-axis limits from 0 to 1
plt.legend()
plt.show()

#%% 


#%% 
# ## Hacky normalization
# outputs_min = outputs.min()
# outputs_max = outputs.max()
# outputs_normalized = (outputs - outputs_min) / (outputs_max - outputs_min) * 100


# X_train, X_test, y_train, y_test = train_test_split(
#     activations, outputs_normalized, test_size=0.1, random_state=42
# )


# X_train = torch.tensor(X_train, dtype=torch.float32)
# y_train = torch.tensor(y_train, dtype=torch.long)  
# X_test = torch.tensor(X_test, dtype=torch.float32)
# y_test = torch.tensor(y_test, dtype=torch.long)

# y_train = y_train.unsqueeze(1).float()
# y_test = y_test.unsqueeze(1).float()


# train_dataset = TensorDataset(X_train, y_train)
# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# test_dataset = TensorDataset(X_test, y_test)
# test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# # Define the linear model
# class LinearModel(nn.Module):
#     def __init__(self):
#         super(LinearModel, self).__init__()
#         self.linear = nn.Linear(768, 20)   
#         self.output = nn.Linear(20, 1)     

#     def forward(self, x):
#         x = self.linear(x)
#         x = torch.relu(x)  # Activation function
#         x = self.output(x)
#         return x

# model = LinearModel()


# criterion = nn.MSELoss()  # Mean Squared Error for regression
# optimizer = optim.Adam(model.parameters(), lr=0.001)


# initial_lr = 0.01

# num_epochs = 10000
# best_loss = float('inf')
# plateau_count = 0
# current_lr = initial_lr


# train_losses = []
# test_losses = []

# for epoch in range(1, num_epochs + 1):
#     model.train()
#     total_loss = 0
#     for inputs, targets in train_loader:
#         optimizer.zero_grad()
#         outputs_pred = model(inputs)
#         loss = criterion(outputs_pred, targets)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
    
#     avg_loss = total_loss / len(train_loader)
#     train_losses.append(avg_loss)
    
#     if epoch % 100 == 0:
#         model.eval()
#         total_test_loss = 0
#         with torch.no_grad():
#             for inputs, targets in test_loader:
#                 outputs_pred = model(inputs)
#                 loss = criterion(outputs_pred, targets)
#                 total_test_loss += loss.item()
#         avg_test_loss = total_test_loss / len(test_loader)
#         test_losses.append(avg_test_loss)
        
#         # Check if loss has improved
#         if avg_test_loss < best_loss:
#             best_loss = avg_test_loss
#             plateau_count = 0
#         else:
#             plateau_count += 1
            
#         if plateau_count >= 1:  # Since we're checking every 100 epochs, this means no improvement for 100 epochs
#             current_lr = current_lr / 2
#             for param_group in optimizer.param_groups:
#                 param_group['lr'] = current_lr
#             print(f"Learning rate reduced to {current_lr}")
#             plateau_count = 0  # Reset counter
            
#         print(f"Epoch {epoch}/{num_epochs}, Train Loss: {avg_loss:.4f}, Test Loss: {avg_test_loss:.4f}, LR: {current_lr:.6f}")

# %%
