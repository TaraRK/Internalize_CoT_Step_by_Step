#%%
import matplotlib.pyplot as plt
import re
import pandas as pd
from IPython.display import display

# Read and parse the log file
log_path = "trained_models/launch2/log.train"
steps = []
ppls = []
accuracies = []
epochs = []  # Store epoch transition points
epoch_numbers = []  # Store epoch numbers

with open(log_path, 'r') as f:
    current_epoch = 0
    for line in f:
        # Check for PPL/accuracy metrics
        match = re.match(r'Step: (\d+)\. PPL: ([\d\.]+)\. Token Accuracy: ([\d\.]+)', line)
        if match:
            steps.append(int(match.group(1)))
            ppls.append(float(match.group(2)))
            accuracies.append(float(match.group(3)))
            epoch_numbers.append(current_epoch)
            
        # Check for epoch transitions
        epoch_match = re.match(r'.*-epoch (\d+)\.', line)
        if epoch_match and len(steps) > 0:  # Only add if we have data points
            epochs.append(len(steps)-1)  # Store index of last step before new epoch
            current_epoch = int(epoch_match.group(1))

# Create high resolution figure with two y-axes
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
fig, ax1 = plt.subplots(figsize=(15,4))
ax2 = ax1.twinx()

# # Increase font sizes
# plt.rcParams.update({'font.size': 28})
# plt.rcParams['axes.titlesize'] = 32
# plt.rcParams['axes.labelsize'] = 28
# plt.rcParams['xtick.labelsize'] = 24
# plt.rcParams['ytick.labelsize'] = 24
# plt.rcParams['legend.fontsize'] = 24

# Plot PPL with higher line width
p1 = ax1.plot(steps, ppls, 'b-', label='PPL', linewidth=2)
ax1.set_ylabel('Perplexity', color='b')
ax1.tick_params(axis='y', labelcolor='b')

# Plot accuracy with higher line width
p2 = ax2.plot(steps, accuracies, 'r-', label='Accuracy', linewidth=2)
ax2.set_ylabel('Token Accuracy', color='r')
ax2.tick_params(axis='y', labelcolor='r')

# Add second x-axis for epochs
ax3 = ax1.twiny()
ax3.set_xlim(ax1.get_xlim())
ax3.set_xticks(steps[::len(steps)//10])  # Show ~10 epoch ticks
ax3.set_xticklabels([f"Epoch: {n}" if n == 0 else f"{n}" for n in epoch_numbers[::len(steps)//10]], ha='right')

# Add vertical lines for epoch transitions with higher line width, but only every 2 epochs
for i, epoch_idx in enumerate(epochs):
    if i % 2 == 0:  # Only draw line for even-numbered epochs
        ax1.axvline(x=steps[epoch_idx], color='gray', linestyle='--', alpha=0.5, linewidth=1.5)

# Add legend
lines = p1 + p2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center right')

ax1.set_xlabel('Training Steps (each epoch has 25000 steps)')
plt.title('Model Performance Evaluation during Training', pad=20)

# Ensure layout is tight to avoid text cutoff
# plt.tight_layout()
plt.show()

# Display summary statistics
df = pd.DataFrame({
    'Step': steps,
    'PPL': ppls, 
    'Accuracy': accuracies,
    'Epoch': epoch_numbers
})

print("\nSummary Statistics:")
display(df.describe())

# Display last few entries
print("\nLast few training steps:")
display(df.tail())


# %%
