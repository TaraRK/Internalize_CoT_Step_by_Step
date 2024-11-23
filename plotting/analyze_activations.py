# # # analyze_activations.py
#%% 

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import seaborn as sns
import argparse
from typing import Dict, List, Tuple
import plotly.graph_objects as go
import sys

#%% 

def extract_labels_from_expressions(expressions_file: str) -> Tuple[List[str], np.ndarray]:
    """Extract sequences and convert them to numbers"""
    full_labels = []
    numeric_values = []
    
    with open(expressions_file, 'r') as f:
        for line in f:
            parts = line.strip().split('#')
            parts2 = line.strip().split('||')
            if len(parts) > 1 and len(parts2) > 1:
                input = parts2[0].strip()[::-1]
                output = parts[-1].strip()[::-1]
                full_labels.append(f"{input} || {output}")
                # Convert space-separated sequence to single number
                num = int(''.join(output.split()))
                numeric_values.append(num)
    return full_labels, np.array(numeric_values)

full_labels, numeric_values = extract_labels_from_expressions("data/4_by_4_mult/test_bigbench.txt")
print(full_labels)
#%% 
def load_activations_with_labels(activation_dir: str, expressions_file: str):
    """Load activations from final folder and their corresponding labels"""
    full_labels, numeric_values = extract_labels_from_expressions(expressions_file)
    print(f"Loaded {len(full_labels)} labels")
    
    final_dir = os.path.join(activation_dir, "final")
    if not os.path.exists(final_dir):
        raise ValueError(f"Final directory not found at {final_dir}")
        
    activations = {}
    for filename in os.listdir(final_dir):
        if not filename.endswith('_first_pred.npy'):
            continue
            
        layer_name = filename.replace('_first_pred.npy', '')
        filepath = os.path.join(final_dir, filename)
        
        print(f"Loading activations from {filepath}")
        data = np.load(filepath)
        print(f"Loaded activation shape: {data.shape}")
        
        if len(data) > len(full_labels):
            data = data[:len(full_labels)]
            print(f"Trimmed to {len(data)} samples to match labels")
        
        activations[layer_name] = data
    
    return activations, full_labels, numeric_values

def create_scatter_plot(transformed, numeric_values, full_labels, layer_name, save_dir, title, xaxis_title, yaxis_title):
    """Create interactive scatter plot with plotly"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=transformed[:, 0],
        y=transformed[:, 1],
        mode='markers',
        marker=dict(
            size=8,
            color=numeric_values,
            colorscale='Viridis',
            colorbar=dict(title='Numeric Value'),
            opacity=0.6
        ),
        text=full_labels,  # Add hover text showing full labels
        hovertemplate='<b>Value:</b> %{marker.color}<br>' +
                     '<b>Label:</b> %{text}<br>' +
                     '<b>{xaxis_title}:</b> %{x:.2f}<br>' +
                     '<b>{yaxis_title}:</b> %{y:.2f}<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title=f'{layer_name} - {title}',
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        width=1000,
        height=800,
        template='plotly_white',  # Clean white background with grid
        xaxis=dict(
            tickfont=dict(size=24),
            titlefont=dict(size=28)
        ),
        yaxis=dict(
            tickfont=dict(size=24),
            titlefont=dict(size=28)
        ),
        title_font_size=32
    )
    
    # Show plot interactively
    fig.show()
    fig.write_image(os.path.join(save_dir, f'{layer_name}_{title.lower()}.png'), scale=2)

def analyze_pca_with_math_labels(
    activations: Dict[str, np.ndarray], 
    full_labels: List[str],
    numeric_values: np.ndarray,
    save_dir: str = 'pca_results_labels'
):
    """Perform PCA analysis with scatter plots colored by numeric value"""
    os.makedirs(save_dir, exist_ok=True)
    
    for layer_name, acts in activations.items().sorted():
        print(f"\nAnalyzing {layer_name}...")
        
        # Perform PCA
        pca = PCA(n_components=2)
        transformed = pca.fit_transform(acts)
        
        create_scatter_plot(transformed, numeric_values, full_labels, layer_name, save_dir, 'PCA Projection', f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')

def analyze_tsne_with_math_labels(
    activations: Dict[str, np.ndarray], 
    full_labels: List[str],
    numeric_values: np.ndarray,
    save_dir: str = 'tsne_results_labels'
):
    """Perform t-SNE analysis with scatter plots colored by numeric value"""
    os.makedirs(save_dir, exist_ok=True)
    
    for layer_name, acts in activations.items():
        print(f"\nAnalyzing {layer_name}...")
        
        # Perform t-SNE
        tsne = TSNE(n_components=2)
        transformed = tsne.fit_transform(acts)
        
        create_scatter_plot(transformed, numeric_values, full_labels, layer_name, save_dir, 't-SNE Projection', 't-SNE Component 1', 't-SNE Component 2')

def main():
    # Check if we're in a notebook environment
    in_notebook = 'ipykernel' in sys.modules
    
    if not in_notebook:
        parser = argparse.ArgumentParser()
        parser.add_argument('--activation_dir', type=str, required=True,
                          help='Directory containing activation files')
        parser.add_argument('--expressions_file', default='data/4_by_4_mult/test_bigbench.txt', type=str,
                          help='File containing math expressions with labels')
        parser.add_argument('--save_dir', type=str, default='pca_results_labels_2',
                          help='Directory to save PCA results')
        parser.add_argument('--visualization', type=str, default='both',
                          choices=['scatter', 'density', 'both'],
                          help='Type of visualization to generate')
        parser.add_argument('--clustering', type=str, default='pca',
                          choices=['pca', 'tsne'],
                          help='Clustering algorithm to use')
        args = parser.parse_args()
    else:
        # Default values for notebook environment
        class Args:
            def __init__(self):
                self.activation_dir = "cached_activations"
                self.expressions_file = "data/4_by_4_mult/test_bigbench.txt"
                self.save_dir = "pca_results_labels_2"
                self.visualization = "both"
                self.clustering = "pca"
        args = Args()
    activations, full_labels, numeric_values = load_activations_with_labels(
        args.activation_dir, 
        args.expressions_file
    )
    
    if args.clustering == 'pca':
        analyze_pca_with_math_labels(activations, full_labels, numeric_values, args.save_dir)
    elif args.clustering == 'tsne':
        analyze_tsne_with_math_labels(activations, full_labels, numeric_values, args.save_dir)
    else:
        raise ValueError(f"Invalid clustering algorithm: {args.clustering}")
    
    print(f"\nResults saved to {args.save_dir}")

if __name__ == "__main__":
    main()


# %%
