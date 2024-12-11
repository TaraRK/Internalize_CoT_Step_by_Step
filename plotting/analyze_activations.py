# # # # analyze_activations.py
# #%% 

# import numpy as np
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
# import os
# # import seaborn as sns
# import argparse
# from typing import Dict, List, Tuple
# import plotly.graph_objects as go
# import sys
# from plotly.subplots import make_subplots

# #%% 

# #def extract_labels_from_expressions(expressions_file: str) -> Tuple[List[str], np.ndarray]:
# #     """Extract sequences and convert them to numbers"""
# #     full_labels = []
# #     numeric_values = []
    
# #     with open(expressions_file, 'r') as f:
# #         for line in f:
# #             parts = line.strip().split('#')
# #             parts2 = line.strip().split('||')
# #             if len(parts) > 1 and len(parts2) > 1:
# #                 input = parts2[0].strip()[::-1]
# #                 output = parts[-1].strip()[::-1]
# #                 full_labels.append(f"{input} || {output}")
# #                 # Convert space-separated sequence to single number
# #                 num = int(''.join(output.split()))
# #                 numeric_values.append(num)
# #     return full_labels, np.array(numeric_values)
# # def extract_multiple_labels(expressions_file: str, pred_token_idx: int) -> Tuple[List[str], Dict[str, np.ndarray]]:
# #     """Extract sequences and multiple types of labels"""
# #     full_labels = []
# #     label_types = {
# #         'first_input1': [],
# #         'last_input1': [],
# #         'first_input2': [],
# #         'last_input2': [],
# #         'first_output': [],
# #         'last_output': []
# #     }
    
# #     with open(expressions_file, 'r') as f:
# #         for line in f:
# #             if '####' in line:
# #                 parts = line.strip().split('####')
# #                 if len(parts) == 2:
# #                     input_part = parts[0].strip()
# #                     output_part = parts[1].strip()
                    
# #                     # Split the input part by || to get input1 and input2
# #                     input_parts = input_part.split('||')
# #                     if len(input_parts) == 2:
# #                         input1 = input_parts[0].strip()
# #                         input2_complex = input_parts[1].strip()
                        
# #                         # Get first number from input2 (before any operators)
# #                         input2_nums = [x for x in input2_complex.split() if x.isdigit()]
# #                         if input2_nums:
# #                             input2 = input2_nums[0]
# #                         else:
# #                             continue
                            
# #                         # Extract numbers
# #                         input1_nums = [x for x in input1.split() if x.isdigit()]
# #                         output_nums = output_part.split()
                        
# #                         if input1_nums and output_nums:
# #                             full_labels.append(f"{input_part} #### {output_part}")
                            
# #                             # Store different label types
# #                             label_types['first_input1'].append(int(input1_nums[0]))
# #                             label_types['last_input1'].append(int(input1_nums[-1]))
# #                             label_types['first_input2'].append(int(input2[0]))
# #                             label_types['last_input2'].append(int(input2[-1]))
# #                             label_types['first_output'].append(int(output_nums[0]))
# #                             label_types['last_output'].append(int(output_nums[-1]))
    
# #     return full_labels, {k: np.array(v) for k, v in label_types.items()}


# # def load_activations_with_multiple_labels(activation_dir: str, expressions_file: str, pred_token_idx: int):
# #     """Load activations and multiple types of labels"""
# #     full_labels, label_types = extract_multiple_labels(expressions_file, pred_token_idx)
# #     print(f"Loaded {len(full_labels)} examples")
    
# #     final_dir = os.path.join(activation_dir, "final")
# #     if not os.path.exists(final_dir):
# #         raise ValueError(f"Final directory not found at {final_dir}")
        
# #     activations = {}
# #     for filename in os.listdir(final_dir):
# #         if not filename.endswith(f'_pred_token_{pred_token_idx}.npy'):
# #             continue
            
# #         layer_name = filename.replace(f'_pred_token_{pred_token_idx}.npy', '')
# #         filepath = os.path.join(final_dir, filename)
        
# #         print(f"Loading activations from {filepath}")
# #         data = np.load(filepath)
# #         print(f"Loaded activation shape: {data.shape}")
        
# #         if len(data) > len(full_labels):
# #             data = data[:len(full_labels)]
# #             print(f"Trimmed to {len(data)} samples to match labels")
        
# #         activations[layer_name] = data
    
# #     return activations, full_labels, label_types


# # def create_multiple_scatter_plots(transformed, label_types, full_labels, layer_name, save_dir, pca_variance_ratio):
# #     """Create a single figure with subplots for different label types using the same PCA projection"""
# #     # Create a 2x3 subplot figure
# #     fig = make_subplots(
# #         rows=2, cols=3,
# #         subplot_titles=[label_type.replace('_', ' ').title() for label_type in label_types.keys()],
# #         vertical_spacing=0.12,
# #         horizontal_spacing=0.1
# #     )

# #     # Calculate subplot positions
# #     positions = [(1,1), (1,2), (1,3), (2,1), (2,2), (2,3)]
    
# #     for (label_type, values), pos in zip(label_types.items(), positions):
# #         fig.add_trace(
# #             go.Scatter(
# #                 x=transformed[:, 0],
# #                 y=transformed[:, 1],
# #                 mode='markers',
# #                 marker=dict(
# #                     size=8,  # Reduced size for subplots
# #                     color=values,
# #                     colorscale='Viridis',
# #                     colorbar=dict(
# #                         title=label_type.replace('_', ' ').title(),
# #                         titlefont=dict(size=12),
# #                         tickfont=dict(size=10),
# #                         len=0.5,
# #                         yanchor="middle",
# #                         y=0.5
# #                     ),
# #                     showscale=True,
# #                     opacity=0.6
# #                 ),
# #                 text=full_labels,
# #                 hovertemplate='<b>Value:</b> %{marker.color}<br>' +
# #                              '<b>Label:</b> %{text}<br>' +
# #                              '<b>PC1:</b> %{x:.2f}<br>' +
# #                              '<b>PC2:</b> %{y:.2f}<extra></extra>',
# #                 hoverlabel=dict(font_size=10)
# #             ),
# #             row=pos[0], col=pos[1]
# #         )
        
# #         # Update axes for each subplot
# #         fig.update_xaxes(title_text=f'PC1 ({pca_variance_ratio[0]:.2%})', row=pos[0], col=pos[1])
# #         fig.update_yaxes(title_text=f'PC2 ({pca_variance_ratio[1]:.2%})', row=pos[0], col=pos[1])

# #     # Update overall layout
# #     fig.update_layout(
# #         title=f'{layer_name} - PCA Projections with Different Labels',
# #         width=1800,
# #         height=1000,
# #         template='plotly_white',
# #         showlegend=False,
# #         title_font_size=24
# #     )

# #     # Save the figure
# #     fig.write_image(os.path.join(save_dir, f'{layer_name}_all_labels_pca.png'), scale=2)

# # def analyze_pca_with_multiple_labels(
# #     activations: Dict[str, np.ndarray],
# #     full_labels: List[str],
# #     label_types: Dict[str, np.ndarray],
# #     save_dir: str = 'pca_results_multiple_labels',
# #     target_layer: str = None
# # ):
# #     """Perform PCA analysis with multiple label types"""
# #     os.makedirs(save_dir, exist_ok=True)
    
# #     for layer_name, acts in activations.items():
# #         if target_layer and layer_name != target_layer:
# #             continue
            
# #         print(f"\nAnalyzing {layer_name}...")
        
# #         # Compute PCA once for this layer
# #         pca = PCA(n_components=2)
# #         transformed = pca.fit_transform(acts)
        
# #         # Create all plots using the same transformation
# #         create_multiple_scatter_plots(
# #             transformed, 
# #             label_types, 
# #             full_labels, 
# #             layer_name, 
# #             save_dir,
# #             pca.explained_variance_ratio_
# #         )

# # def main():
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument('--activation_dir', type=str, required=True,
# #                       help='Directory containing activation files')
# #     parser.add_argument('--expressions_file', type=str, required=True,
# #                       help='File containing math expressions with labels')
# #     parser.add_argument('--pred_token_idx', type=int, default=2,
# #                       help='Index of the predicted token')
# #     parser.add_argument('--save_dir', type=str, default='pca_results_multiple_labels',
# #                       help='Directory to save PCA results')
# #     parser.add_argument('--target_layer', type=str, default=None,
# #                       help='Specific layer to analyze (optional)')
    
# #     args = parser.parse_args()
    
# #     activations, full_labels, label_types = load_activations_with_multiple_labels(
# #         args.activation_dir,
# #         args.expressions_file,
# #         args.pred_token_idx
# #     )
    
# #     analyze_pca_with_multiple_labels(
# #         activations,
# #         full_labels,
# #         label_types,
# #         args.save_dir,
# #         args.target_layer
# #     )
    
# #     print(f"\nResults saved to {args.save_dir}")

# # if __name__ == "__main__":
# #     main()

# def extract_labels_from_expressions(expressions_file: str, pred_token_idx: int) -> Tuple[List[str], np.ndarray]:
#     """Extract sequences and convert them to numbers based on prediction index after ####"""
#     full_labels = []
#     numeric_values = []
    
#     with open(expressions_file, 'r') as f:
#         for line in f:
#             if '####' in line:
#                 parts = line.strip().split('####')
#                 if len(parts) == 2:
#                     input_part = parts[0].strip()
#                     prediction_part = parts[1].strip()
#                     # Split prediction part into tokens and get the token at pred_token_idx
#                     pred_tokens = prediction_part.split()
#                     if pred_token_idx < len(pred_tokens):
#                         target_token = pred_tokens[pred_token_idx]
#                         full_labels.append(f"{input_part} #### {prediction_part}")
#                         numeric_values.append(int(target_token))
    
#     return full_labels, np.array(numeric_values)

# # full_labels, numeric_values = extract_labels_from_expressions("data/4_by_4_mult/test_bigbench.txt")
# #%% 
# def load_activations_with_labels(activation_dir: str, expressions_file: str, pred_token_idx: int):
#     """Load activations from final folder and their corresponding labels"""
#     full_labels, numeric_values = extract_labels_from_expressions(expressions_file, pred_token_idx)
#     print(f"Loaded {len(full_labels)} labels")
    
#     final_dir = os.path.join(activation_dir, "final")
#     if not os.path.exists(final_dir):
#         raise ValueError(f"Final directory not found at {final_dir}")
        
#     activations = {}
#     for filename in os.listdir(final_dir):
#         if not filename.endswith(f'_pred_token_{pred_token_idx}.npy'):
#             continue
            
#         layer_name = filename.replace(f'_pred_token_{pred_token_idx}.npy', '')
#         filepath = os.path.join(final_dir, filename)
        
#         print(f"Loading activations from {filepath}")
#         data = np.load(filepath)
#         print(f"Loaded activation shape: {data.shape}")
        
#         if len(data) > len(full_labels):
#             data = data[:len(full_labels)]
#             print(f"Trimmed to {len(data)} samples to match labels")
        
#         activations[layer_name] = data
    
#     return activations, full_labels, numeric_values

# def create_scatter_plot(transformed, numeric_values, full_labels, layer_name, save_dir, title, xaxis_title, yaxis_title):
#     """Create interactive scatter plot with plotly"""
#     fig = go.Figure()
    
#     fig.add_trace(go.Scatter(
#         x=transformed[:, 0],
#         y=transformed[:, 1],
#         mode='markers',
#         marker=dict(
#             size=16,  # Increased marker size
#             color=numeric_values,
#             colorscale='Viridis',
#             colorbar=dict(
#                 title='Numeric Value',
#                 titlefont=dict(size=24),  # Increased font size
#                 tickfont=dict(size=20)  # Increased tick font size
#             ),
#             opacity=0.6
#         ),
#         text=full_labels,  # Add hover text showing full labels
#         hovertemplate='<b style="font-size:20px">Value:</b> %{marker.color}<br>' +
#                      '<b style="font-size:20px">Label:</b> %{text}<br>' +
#                      '<b style="font-size:20px">{xaxis_title}:</b> %{x:.2f}<br>' +
#                      '<b style="font-size:20px">{yaxis_title}:</b> %{y:.2f}<extra></extra>',
#         hoverlabel=dict(font_size=20)  # Increased hover label font size
#     ))
    
#     # Update layout
#     fig.update_layout(
#         title=f'{layer_name} - {title}',
#         xaxis_title=xaxis_title,
#         yaxis_title=yaxis_title,
#         width=1200,  # Increased width
#         height=600,  # Decreased height
#         template='plotly_white',  # Clean white background with grid
#         xaxis=dict(
#             tickfont=dict(size=24),
#             titlefont=dict(size=28)
#         ),
#         yaxis=dict(
#             tickfont=dict(size=24),
#             titlefont=dict(size=28)
#         ),
#         title_font_size=32
#     )
    
#     # Show plot interactively
#     # fig.show()
#     fig.write_image(os.path.join(save_dir, f'{layer_name}_{title.lower()}.png'), scale=2)

# def analyze_pca_with_math_labels(
#     activations: Dict[str, np.ndarray], 
#     full_labels: List[str],
#     numeric_values: np.ndarray,
#     save_dir: str = 'pca_results_labels'
# ):
#     """Perform PCA analysis with scatter plots colored by numeric value"""
#     os.makedirs(save_dir, exist_ok=True)
    
#     for layer_name, acts in activations.items():
#         print(layer_name)
#         print(f"\nAnalyzing {layer_name}...")
        
#         # Perform PCA
#         pca = PCA(n_components=2)
#         transformed = pca.fit_transform(acts)
        
#         create_scatter_plot(transformed, numeric_values, full_labels, layer_name, save_dir, 'PCA Projection', f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')

# def analyze_tsne_with_math_labels(
#     activations: Dict[str, np.ndarray], 
#     full_labels: List[str],
#     numeric_values: np.ndarray,
#     save_dir: str = 'tsne_results_labels'
# ):
#     """Perform t-SNE analysis with scatter plots colored by numeric value"""
#     os.makedirs(save_dir, exist_ok=True)
    
#     for layer_name, acts in activations.items():
#         print(f"\nAnalyzing {layer_name}...")
        
#         # Perform t-SNE
#         tsne = TSNE(n_components=2)
#         transformed = tsne.fit_transform(acts)
        
#         create_scatter_plot(transformed, numeric_values, full_labels, layer_name, save_dir, 't-SNE Projection', 't-SNE Component 1', 't-SNE Component 2')

# def main():
#     # Check if we're in a notebook environment
#     in_notebook = 'ipykernel' in sys.modules
    
#     if not in_notebook:
#         parser = argparse.ArgumentParser()
#         parser.add_argument('--activation_dir', type=str, required=True,
#                           help='Directory containing activation files')
#         parser.add_argument('--expressions_file', default='data/4_by_4_mult/test_bigbench.txt', type=str,
#                           help='File containing math expressions with labels')
#         parser.add_argument('--pred_token_idx', default=2, type=int,
#                           help='Index of the predicted token')
#         parser.add_argument('--save_dir', type=str, default='pca_results_labels_2',
#                           help='Directory to save PCA results')
#         parser.add_argument('--visualization', type=str, default='both',
#                           choices=['scatter', 'density', 'both'],
#                           help='Type of visualization to generate')
#         parser.add_argument('--clustering', type=str, default='pca',
#                           choices=['pca', 'tsne'],
#                           help='Clustering algorithm to use')
#         args = parser.parse_args()
#     else:
#         # Default values for notebook environment
#         class Args:
#             def __init__(self):
#                 self.activation_dir = "cached_activations"
#                 self.expressions_file = "data/4_by_4_mult/test_bigbench.txt"
#                 self.save_dir = "pca_results_labels_2"
#                 self.visualization = "both"
#                 self.clustering = "pca"
#         args = Args()
#     activations, full_labels, numeric_values = load_activations_with_labels(
#         args.activation_dir, 
#         args.expressions_file,
#         args.pred_token_idx
#     )
    
#     if args.clustering == 'pca':
#         analyze_pca_with_math_labels(activations, full_labels, numeric_values, args.save_dir)
#     elif args.clustering == 'tsne':
#         analyze_tsne_with_math_labels(activations, full_labels, numeric_values, args.save_dir)
#     else:
#         raise ValueError(f"Invalid clustering algorithm: {args.clustering}")
    
#     print(f"\nResults saved to {args.save_dir}")

# if __name__ == "__main__":
#     main()
# import numpy as np
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
# import os
# import argparse
# from typing import Dict, List, Tuple
# import plotly.graph_objects as go
# import sys

# def extract_multiple_labels(expressions_file: str, pred_token_idx: int) -> Tuple[List[str], Dict[str, np.ndarray]]:
#     """Extract sequences and multiple types of labels"""
#     full_labels = []
#     label_types = {
#         'first_input1': [],
#         'last_input1': [],
#         'first_input2': [],
#         'last_input2': [],
#         'first_output': [],
#         'last_output': []
#     }
    
#     with open(expressions_file, 'r') as f:
#         for line in f:
#             if '####' in line:
#                 parts = line.strip().split('####')
#                 if len(parts) == 2:
#                     input_part = parts[0].strip()
#                     output_part = parts[1].strip()
                    
#                     # Split the input part by || to get input1 and input2
#                     input_parts = input_part.split('||')
#                     if len(input_parts) == 2:
#                         input1 = input_parts[0].strip()
#                         input2_complex = input_parts[1].strip()
                        
#                         # Get first number from input2 (before any operators)
#                         input2_nums = [x for x in input2_complex.split() if x.isdigit()]
#                         if input2_nums:
#                             input2 = input2_nums[0]
#                         else:
#                             continue
                            
#                         # Extract numbers
#                         input1_nums = [x for x in input1.split() if x.isdigit()]
#                         output_nums = output_part.split()
                        
#                         if input1_nums and output_nums:
#                             full_labels.append(f"{input_part} #### {output_part}")
                            
#                             # Store different label types
#                             label_types['first_input1'].append(int(input1_nums[0]))
#                             label_types['last_input1'].append(int(input1_nums[-1]))
#                             label_types['first_input2'].append(int(input2[0]))
#                             label_types['last_input2'].append(int(input2[-1]))
#                             label_types['first_output'].append(int(output_nums[0]))
#                             label_types['last_output'].append(int(output_nums[-1]))
    
#     return full_labels, {k: np.array(v) for k, v in label_types.items()}

# def load_activations_with_multiple_labels(activation_dir: str, expressions_file: str, pred_token_idx: int):
#     """Load activations and multiple types of labels"""
#     full_labels, label_types = extract_multiple_labels(expressions_file, pred_token_idx)
#     print(f"Loaded {len(full_labels)} examples")
    
#     final_dir = os.path.join(activation_dir, "final")
#     if not os.path.exists(final_dir):
#         raise ValueError(f"Final directory not found at {final_dir}")
        
#     activations = {}
#     for filename in os.listdir(final_dir):
#         if not filename.endswith(f'_pred_token_{pred_token_idx}.npy'):
#             continue
            
#         layer_name = filename.replace(f'_pred_token_{pred_token_idx}.npy', '')
#         filepath = os.path.join(final_dir, filename)
        
#         print(f"Loading activations from {filepath}")
#         data = np.load(filepath)
#         print(f"Loaded activation shape: {data.shape}")
        
#         if len(data) > len(full_labels):
#             data = data[:len(full_labels)]
#             print(f"Trimmed to {len(data)} samples to match labels")
        
#         activations[layer_name] = data
    
#     return activations, full_labels, label_types

# def create_multiple_scatter_plots(transformed, label_types, full_labels, layer_name, save_dir):
#     """Create multiple scatter plots for different label types"""
#     for label_type, values in label_types.items():
#         fig = go.Figure()
        
#         fig.add_trace(go.Scatter(
#             x=transformed[:, 0],
#             y=transformed[:, 1],
#             mode='markers',
#             marker=dict(
#                 size=16,
#                 color=values,
#                 colorscale='Viridis',
#                 colorbar=dict(
#                     title=label_type.replace('_', ' ').title(),
#                     titlefont=dict(size=24),
#                     tickfont=dict(size=20)
#                 ),
#                 opacity=0.6
#             ),
#             text=full_labels,
#             hovertemplate='<b>Value:</b> %{marker.color}<br>' +
#                          '<b>Label:</b> %{text}<br>' +
#                          '<b>PC1:</b> %{x:.2f}<br>' +
#                          '<b>PC2:</b> %{y:.2f}<extra></extra>',
#             hoverlabel=dict(font_size=20)
#         ))
        
#         title = f'PCA - {label_type.replace("_", " ").title()}'
#         fig.update_layout(
#             title=f'{layer_name} - {title}',
#             xaxis_title='PC1',
#             yaxis_title='PC2',
#             width=1200,
#             height=600,
#             template='plotly_white',
#             xaxis=dict(tickfont=dict(size=24), titlefont=dict(size=28)),
#             yaxis=dict(tickfont=dict(size=24), titlefont=dict(size=28)),
#             title_font_size=32
#         )
        
#         fig.write_image(os.path.join(save_dir, f'{layer_name}_{label_type}_pca.png'), scale=2)

# def analyze_pca_with_multiple_labels(
#     activations: Dict[str, np.ndarray],
#     full_labels: List[str],
#     label_types: Dict[str, np.ndarray],
#     save_dir: str = 'pca_results_multiple_labels',
#     target_layer: str = None
# ):
#     """Perform PCA analysis with multiple label types"""
#     os.makedirs(save_dir, exist_ok=True)
    
#     for layer_name, acts in activations.items():
#         if target_layer and layer_name != target_layer:
#             continue
            
#         print(f"\nAnalyzing {layer_name}...")
        
#         pca = PCA(n_components=2)
#         transformed = pca.fit_transform(acts)
        
#         create_multiple_scatter_plots(transformed, label_types, full_labels, layer_name, save_dir)

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--activation_dir', type=str, required=True,
#                       help='Directory containing activation files')
#     parser.add_argument('--expressions_file', type=str, required=True,
#                       help='File containing math expressions with labels')
#     parser.add_argument('--pred_token_idx', type=int, default=2,
#                       help='Index of the predicted token')
#     parser.add_argument('--save_dir', type=str, default='pca_results_multiple_labels',
#                       help='Directory to save PCA results')
#     parser.add_argument('--target_layer', type=str, default=None,
#                       help='Specific layer to analyze (optional)')
    
#     args = parser.parse_args()
    
#     activations, full_labels, label_types = load_activations_with_multiple_labels(
#         args.activation_dir,
#         args.expressions_file,
#         args.pred_token_idx
#     )
    
#     analyze_pca_with_multiple_labels(
#         activations,
#         full_labels,
#         label_types,
#         args.save_dir,
#         args.target_layer
#     )
    
#     print(f"\nResults saved to {args.save_dir}")

# if __name__ == "__main__":
#     main()

# import numpy as np
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
# import os
# from typing import Dict, List, Tuple
# import plotly.graph_objects as go
# from tqdm import tqdm
# import argparse
# import numpy as np
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
# import os
# from typing import Dict, List, Tuple
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# from tqdm import tqdm
# import argparse

# # def extract_multiple_labels(expressions_file: str, pred_token_idx: int) -> Tuple[List[str], Dict[str, np.ndarray]]:
# #     """Extract sequences and multiple types of labels"""
# #     full_labels = []
# #     label_types = {
# #         'first_input1': [],
# #         'last_input1': [],
# #         'first_input2': [],
# #         'last_input2': [],
# #         'first_output': [],
# #         'last_output': []
# #     }
    
# #     print(f"Reading file: {expressions_file}")
# #     with open(expressions_file, 'r') as f:
# #         lines = f.readlines()
    
# #     print(f"Total lines in file: {len(lines)}")
    
# #     for line_idx, line in enumerate(tqdm(lines, desc="Processing labels")):
# #         try:
# #             # Debug: Print first few lines
# #             if line_idx < 5:
# #                 print(f"\nProcessing line {line_idx}: {line.strip()}")
            
# #             if '####' in line:
# #                 parts = line.strip().split('####')
# #                 if len(parts) == 2:
# #                     input_part = parts[0].strip()
# #                     output_part = parts[1].strip()
                    
# #                     # Debug
# #                     if line_idx < 5:
# #                         print(f"Input part: {input_part}")
# #                         print(f"Output part: {output_part}")
                    
# #                     # Split input by spaces first to handle the format in your first file
# #                     input_tokens = input_part.split()
                    
# #                     # Looking for the '*' token to separate inputs
# #                     try:
# #                         star_index = input_tokens.index('*')
# #                         input1_tokens = input_tokens[:star_index]
# #                         input2_tokens = input_tokens[star_index + 1:]
                        
# #                         if line_idx < 5:
# #                             print(f"Input1 tokens: {input1_tokens}")
# #                             print(f"Input2 tokens: {input2_tokens}")
                        
# #                         # Extract numbers
# #                         input1_nums = [x for x in input1_tokens if x.isdigit()]
# #                         input2_nums = [x for x in input2_tokens if x.isdigit()]
# #                         output_nums = [x for x in output_part.split() if x.isdigit()]
                        
# #                         if input1_nums and input2_nums and output_nums:
# #                             full_labels.append(line.strip())
                            
# #                             label_types['first_input1'].append(int(input1_nums[0]))
# #                             label_types['last_input1'].append(int(input1_nums[-1]))
# #                             label_types['first_input2'].append(int(input2_nums[0]))
# #                             label_types['last_input2'].append(int(input2_nums[-1]))
# #                             label_types['first_output'].append(int(output_nums[0]))
# #                             label_types['last_output'].append(int(output_nums[-1]))
                            
# #                             if line_idx < 5:
# #                                 print("Successfully processed example")
                                
# #                     except ValueError as e:
# #                         if line_idx < 5:
# #                             print(f"Error processing line {line_idx}: {str(e)}")
# #                         continue
                        
# #         except Exception as e:
# #             print(f"Error processing line {line_idx}: {str(e)}")
# #             continue
    
# #     print(f"\nTotal examples processed: {len(full_labels)}")
# #     print("Sample of extracted labels:")
# #     for key in label_types:
# #         if label_types[key]:
# #             print(f"{key}: {label_types[key][:5]}")
    
# #     return full_labels, {k: np.array(v) for k, v in label_types.items()}


# # def load_activations_with_labels(activation_dir: str, expressions_file: str, pred_token_idx: int):
# #     """Load activations and their corresponding labels"""
# #     # First get the labels
# #     print("Extracting labels...")
# #     full_labels, label_types = extract_multiple_labels(expressions_file, pred_token_idx)
# #     print(f"Loaded {len(full_labels)} examples")
    
# #     if len(full_labels) == 0:
# #         raise ValueError("No valid examples found in the expressions file")
    
# #     # Then load activations
# #     final_dir = os.path.join(activation_dir, "final")
# #     if not os.path.exists(final_dir):
# #         raise ValueError(f"Final directory not found at {final_dir}")
    
# #     activations = {}
    
# #     # First handle embedding layer if it exists
# #     embedding_file = f'embedding_pred_token_{pred_token_idx}.npy'
# #     embedding_path = os.path.join(final_dir, embedding_file)
# #     if os.path.exists(embedding_path):
# #         print(f"Loading activations from {embedding_path}")
# #         data = np.load(embedding_path)
# #         print(f"Loaded activation shape: {data.shape}")
# #         if len(data) > len(full_labels):
# #             data = data[:len(full_labels)]
# #         activations['embedding'] = data
    
# #     # Then handle transformer layers in order
# #     for layer_idx in range(13):  # 0 through 12
# #         layer_name = f'transformer_layer_{layer_idx}'
# #         filename = f'{layer_name}_pred_token_{pred_token_idx}.npy'
# #         filepath = os.path.join(final_dir, filename)
        
# #         if not os.path.exists(filepath):
# #             print(f"Warning: {filepath} not found, skipping...")
# #             continue
            
# #         print(f"Loading activations from {filepath}")
# #         data = np.load(filepath)
# #         print(f"Loaded activation shape: {data.shape}")
        
# #         if len(data) > len(full_labels):
# #             data = data[:len(full_labels)]
# #             print(f"Trimmed to {len(data)} samples to match labels")
        
# #         activations[layer_name] = data
    
# #     return activations, full_labels, label_types

# # def create_pca_plots(activations: Dict[str, np.ndarray], 
# #                     label_types: Dict[str, np.ndarray], 
# #                     full_labels: List[str],
# #                     save_dir: str = 'pca_results'):
# #     """Create PCA plots for each activation layer and label type"""
# #     os.makedirs(save_dir, exist_ok=True)
    
# #     for layer_name, acts in activations.items():
# #         print(f"\nProcessing layer: {layer_name}")
        
# #         # Perform PCA
# #         pca = PCA(n_components=2)
# #         transformed = pca.fit_transform(acts)
        
# #         # Create plots for each label type
# #         for label_name, labels in label_types.items():
# #             fig = go.Figure()
            
# #             fig.add_trace(go.Scatter(
# #                 x=transformed[:, 0],
# #                 y=transformed[:, 1],
# #                 mode='markers',
# #                 marker=dict(
# #                     size=16,
# #                     color=labels,
# #                     colorscale='Viridis',
# #                     colorbar=dict(
# #                         title=label_name.replace('_', ' ').title(),
# #                         titlefont=dict(size=24),
# #                         tickfont=dict(size=20)
# #                     ),
# #                     opacity=0.6
# #                 ),
# #                 text=full_labels,
# #                 hovertemplate='<b>Value:</b> %{marker.color}<br>' +
# #                              '<b>Example:</b> %{text}<br>' +
# #                              '<b>PC1:</b> %{x:.2f}<br>' +
# #                              '<b>PC2:</b> %{y:.2f}<extra></extra>',
# #                 hoverlabel=dict(font_size=20)
# #             ))
            
# #             fig.update_layout(
# #                 title=f'{layer_name} - {label_name.replace("_", " ").title()}',
# #                 xaxis_title='PC1',
# #                 yaxis_title='PC2',
# #                 width=1200,
# #                 height=800,
# #                 template='plotly_white',
# #                 title_font_size=32,
# #                 xaxis=dict(tickfont=dict(size=24), titlefont=dict(size=28)),
# #                 yaxis=dict(tickfont=dict(size=24), titlefont=dict(size=28))
# #             )
            
# #             output_path = os.path.join(save_dir, f'{layer_name}_{label_name}_pca.png')
# #             fig.write_image(output_path, scale=2)
# #             print(f"Saved plot to {output_path}")

# # def main():
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument('--activation_dir', type=str, required=True,
# #                       help='Directory containing activation files')
# #     parser.add_argument('--expressions_file', type=str, required=True,
# #                       help='File containing math expressions with labels')
# #     parser.add_argument('--pred_token_idx', type=int, default=2,
# #                       help='Index of the predicted token')
# #     parser.add_argument('--save_dir', type=str, default='pca_results_multiple_labels',
# #                       help='Directory to save PCA results')
# #     parser.add_argument('--target_layer', type=str, default=None,
# #                       help='Specific layer to analyze (optional)')
    
# #     args = parser.parse_args()
    
# #     # Load activations and labels
# #     activations, full_labels, label_types = load_activations_with_labels(
# #         args.activation_dir,
# #         args.expressions_file,
# #         args.pred_token_idx,
# #         args.target_layer
# #     )
    
# #     # Create PCA plots
# #     create_pca_plots(activations, label_types, full_labels, args.save_dir)
    
# #     print(f"\nAnalysis complete. Results saved to {args.save_dir}")

# # if __name__ == "__main__":
# #     main()

# def extract_labels(expressions_file):
#     """Extract labels from expressions file"""
#     full_labels = []
#     label_types = {
#         'first_input1': [],
#         'last_input1': [],
#         'first_input2': [],
#         'last_input2': [],
#         'first_output': [],
#         'last_output': []
#     }
    
#     with open(expressions_file, 'r') as f:
#         for line in f:
#             if '####' in line:
#                 parts = line.strip().split('####')
#                 if len(parts) == 2:
#                     input_part = parts[0].strip()
#                     output_part = parts[1].strip()
                    
#                     input_tokens = input_part.split()
#                     try:
#                         star_index = input_tokens.index('*')
#                         input1_tokens = input_tokens[:star_index]
#                         input2_tokens = input_tokens[star_index + 1:]
                        
#                         input1_nums = [x for x in input1_tokens if x.isdigit()]
#                         input2_nums = [x for x in input2_tokens if x.isdigit()]
#                         output_nums = [x for x in output_part.split() if x.isdigit()]
                        
#                         if input1_nums and input2_nums and output_nums:
#                             full_labels.append(line.strip())
#                             label_types['first_input1'].append(int(input1_nums[0]))
#                             label_types['last_input1'].append(int(input1_nums[-1]))
#                             label_types['first_input2'].append(int(input2_nums[0]))
#                             label_types['last_input2'].append(int(input2_nums[-1]))
#                             label_types['first_output'].append(int(output_nums[0]))
#                             label_types['last_output'].append(int(output_nums[-1]))
#                     except:
#                         continue
    
#     return {k: np.array(v) for k, v in label_types.items()}

# def create_pca_plot(activation_path, labels, output_path):
#     """Create PCA plot with 6 subplots for different label types"""
#     # Load activation data
#     activations = np.load(activation_path)
    
#     # Perform PCA
#     pca = PCA(n_components=2)
#     transformed = pca.fit_transform(activations)
    
#     # Calculate explained variance
#     explained_var = pca.explained_variance_ratio_ * 100
#     total_var = sum(explained_var)
    
#     print(f"Explained variance:")
#     print(f"PC1: {explained_var[0]:.2f}%")
#     print(f"PC2: {explained_var[1]:.2f}%")
#     print(f"Total: {total_var:.2f}%")
    
#     # Create subplot figure
#     fig = make_subplots(
#         rows=2, cols=3,
#         subplot_titles=list(labels.keys()),
#         horizontal_spacing=0.15,
#         vertical_spacing=0.15
#     )
    
#     # Add traces for each label type
#     for idx, (label_name, label_values) in enumerate(labels.items()):
#         row = idx // 3 + 1
#         col = idx % 3 + 1
        
#         fig.add_trace(
#             go.Scatter(
#                 x=transformed[:, 0],
#                 y=transformed[:, 1],
#                 mode='markers',
#                 marker=dict(
#                     size=2,
#                     color=label_values,
#                     colorscale='Viridis',
#                     showscale=True,
#                     opacity=0.6
#                 ),
#                 showlegend=False
#             ),
#             row=row,
#             col=col
#         )
        
#         # Update axes
#         fig.update_xaxes(title_text=f"PC1 ({explained_var[0]:.0f}%)", row=row, col=col)
#         fig.update_yaxes(title_text=f"PC2 ({explained_var[1]:.0f}%)", row=row, col=col)
    
#     # Update layout
#     fig.update_layout(
#         title=f'PCA Analysis (Total variance: {total_var:.0f}%)',
#         width=800,
#         height=800,
#         showlegend=False,
#         margin=dict(t=60, b=40, l=40, r=40)
#     )
    
#     # Save plot
#     fig.write_image(output_path)
#     print(f"Saved plot to {output_path}")

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--activation_path', type=str, required=True,
#                       help='Path to the activation .npy file')
#     parser.add_argument('--expressions_file', type=str, required=True,
#                       help='Path to expressions file')
#     parser.add_argument('--output_path', type=str, required=True,
#                       help='Path to save the output plot')
    
#     args = parser.parse_args()
    
#     # Extract labels
#     print("Extracting labels...")
#     labels = extract_labels(args.expressions_file)
    
#     # Create and save plot
#     print("Creating PCA plot...")
#     create_pca_plot(args.activation_path, labels, args.output_path)

# if __name__ == "__main__":
#     main()

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple
import plotly.graph_objects as go
from tqdm import tqdm
import argparse
from plotly.subplots import make_subplots

def extract_multiple_labels(expressions_file: str, pred_token_idx: int) -> Tuple[List[str], Dict[str, np.ndarray]]:
    """Extract sequences and multiple types of labels"""
    full_labels = []
    label_types = {
        'first_input1': [],
        'last_input1': [],
        'first_input2': [],
        'last_input2': [],
        'first_output': [],
        'last_output': []
    }
    
    with open(expressions_file, 'r') as f:
        for line in tqdm(f, desc="Processing labels"):
            if '####' in line:
                parts = line.strip().split('####')
                if len(parts) == 2:
                    input_part = parts[0].strip()
                    output_part = parts[1].strip()
                    
                    input_parts = input_part.split('||')
                    if len(input_parts) == 2:
                        input1 = input_parts[0].strip()
                        input2_complex = input_parts[1].strip()
                        
                        input2_nums = [x for x in input2_complex.split() if x.isdigit()]
                        if input2_nums:
                            input2 = input2_nums[0]
                        else:
                            continue
                            
                        input1_nums = [x for x in input1.split() if x.isdigit()]
                        output_nums = output_part.split()
                        
                        if input1_nums and output_nums:
                            full_labels.append(f"{input_part} #### {output_part}")
                            
                            label_types['first_input1'].append(int(input1_nums[0]))
                            label_types['last_input1'].append(int(input1_nums[-1]))
                            label_types['first_input2'].append(int(input2[0]))
                            label_types['last_input2'].append(int(input2[-1]))
                            label_types['first_output'].append(int(output_nums[0]))
                            label_types['last_output'].append(int(output_nums[-1]))
    
    return full_labels, {k: np.array(v) for k, v in label_types.items()}

def load_activations_with_labels(activation_dir: str, expressions_file: str, pred_token_idx: int, target_layer: str = None):
    """Load activations and their corresponding labels"""
    # First get the labels
    print("Extracting labels...")
    full_labels, label_types = extract_multiple_labels(expressions_file, pred_token_idx)
    print(f"Loaded {len(full_labels)} examples")
    
    # Then load activations
    final_dir = os.path.join(activation_dir, "final")
    if not os.path.exists(final_dir):
        raise ValueError(f"Final directory not found at {final_dir}")
    
    activations = {}
    for filename in os.listdir(final_dir):
        if not filename.endswith(f'_pred_token_{pred_token_idx}.npy'):
            continue
            
        layer_name = filename.replace(f'_pred_token_{pred_token_idx}.npy', '')
        
        # Skip if not the target layer (when specified)
        if target_layer and layer_name != target_layer:
            continue
            
        filepath = os.path.join(final_dir, filename)
        print(f"Loading activations from {filepath}")
        
        data = np.load(filepath)
        print(f"Loaded activation shape: {data.shape}")
        
        if len(data) > len(full_labels):
            data = data[:len(full_labels)]
            print(f"Trimmed to {len(data)} samples to match labels")
        
        activations[layer_name] = data
    
    return activations, full_labels, label_types

# def create_pca_plots(activations: Dict[str, np.ndarray], 
#                     label_types: Dict[str, np.ndarray], 
#                     full_labels: List[str],
#                     save_dir: str = 'pca_results'):
#     """Create PCA plots for each activation layer and label type"""
#     os.makedirs(save_dir, exist_ok=True)
    
#     for layer_name, acts in activations.items():
#         print(f"\nProcessing layer: {layer_name}")
        
#         # Perform PCA
#         pca = PCA(n_components=2)
#         transformed = pca.fit_transform(acts)
        
#         # Create plots for each label type
#         for label_name, labels in label_types.items():
#             fig = go.Figure()
            
#             fig.add_trace(go.Scatter(
#                 x=transformed[:, 0],
#                 y=transformed[:, 1],
#                 mode='markers',
#                 marker=dict(
#                     size=16,
#                     color=labels,
#                     colorscale='Viridis',
#                     colorbar=dict(
#                         title=label_name.replace('_', ' ').title(),
#                         titlefont=dict(size=24),
#                         tickfont=dict(size=20)
#                     ),
#                     opacity=0.6
#                 ),
#                 text=full_labels,
#                 hovertemplate='<b>Value:</b> %{marker.color}<br>' +
#                              '<b>Example:</b> %{text}<br>' +
#                              '<b>PC1:</b> %{x:.2f}<br>' +
#                              '<b>PC2:</b> %{y:.2f}<extra></extra>',
#                 hoverlabel=dict(font_size=20)
#             ))
            
#             fig.update_layout(
#                 title=f'{layer_name} - {label_name.replace("_", " ").title()}',
#                 xaxis_title='PC1',
#                 yaxis_title='PC2',
#                 width=1200,
#                 height=800,
#                 template='plotly_white',
#                 title_font_size=32,
#                 xaxis=dict(tickfont=dict(size=24), titlefont=dict(size=28)),
#                 yaxis=dict(tickfont=dict(size=24), titlefont=dict(size=28))
#             )
            
#             output_path = os.path.join(save_dir, f'{layer_name}_{label_name}_pca.png')
#             fig.write_image(output_path, scale=2)
#             print(f"Saved plot to {output_path}")

def create_pca_plots(activations: Dict[str, np.ndarray], 
                    label_types: Dict[str, np.ndarray], 
                    full_labels: List[str],
                    save_dir: str = 'pca_results'):
    """Create PCA plots for each activation layer and label type"""
    os.makedirs(save_dir, exist_ok=True)
    
    for layer_name, acts in activations.items():
        print(f"\nProcessing layer: {layer_name}")
        
        # Perform PCA
        pca = PCA(n_components=2)
        transformed = pca.fit_transform(acts)
        
        # Calculate explained variance
        explained_var = pca.explained_variance_ratio_ * 100
        total_var = sum(explained_var)
        
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=['Last digit of input 1', 'First digit of input 1', 'Last digit of input 2', 'First digit of input 2', 'Last digit of output', 'First digit of output'],
            horizontal_spacing=0.2,  # Increased spacing
            vertical_spacing=0.2     # Increased spacing
        )
        
        # Add each label type as a subplot
        for idx, (label_name, labels) in enumerate(label_types.items()):
            row = idx // 3 + 1
            col = idx % 3 + 1
            
            fig.add_trace(
                go.Scatter(
                    x=transformed[:, 0],
                    y=transformed[:, 1],
                    mode='markers',
                    marker=dict(
                        size=3,  # Slightly larger markers
                        color=labels,
                        colorscale='Viridis',
                        colorbar=dict(
                            title=None,
                            thickness=10,  # Thicker colorbar
                            len=0.3,
                            x=1.02 if col == 3 else 0.985  # Adjust position based on column
                        ),
                        showscale=True,
                        opacity=0.7  # Slightly more opaque
                    ),
                    showlegend=False
                ),
                row=row,
                col=col
            )
            
            # Update axes labels with variance information and adjusted position
            fig.update_xaxes(
                title=dict(
                    text=f"PC1 (var: {explained_var[0]:.1f}%)",
                    font=dict(size=10),
                    standoff=25  # More space between axis and title
                ),
                tickfont=dict(size=9),
                row=row,
                col=col
            )
            fig.update_yaxes(
                title=dict(
                    text=f"PC2 (var: {explained_var[1]:.1f}%)",
                    font=dict(size=10),
                    standoff=25  # More space between axis and title
                ),
                tickfont=dict(size=9),
                row=row,
                col=col
            )
        
        # Update layout with better margins
        fig.update_layout(
            title=dict(
                text=f'',
                font=dict(size=10),
                y=0.95
            ),
            width=900,
            height=900,
            showlegend=False,
            margin=dict(t=80, b=50, l=80, r=80),  # Increased margins
            template='plotly_white',
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Adjust subplot titles position
        for annotation in fig['layout']['annotations']:
            annotation['y'] = annotation['y'] + 0.03  # Move subplot titles up slightly
        
        # Save the combined plot with higher quality
        output_path = os.path.join(save_dir, f'{layer_name}_combined_pca.png')
        try:
            fig.write_image(output_path, scale=1.5)  # Increased scale for better quality
            print(f"Saved combined plot to {output_path}")
            
            # Save variance data
            variance_data = {
                'PC1_variance': explained_var[0],
                'PC2_variance': explained_var[1],
                'total_variance': total_var
            }
            variance_path = os.path.join(save_dir, f'{layer_name}_variance.npy')
            np.save(variance_path, variance_data)
            
        except Exception as e:
            print(f"Error saving plot: {str(e)}")
        
        # Clear memory
        plt.close('all')
        del fig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--activation_dir', type=str, required=True,
                      help='Directory containing activation files')
    parser.add_argument('--expressions_file', type=str, required=True,
                      help='File containing math expressions with labels')
    parser.add_argument('--pred_token_idx', type=int, default=2,
                      help='Index of the predicted token')
    parser.add_argument('--save_dir', type=str, default='pca_results_multiple_labels',
                      help='Directory to save PCA results')
    parser.add_argument('--target_layer', type=str, default=None,
                      help='Specific layer to analyze (optional)')
    
    args = parser.parse_args()
    
    # Load activations and labels
    activations, full_labels, label_types = load_activations_with_labels(
        args.activation_dir,
        args.expressions_file,
        args.pred_token_idx,
        args.target_layer
    )
    
    # Create PCA plots
    create_pca_plots(activations, label_types, full_labels, args.save_dir)
    
    print(f"\nAnalysis complete. Results saved to {args.save_dir}")

if __name__ == "__main__":
    main()

# %%
