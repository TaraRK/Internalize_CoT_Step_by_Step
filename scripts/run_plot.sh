
python plotting/analyze_activations.py --activation_dir cached_activations --clustering pca --save_dir pca_results --pred_token_idx 0
python plotting/analyze_activations.py --activation_dir cached_activations --clustering tsne --save_dir tsne_results --pred_token_idx 0
# python plotting/utils.py --folder_path pca_results
# python plotting/utils.py --folder_path tsne_results