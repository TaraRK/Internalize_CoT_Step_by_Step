# Internalize CoT Step by Step

This is the fork of the original repository. The code structure is as follows: 

- probing experiment: our probing exeriments
- plotting: any code used for visualizations and plots produces including the pca visualizations and the t-SNE visualizations
- src: source code for generation, we attached hooks to cache activations in generate.py and utils.py, we also use train.py to train the model.
- scripts: examples of scripts we use to run the code
- data and models are directories from the original repository we use to maintain the datasets and model checkpoints.
