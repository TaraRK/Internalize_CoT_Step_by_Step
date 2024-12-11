#%%
import numpy as np
from transformers import AutoModelForCausalLM, GPT2LMHeadModel
from nnsight import LanguageModel
from model import ImplicitModel
from transformer_lens import HookedTransformer
import torch
from tqdm import tqdm
#%%

# load activatiosn of all layers 
tok_position = 0
activations = [
    np.load(f"../cached_activations/final/embedding_pred_token_{tok_position}.npy")
] + [
    np.load(f"../cached_activations/final/transformer_layer_{layer}_pred_token_{tok_position}.npy")
    for layer in range(12)
] 

activations = np.array(activations)
print(activations.shape)

# %%
device = "cuda:2"
checkpoint_dir = "../trained_models/final_checkpoint"
impl_model = ImplicitModel.from_pretrained(checkpoint_dir)
model = HookedTransformer.from_pretrained('gpt2', hf_model=impl_model.base_model)
tokenizer = impl_model.tokenizer
W_U = model.W_U.detach().cpu().numpy()
print(W_U.shape)
# %%
activations_logits = []
top1_counts = [] 
for layer in tqdm(range(13)):
    logits = (activations[layer] @ W_U) 
    activations_logits.append(logits.mean(axis=0))
    probs = torch.from_numpy(logits).softmax(dim=-1)
    top1 = probs.argmax(dim=-1)
    unique, counts = np.unique(top1, return_counts=True)
    top1_counts.append(dict(zip(unique, counts)))

activations_logits = np.array(activations_logits)
print(activations_logits.shape)
#%% 
for i, top1_count in enumerate(top1_counts):
    print(f"Layer {i}:")
    for token, count in top1_count.items():
        print(f"Token: {tokenizer.decode(token)}, Count: {count}")
    print("\n")
#%% 
for layer in tqdm(range(13)):
    probs = torch.from_numpy(activations_logits[layer]).softmax(dim=-1) 
    top1 = probs.argmax(dim=-1)
    top5 = probs.topk(5, dim=-1).indices
    print(f"Layer {layer}:")
    print(f"Top 1: {tokenizer.decode(top1)}")
    print(f"Top 5: {tokenizer.decode(top5)}")
    print("\n")

# %%
