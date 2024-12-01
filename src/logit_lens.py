#%%
import numpy as np
from transformers import AutoModelForCausalLM, GPT2LMHeadModel
from nnsight import LanguageModel
from model import ImplicitModel
from transformer_lens import HookedTransformer
import torch
activations = np.load("../cached_activations/final/transformer_layer_11_pred_token_2.npy")
print(activations.shape)

# %%
device = "cuda:5"
checkpoint_dir = "../trained_models/final_checkpoint"
impl_model = ImplicitModel.from_pretrained(checkpoint_dir)
model = HookedTransformer.from_pretrained('gpt2', hf_model=impl_model.base_model)
tokenizer = impl_model.tokenizer
# %%
W_U = model.W_U.detach().cpu().numpy()
print(W_U.shape)
# %%
activations_logits = activations @ W_U
print(activations_logits.shape)
# %%
probs = torch.from_numpy(activations_logits).softmax(dim=-1).argmax(dim=-1)
print(probs.shape)
print(probs)
print(tokenizer.decode(probs))
# %%
probs[1]

# %%
