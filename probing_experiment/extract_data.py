#%% 

import numpy as np
import os
from tqdm import tqdm

# Initialize lists to store expected outputs for each probe
probe_labels = { 
    'First Num of Output': [], 
    'Second Num of Output': [],
    'First Num of Output (Reversed)': [],
    'First Input': [],
    'Second Input': [],
    'Output': [],
    'Mid 1': [],
    'Mid 2': [],
    'Mid 3': [],
    'Mid 4': [],
}

# Assuming your dataset is a list of strings (one per example)
# Load dataset from file
with open('data/4_by_4_mult/test_bigbench.txt', 'r') as f:
    dataset = [line.strip() for line in f.readlines()]
    num_examples = len(dataset)

for idx, data_point in tqdm(enumerate(dataset)):
    tokens = data_point.split(" ")
    
    if len(tokens) < 9:
        print(f"Data point {idx} does not have enough tokens.")
        continue
    
    first_four_digits = tokens[0:4]  # ['a', 'b', 'c', 'd']
    # Reverse them to get ['d', 'c', 'b', 'a']
    reversed_digits = first_four_digits[::-1]
    dcba_str = ''.join(reversed_digits)
    try:
        dcba = int(dcba_str)
        # print(dcba)
    except ValueError:
        print(f"Data point {idx}: Unable to convert dcba '{dcba_str}' to integer.")
        continue
    
    try:
        star_index = tokens.index('*')
    except ValueError:
        print(f"Data point {idx} does not contain '*' token.")
        continue

    if len(tokens) <= star_index + 4:
        print(f"Data point {idx} does not have four digits after '*'.")
        continue
    
    next_four_digits = tokens[star_index + 1 : star_index + 5]
    try:
        e = int(next_four_digits[0])
        f = int(next_four_digits[1])
        g = int(next_four_digits[2])
        h = int (next_four_digits[3].split('||')[0])
    except ValueError:
        print(f"Data point {idx}: Non-integer value found in digits after '*'.")
        continue
    
    output = data_point.split('####')[1].strip().split(' ')
    probe_labels['Output'].append(int("".join(output[::-1])))
    probe_labels['First Num of Output'].append(int(output[0]))
    probe_labels['Second Num of Output'].append(int(output[1]))
    probe_labels['First Num of Output (Reversed)'].append(int(output[-1]))
    probe_labels['First Input'].append(dcba)

    hgfe = h * 1000 + g * 100 + f * 10 + e
    probe_labels['Second Input'].append(hgfe)

    mid1 = dcba * e
    probe_labels['Mid 1'].append(mid1)
    
    mid2 = dcba * (f * 10)
    probe_labels['Mid 2'].append(mid2)
    
    mid3 = dcba * (g * 100)
    probe_labels['Mid 3'].append(mid3)
    
    mid4 = dcba * (h * 1000)
    probe_labels['Mid 4'].append(mid4)
    

    if (idx + 1) % 100 == 0:
        print(f"Processed {idx + 1}/{num_examples} data points.")
    # print(data_point) 
    # print(probe_labels)

for key, value in probe_labels.items():
    probe_labels[key] = np.array(value)

np.save('probe_labels.npy', probe_labels)

# %%
probe_labels = np.load('probe_labels.npy', allow_pickle=True).item()
# %%
