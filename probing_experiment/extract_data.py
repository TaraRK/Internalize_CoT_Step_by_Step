#%% 

import numpy as np
import os
from tqdm import tqdm

# Initialize lists to store expected outputs for each probe
probe_labels_logistical = { 
    'First Num of Output': [],
    'Second Num of Output': [],
    'Actual First Num of Output': [],
}
probe_labels_linear = {
    'First Input': [],
    'Second Input': [],
    'Output': [],
    '1st Input x Last Digit': [],
    '1st Input x 10 x 2nd Digit': [],
    '1st Input x 100 x 3rd Digit': [],
    '1st Input x 1000 x 4th Digit': [], 
}
probe_labels_log = {
    'Log Output': [],
    'Log Output - 1st Input x Last Digit': [],
    'Log Output - 1st Input x 10 x 2nd Digit': [],
    'Log Output - 1st Input x 100 x 3rd Digit': [],
    'Log Output - 1st Input x 1000 x 4th Digit': [],
    'Log First Input': [],
    'Log Second Input': [],
}
probe_labels_is_digit_x = [
    [ [] for _ in range(10)] for _ in range(8)
]


# Assuming your dataset is a list of strings (one per example)
# Load dataset from file
num_examples = 10000
with open('data/4_by_4_mult/train.txt', 'r') as f:
    dataset = [line.strip() for line in f.readlines()][:num_examples]

#%%
dataset[:10]
#%% 
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
    probe_labels_linear['Output'].append(int("".join(output[::-1])))
    probe_labels_logistical['First Num of Output'].append(int(output[0]))
    probe_labels_logistical['Second Num of Output'].append(int(output[1]))
    probe_labels_logistical['Actual First Num of Output'].append(int(output[-1]))
    probe_labels_linear['First Input'].append(dcba)

    hgfe = h * 1000 + g * 100 + f * 10 + e
    probe_labels_linear['Second Input'].append(hgfe)

    mid1 = dcba * e
    probe_labels_linear['1st Input x Last Digit'].append(mid1)
    
    mid2 = dcba * (f * 10)
    probe_labels_linear['1st Input x 10 x 2nd Digit'].append(mid2)
    
    mid3 = dcba * (g * 100)
    probe_labels_linear['1st Input x 100 x 3rd Digit'].append(mid3)
    
    mid4 = dcba * (h * 1000)
    probe_labels_linear['1st Input x 1000 x 4th Digit'].append(mid4)
    
    probe_labels_log['Log Output'].append(np.log10(float("".join(output[::-1]))+1e-10))
    probe_labels_log['Log Output - 1st Input x Last Digit'].append(np.log10(float(mid1) + 1e-10))
    probe_labels_log['Log Output - 1st Input x 10 x 2nd Digit'].append(np.log10(float(mid2) + 1e-10))
    probe_labels_log['Log Output - 1st Input x 100 x 3rd Digit'].append(np.log10(float(mid3) + 1e-10))
    probe_labels_log['Log Output - 1st Input x 1000 x 4th Digit'].append(np.log10(float(mid4) + 1e-10))
    probe_labels_log['Log First Input'].append(np.log10(float(dcba) + 1e-10))
    probe_labels_log['Log Second Input'].append(np.log10(float(hgfe) + 1e-10))
    for i in range(len(output)): 
        output_digit = int(output[i])
        for digit in range(10): 
            if output_digit == digit: 
                probe_labels_is_digit_x[i][digit].append(1)
            else: 
                probe_labels_is_digit_x[i][digit].append(0)
    


    if (idx + 1) % 100 == 0:
        print(f"Processed {idx + 1}/{num_examples} data points.")
    # print(data_point) 
    # print(probe_labels)

for key, value in probe_labels_logistical.items():
    probe_labels_logistical[key] = np.array(value)

for key, value in probe_labels_linear.items():
    # if key != 'Output':     
    #     probe_labels_linear[key] = np.array(value)
    # else: 
    min_val = min(value)
    max_val = max(value)
    if max_val != min_val:  # Avoid division by zero
        value = [((x - min_val) / (max_val - min_val)) * 100 for x in value]
    probe_labels_linear[key] = np.array(value)

for key, value in probe_labels_log.items():
    probe_labels_log[key] = np.array(value)

probe_labels_is_digit_x = np.array(probe_labels_is_digit_x)

np.save('probe_labels_logistical.npy', probe_labels_logistical)
np.save('probe_labels_linear.npy', probe_labels_linear)
np.save('probe_labels_log.npy', probe_labels_log)
np.save('probe_labels_is_digit_x.npy', probe_labels_is_digit_x)

# %%
for data in dataset[:5]:
    print(data)

for key, value in probe_labels_linear.items():
    print(key)
    print(value[:5])
# %%
