import json
import numpy as np
import random
import os
import glob
import pandas as pd
from tqdm import tqdm
import pickle

out_dir = 'split1/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

prompt_list = []
neg_list = []
pos_list = []

file_list = sorted(glob.glob('generated/*.parquet'))

for file_name in tqdm(file_list, total=len(file_list)):
    prompt_list += list(pd.read_parquet('promt/'+file_name.split('/')[-1])['promt'].values)
    neg_list += list(pd.read_parquet('text/'+file_name.split('/')[-1])['text'].values)
    pos_list += list(pd.read_parquet('generated/'+file_name.split('/')[-1])['generated'].values)

print(len(prompt_list), len(neg_list), len(pos_list))

rand_idx = np.random.permutation(len(neg_list))
train_idx = rand_idx[5000:]
valid_idx = rand_idx[:5000]

train_prompt_list = []
train_neg_list = []
train_pos_list = []
for idx in tqdm(train_idx, total=len(train_idx)):
    train_prompt_list.append(prompt_list[idx])
    train_neg_list.append(neg_list[idx])
    train_pos_list.append(pos_list[idx])

valid_prompt_list = []
valid_neg_list = []
valid_pos_list = []
for idx in tqdm(valid_idx, total=len(valid_idx)):
    valid_prompt_list.append(prompt_list[idx])
    valid_neg_list.append(neg_list[idx])
    valid_pos_list.append(pos_list[idx])

print(len(train_prompt_list), len(train_neg_list), len(train_pos_list))
print(len(valid_prompt_list), len(valid_neg_list), len(valid_pos_list))

with open(out_dir+'train_prompt_list.pickle', 'wb') as f:
    pickle.dump(train_prompt_list, f, protocol=pickle.HIGHEST_PROTOCOL)
with open(out_dir+'train_neg_list.pickle', 'wb') as f:
    pickle.dump(train_neg_list, f, protocol=pickle.HIGHEST_PROTOCOL)
with open(out_dir+'train_pos_list.pickle', 'wb') as f:
    pickle.dump(train_pos_list, f, protocol=pickle.HIGHEST_PROTOCOL)

with open(out_dir+'valid_prompt_list.pickle', 'wb') as f:
    pickle.dump(valid_prompt_list, f, protocol=pickle.HIGHEST_PROTOCOL)
with open(out_dir+'valid_neg_list.pickle', 'wb') as f:
    pickle.dump(valid_neg_list, f, protocol=pickle.HIGHEST_PROTOCOL)
with open(out_dir+'valid_pos_list.pickle', 'wb') as f:
    pickle.dump(valid_pos_list, f, protocol=pickle.HIGHEST_PROTOCOL)


