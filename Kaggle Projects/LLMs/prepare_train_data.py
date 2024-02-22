import numpy as np
import pandas as pd
import os
import glob
from tqdm import tqdm

the_prompts = [
    "Exploring Venus",
    "The Face on Mars",
    "Facial action coding system",
    "A Cowboy Who Rode the Waves",
    '"A Cowboy Who Rode the Waves"',
    "Driverless cars",
]

df = pd.read_csv('../../external/persuade_corpus_2.0/persuade_2.0_human_scores_demo_id_github.csv')
text_list = df['full_text'].values
prompt_name_list = df['prompt_name'].values

text_list1 = []
for i in range(len(text_list)):
    if prompt_name_list[i] in the_prompts:
        text_list1.append(text_list[i])

print(len(text_list1))

text_list = []
file_list = sorted(glob.glob('generated/*.csv'))
for file_name in tqdm(file_list, total=len(file_list)):
    text_list += list(pd.read_csv('generated/'+file_name.split('/')[-1])['generated'].values)

print(len(text_list))

label_list = [1]*len(text_list) + [0]*len(text_list1)
text_list += text_list1

print(len(text_list), len(label_list))

train_df = pd.DataFrame(data={'text': text_list, 'label': label_list})

out_dir = 'split2/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

train_df.to_csv(out_dir+'train_df.csv', index=False)
