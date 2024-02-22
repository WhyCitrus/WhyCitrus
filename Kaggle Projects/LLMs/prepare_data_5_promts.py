import numpy as np
import pandas as pd
import os

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
assignment_list = df['assignment'].values
prompt_name_list = df['prompt_name'].values

text_list1 = []
assignment_list1 = []
for i in range(len(text_list)):
    if prompt_name_list[i] in the_prompts:
        text_list1.append(text_list[i])
        assignment_list1.append(assignment_list[i])

train_df = pd.DataFrame(data={'instruction': assignment_list1, 'output': text_list1})

out_dir = 'data/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

train_df.to_csv(out_dir+'persuade_5_prompts.csv', index=False)
