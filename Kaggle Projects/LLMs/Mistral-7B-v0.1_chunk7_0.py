import numpy as np
import random
import os
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import time

file_name = 'chunk7_0'
print("Now processing: ", file_name)

out_dir_generated = '../generated/'
if not os.path.exists(out_dir_generated):
    os.makedirs(out_dir_generated)

start_time = time.time()

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", device_map="auto", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", padding_side="left", truncation_side="left")

suppress_id_list = []
suppress_id_list.append(tokenizer.convert_tokens_to_ids('#'))
suppress_id_list.append(tokenizer.convert_tokens_to_ids('##'))
suppress_id_list.append(tokenizer.convert_tokens_to_ids('###'))
suppress_id_list.append(tokenizer.convert_tokens_to_ids('####'))

prompt_list = list(pd.read_parquet('../promt/'+file_name+'.parquet')['promt'].values)
print(len(prompt_list))

text_generated_list = []
batch_size = 12
for i in range(0, len(prompt_list), batch_size):
    print(i, i+batch_size)
    model_inputs = tokenizer(prompt_list[i:i+batch_size], return_tensors="pt", truncation=True, max_length=1024).to("cuda")
    generated_ids = model.generate(**model_inputs, min_new_tokens=1024, max_new_tokens=1024, do_sample=True, suppress_tokens=suppress_id_list)
    text_generated = tokenizer.batch_decode(generated_ids[:, 1024:], skip_special_tokens=True)
    text_generated_list += text_generated

end_time = time.time()
print(end_time-start_time)

df = pd.DataFrame(data={'generated': text_generated_list})
df.to_parquet(out_dir_generated+file_name+'.parquet', index=False)
