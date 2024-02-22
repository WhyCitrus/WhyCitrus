import numpy as np
import random
import os
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import time

out_dir_generated = 'generated/'
if not os.path.exists(out_dir_generated):
    os.makedirs(out_dir_generated)

model_name = 'h2ogpt-4096-llama2-7b_1'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)
model.eval()

num_generated_per_prompt = 25
run = 100

prompt = '<|prompt|>In "The Challenge of Exploring Venus," the author suggests studying Venus is a worthy pursuit despite the dangers it presents. Using details from the article, write an essay evaluating how well the author supports this idea. Be sure to include: a claim that evaluates how well the author supports the idea that studying Venus is a worthy pursuit despite the dangers; an explanation of the evidence from the article that supports your claim; an introduction, a body, and a conclusion to your essay. <|answer|>'

prompt_list = [prompt]*num_generated_per_prompt
text_generated_list = []
batch_size = 25
for i in range(0, len(prompt_list), batch_size):
    print(i, i+batch_size)
    inputs = tokenizer(prompt_list[i:i+batch_size], return_tensors="pt", add_special_tokens=False).to("cuda")
    tokens = model.generate(**inputs, max_new_tokens=1024, do_sample=True, temperature=1.0)
    tokens = tokens[:, inputs["input_ids"].shape[1]:]
    text_generated = tokenizer.batch_decode(tokens, skip_special_tokens=True)
    text_generated_list += text_generated

df = pd.DataFrame(data={'generated': text_generated_list})
df.to_csv(out_dir_generated+model_name+'_prompt1_run{}.csv'.format(run), index=False)


prompt = "<|prompt|>You have read the article 'Unmasking the Face on Mars.' Imagine you are a scientist at NASA discussing the Face with someone who thinks it was created by aliens. Using information in the article, write an argumentative essay to convince someone that the Face is just a natural landform.Be sure to include: claims to support your argument that the Face is a natural landform; evidence from the article to support your claims; an introduction, a body, and a conclusion to your argumentative essay. <|answer|>"

prompt_list = [prompt]*num_generated_per_prompt
text_generated_list = []
batch_size = 25
for i in range(0, len(prompt_list), batch_size):
    print(i, i+batch_size)
    inputs = tokenizer(prompt_list[i:i+batch_size], return_tensors="pt", add_special_tokens=False).to("cuda")
    tokens = model.generate(**inputs, max_new_tokens=1024, do_sample=True, temperature=1.0)
    tokens = tokens[:, inputs["input_ids"].shape[1]:]
    text_generated = tokenizer.batch_decode(tokens, skip_special_tokens=True)
    text_generated_list += text_generated

df = pd.DataFrame(data={'generated': text_generated_list})
df.to_csv(out_dir_generated+model_name+'_prompt2_run{}.csv'.format(run), index=False)


prompt = '<|prompt|>In the article "Making Mona Lisa Smile," the author describes how a new technology called the Facial Action Coding System enables computers to identify human emotions. Using details from the article, write an essay arguing whether the use of this technology to read the emotional expressions of students in a classroom is valuable. <|answer|>'

prompt_list = [prompt]*num_generated_per_prompt
text_generated_list = []
batch_size = 25
for i in range(0, len(prompt_list), batch_size):
    print(i, i+batch_size)
    inputs = tokenizer(prompt_list[i:i+batch_size], return_tensors="pt", add_special_tokens=False).to("cuda")
    tokens = model.generate(**inputs, max_new_tokens=1024, do_sample=True, temperature=1.0)
    tokens = tokens[:, inputs["input_ids"].shape[1]:]
    text_generated = tokenizer.batch_decode(tokens, skip_special_tokens=True)
    text_generated_list += text_generated

df = pd.DataFrame(data={'generated': text_generated_list})
df.to_csv(out_dir_generated+model_name+'_prompt3_run{}.csv'.format(run), index=False)


prompt = "<|prompt|>You have just read the article, 'A Cowboy Who Rode the Waves.' Luke's participation in the Seagoing Cowboys program allowed him to experience adventures and visit many unique places. Using information from the article, write an argument from Luke's point of view convincing others to participate in the Seagoing Cowboys program. Be sure to include: reasons to join the program; details from the article to support Luke's claims; an introduction, a body, and a conclusion to your essay. <|answer|>"

prompt_list = [prompt]*num_generated_per_prompt
text_generated_list = []
batch_size = 25
for i in range(0, len(prompt_list), batch_size):
    print(i, i+batch_size)
    inputs = tokenizer(prompt_list[i:i+batch_size], return_tensors="pt", add_special_tokens=False).to("cuda")
    tokens = model.generate(**inputs, max_new_tokens=1024, do_sample=True, temperature=1.0)
    tokens = tokens[:, inputs["input_ids"].shape[1]:]
    text_generated = tokenizer.batch_decode(tokens, skip_special_tokens=True)
    text_generated_list += text_generated

df = pd.DataFrame(data={'generated': text_generated_list})
df.to_csv(out_dir_generated+model_name+'_prompt4_run{}.csv'.format(run), index=False)


prompt = "<|prompt|>In the article “Driverless Cars are Coming,” the author presents both positive and negative aspects of driverless cars. Using details from the article, create an argument for or against the development of these cars.  Be sure to include: your position on driverless cars; appropriate details from the article that support your position; an introduction, a body, and a conclusion to your argumentative essay. <|answer|>"

prompt_list = [prompt]*num_generated_per_prompt
text_generated_list = []
batch_size = 25
for i in range(0, len(prompt_list), batch_size):
    print(i, i+batch_size)
    inputs = tokenizer(prompt_list[i:i+batch_size], return_tensors="pt", add_special_tokens=False).to("cuda")
    tokens = model.generate(**inputs, max_new_tokens=1024, do_sample=True, temperature=1.0)
    tokens = tokens[:, inputs["input_ids"].shape[1]:]
    text_generated = tokenizer.batch_decode(tokens, skip_special_tokens=True)
    text_generated_list += text_generated

df = pd.DataFrame(data={'generated': text_generated_list})
df.to_csv(out_dir_generated+model_name+'_prompt5_run{}.csv'.format(run), index=False)


