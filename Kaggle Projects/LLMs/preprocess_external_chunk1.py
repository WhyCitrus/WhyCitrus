import zstandard
import io
import json
import numpy as np
import random
import os
import glob
import pandas as pd
from tqdm import tqdm

out_dir_promt = 'promt/'
if not os.path.exists(out_dir_promt):
    os.makedirs(out_dir_promt)
out_dir_text = 'text/'
if not os.path.exists(out_dir_text):
    os.makedirs(out_dir_text)

prompt_word_len = 1024
text_word_len = 1024
word_len = prompt_word_len + text_word_len

promt_list = []
text_list = []

num_files = 0
file_name_idx = 0

chunk_name = 'chunk1/'

file_list = sorted(glob.glob('../../external/SlimPajama-627B/train/' + chunk_name + '*.zst'))

for file_name in tqdm(file_list, total=len(file_list)):
    num_files += 1

    with open(file_name, 'rb') as f:
        decompressor = zstandard.ZstdDecompressor(max_window_size=2147483648)
        stream_reader = decompressor.stream_reader(f)
        stream = io.TextIOWrapper(stream_reader, encoding='utf-8')
        for n, line in enumerate(stream):
            data = json.loads(line)
            words = data['text'].split(' ')
            if len(words) > word_len and data['meta']['redpajama_set_name'] == 'RedPajamaC4':
                start_idx = random.randint(0, len(words) - word_len)
                promt = ' '.join(words[start_idx:start_idx+prompt_word_len])
                text = ' '.join(words[start_idx+prompt_word_len:start_idx+word_len])
                promt_list.append(promt)
                text_list.append(text)
    
    if num_files%10 == 0:
        print(num_files, file_name_idx, len(promt_list), len(text_list))

        df = pd.DataFrame(data={'promt': promt_list})
        df.to_parquet(out_dir_promt+chunk_name[:-1]+'_{}.parquet'.format(file_name_idx), index=False)

        df = pd.DataFrame(data={'text': text_list})
        df.to_parquet(out_dir_text+chunk_name[:-1]+'_{}.parquet'.format(file_name_idx), index=False)

        file_name_idx += 1
        promt_list = []
        text_list = []


