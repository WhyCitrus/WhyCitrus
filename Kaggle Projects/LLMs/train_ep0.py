import argparse
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch
import random
import pickle
from torch.cuda.amp import autocast, GradScaler
import time
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_linear_schedule_with_warmup
import copy
import re

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class DAIGTDataset(Dataset):
    def __init__(self, text_list, tokenizer, max_len, label_list):
        self.text_list=text_list
        self.tokenizer=tokenizer
        self.max_len=max_len
        self.label_list=label_list
    def __len__(self):
        return len(self.text_list)
    def __getitem__(self, index):
        text = self.text_list[index]
        label = self.label_list[index]
        tokenized = self.tokenizer(text=text,
                                   padding='max_length',
                                   truncation=True,
                                   max_length=self.max_len,
                                   return_tensors='pt')
        return tokenized['input_ids'].squeeze(), tokenized['attention_mask'].squeeze(), label

class DAIGTModel(nn.Module):
    def __init__(self, model_path, config, tokenizer, pretrained=False):
        super().__init__()
        if pretrained:
            self.model = AutoModel.from_pretrained(model_path, config=config)
        else:
            self.model = AutoModel.from_config(config)
        self.classifier = nn.Linear(config.hidden_size, 1)   
    def forward_features(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        embeddings = sum_embeddings / sum_mask
        return embeddings
    def forward(self, input_ids, attention_mask):
        embeddings = self.forward_features(input_ids, attention_mask)
        logits = self.classifier(embeddings)
        return logits

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    args = parser.parse_args()
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend="nccl")
    args.device = device

    seed = 3001
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # prepare input
    with open('../preprocessing/split1/train_neg_list.pickle', 'rb') as f:
        neg_list = pickle.load(f)
    with open('../preprocessing/split1/train_pos_list.pickle', 'rb') as f:
        pos_list = pickle.load(f)
    text_list = neg_list + pos_list
    label_list = [0]*len(neg_list) + [1]*len(pos_list)
    print(len(neg_list), len(pos_list))

    # hyperparameters
    learning_rate = 0.00001
    max_len = 768
    batch_size = 8
    num_epoch = 1
    model_path = 'microsoft/deberta-large'
    mode_save_iter = 5000

    # build model
    if args.local_rank != 0:
        torch.distributed.barrier()
    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = DAIGTModel(model_path, config, tokenizer, pretrained=True)
    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    num_train_steps = int(len(text_list)/(batch_size*3)*num_epoch)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)

    # training
    train_datagen = DAIGTDataset(text_list, tokenizer, max_len, label_list)
    train_sampler = DistributedSampler(train_datagen)
    train_generator = DataLoader(dataset=train_datagen,
                                 sampler=train_sampler,
                                 batch_size=batch_size,
                                 num_workers=8,
                                 pin_memory=False)

    if args.local_rank == 0:
        start_time = time.time()

    scaler = GradScaler()
    for ep in range(num_epoch):
        losses = AverageMeter()
        model.train()
        for j, (batch_input_ids, batch_attention_mask, batch_labels) in enumerate(train_generator):
            batch_input_ids = batch_input_ids.to(args.device)
            batch_attention_mask = batch_attention_mask.to(args.device)
            batch_labels = batch_labels.float().to(args.device)

            with autocast():
                logits = model(batch_input_ids, batch_attention_mask)
                loss = nn.BCEWithLogitsLoss()(logits.view(-1), batch_labels)

            losses.update(loss.item(), batch_input_ids.size(0))

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if args.local_rank == 0:
                print('\r',end='',flush=True)
                message = '%s %5.1f %6.1f    |     %0.3f     |' % ("train",j/len(train_generator)+ep,ep,losses.avg)
                print(message , end='',flush=True)

        if args.local_rank == 0:
            print('epoch: {}, train_loss: {}'.format(ep, losses.avg), flush=True)

    if args.local_rank == 0:
        out_dir = 'weights/'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        torch.save(model.module.state_dict(), out_dir+'weights_ep{}'.format(ep))

    if args.local_rank == 0:
        end_time = time.time()
        print(end_time-start_time)

if __name__ == "__main__":
    main()
