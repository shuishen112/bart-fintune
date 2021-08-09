import argparse
import glob
import os
import json
import time
import logging
import random
import re
from itertools import chain
from string import punctuation

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.loggers import WandbLogger
from nlp import load_metric
from IR_transformers.tokenization_t5 import T5Tokenizer
from IR_transformers.modeling_t5 import T5ForConditionalGeneration
from nlp import load_dataset
from nlp import list_datasets

class wikihow(Dataset):
    def __init__(self, tokenizer, type_path, num_samples, input_length, output_length, print_text=False):         
        self.dataset =  load_dataset('wikihow', 'all', data_dir='../summary_dataset/', split=type_path)
        if num_samples:
            self.dataset = self.dataset.select(list(range(0, num_samples)))
        self.input_length = input_length
        self.tokenizer = tokenizer
        self.output_length = output_length
        self.print_text = print_text
  
    def __len__(self):
        return self.dataset.shape[0]
    
    def clean_text(self, text):
        text = text.replace('Example of text:', '')
        text = text.replace('Example of Summary:', '')
        text = text.replace('\n','')
        text = text.replace('``', '')
        text = text.replace('"', '')
        
        return text
    
    
    def convert_to_features(self, example_batch):
        # Tokenize contexts and questions (as pairs of inputs)
        
        if self.print_text:
            print("Input Text: ", self.clean_text(example_batch['text']))
#         input_ = self.clean_text(example_batch['text']) + " </s>"
#         target_ = self.clean_text(example_batch['headline']) + " </s>"
        
        input_ = self.clean_text(example_batch['text'])
        target_ = self.clean_text(example_batch['headline'])
        
        source = self.tokenizer.batch_encode_plus([input_], max_length=self.input_length, 
                                                     padding='max_length', truncation=True, return_tensors="pt")
        
        targets = self.tokenizer.batch_encode_plus([target_], max_length=self.output_length, 
                                                     padding='max_length', truncation=True, return_tensors="pt")
    
       
        return source, targets
  
    def __getitem__(self, index):
        source, targets = self.convert_to_features(self.dataset[index])
        
        source_ids = source["input_ids"].squeeze()
        target_ids = targets["input_ids"].squeeze()

        src_mask = source["attention_mask"].squeeze()
        target_mask = targets["attention_mask"].squeeze()

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}

datasets_list = list_datasets()
print(', '.join(dataset.id for dataset in datasets_list))

dataset = load_dataset("spacemanidol/msmarco_passage_ranking")

## Load dataset using NLP

'''
datasets_list = list_datasets()
print(', '.join(dataset.id for dataset in datasets_list))

dataset = load_dataset('wikihow', 'all', data_dir='../summary_dataset/')
print("Size of train dataset: ", dataset['train'].shape)
print("Size of Validation dataset: ", dataset['validation'].shape)
print("Size of ca test dataset: ", dataset['test'].shape)
print(dataset['train'][0].keys())

# 测试一下数据
tokenizer = T5Tokenizer.from_pretrained('t5-small')
dataset = wikihow(tokenizer, "validation", None, 512, 150, True)
print(len(dataset))

data = dataset[50]
print()
print("Shape of Tokenized Text: ", data['source_ids'].shape)
print()
print("Sanity check - Decode Text: ", tokenizer.decode(data['source_ids']))
print("====================================")
print("Sanity check - Decode Summary: ", tokenizer.decode(data['target_ids']))
'''

# 获取arguments

args_dict = dict(
    output_dir="", # path to save the checkpoints
    model_name_or_path='t5-small',
    tokenizer_name_or_path='t5-small',
    max_input_length=512,
    max_output_length=150,
    freeze_encoder=False,
    freeze_embeds=False,
    learning_rate=3e-4,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    warmup_steps=0,
    train_batch_size=4,
    eval_batch_size=4,
    num_train_epochs=2,
    gradient_accumulation_steps=8,
    n_gpu=1,
    resume_from_checkpoint=None, 
    val_check_interval = 0.05, 
    n_val=1000,
    n_train=-1,
    n_test=-1,
    early_stop_callback=False,
    fp_16=False, # if you want to enable 16-bit training then install apex and set this to true
    opt_level='O1', # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
    max_grad_norm=1.0, # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
    seed=42,
)

tokenizer = T5Tokenizer.from_pretrained('t5-small')

# 获取数据
args_dict.update({'output_dir': 't5_wikihow', 'num_train_epochs':2,
                 'train_batch_size': 4, 'eval_batch_size': 4})
args = argparse.Namespace(**args_dict)
print(args_dict)


def get_dataset(tokenizer, type_path, num_samples, args):
      return wikihow(tokenizer=tokenizer, type_path=type_path, num_samples=num_samples,  input_length=args.max_input_length, 
                        output_length=args.max_output_length)


train_dataset = get_dataset(tokenizer = tokenizer, type_path="train", num_samples=2000, args = args)
dataloader = DataLoader(train_dataset,batch_size=args.train_batch_size)

# 进行训练
model = T5ForConditionalGeneration.from_pretrained("t5-small")


# 构建optimizer
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

model.train()

if torch.cuda.is_available():
    model.cuda()


for epoch in range(args.num_train_epochs):
    # train loop
    for e, train_batch in enumerate(dataloader):
        lm_labels = train_batch["target_ids"]
        lm_labels[lm_labels[:,:] == tokenizer.pad_token_id] = -100


        outputs = model(input_ids = train_batch["source_ids"].cuda(),
        attention_mask = train_batch['source_mask'].cuda(),
        labels = lm_labels.cuda(),
        decoder_attention_mask = train_batch['target_mask'].cuda()
        )

        loss = outputs[0]


        # 1 计算反向传播的值
        loss.backward()
        # 2 反向传播更新梯度
        optimizer.step()
        # 3 将梯度清零
        optimizer.zero_grad()
        print(loss.item())

    