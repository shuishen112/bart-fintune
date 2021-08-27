'''
Author: your name
Date: 2021-07-13 20:14:35
LastEditTime: 2021-07-14 10:49:04
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /bart-fintune/test_bart.py
'''
from IR_transformers.modeling_bart import BartForConditionalGeneration
from IR_transformers.tokenization_bart import BartTokenizer
from IR_transformers.modeling_t5 import T5ForConditionalGeneration
from IR_transformers.tokenization_t5 import T5Tokenizer


from transformers import AdamW
import torch

import pandas as pd 
import sklearn


from gensim import corpora
from gensim.summarization import bm25
from torch.utils.data import Dataset


model_name = "t5"
df_train = pd.read_csv("/root/program/wiki/train.txt",sep = '\t',quoting = 3,names = ['question','answer','flag'])
print(df_train.head())

if model_name == 't5':
    model = T5ForConditionalGeneration.from_pretrained("t5_finetune_model")
    tokenizer = T5Tokenizer.from_pretrained("t5_finetune_model")
elif model_name == 'bart':
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
else:
    print("invalid model name")

# model for gpu

# if torch.cuda.is_available():
#     model.cuda()


def query_expansion(query,return_sequence = 2):
    querys = []
    inputs = tokenizer([query], max_length=1024, return_tensors='pt').to(model.device)
    outputs = model.generate(inputs['input_ids'], num_return_sequences = return_sequence, do_sample = True,max_length=50, top_k=50, top_p = 0.95,output_scores = True,return_dict_in_generate =True)
    sequences = outputs['sequences']
    sequences_scores = outputs['sequences_scores']

    for s in sequences:
        querys.append(tokenizer.decode(s, skip_special_tokens = True, clean_up_tokenization_spaces = False))
    return (querys,sequences_scores)



def get_bm25_score(query,group):

    # get bm25
    texts = [doc.split() for doc in group['answer']]
    dictionary = corpora.Dictionary(texts)

    corpus = [dictionary.doc2bow(text) for text in texts]
    bm25_obj = bm25.BM25(corpus)

    query_doc = dictionary.doc2bow(query.split())
    group['score'] = list(bm25_obj.get_scores(query_doc))

    # get map
    ap = 0
    group = sklearn.utils.shuffle(group,random_state = 132)
    candidates = group.sort_values(by = 'score',ascending = False).reset_index()
    correct_candidates = candidates[candidates['flag'] == 1]

    if len(correct_candidates) == 0:
        return 0
    for i,index in enumerate(correct_candidates.index):
        ap += 1.0 * (i + 1) / (index + 1)
    return ap / len(correct_candidates)

class wiki_data(Dataset):
    def __init__(self, data_path):
        df = pd.read_csv(data_path,sep = '\t',quoting = 3,names = ['question','answer','flag'])

# ARTICLE_TO_SUMMARIZE = "My friends are cool but they eat too many carbs."
# querys, sequences_scores = query_expansion(ARTICLE_TO_SUMMARIZE)

# print(querys)
# print(sequences_scores)

# dataset

# 设置loss
loss_fn = torch.nn.MSELoss()

# optimizer

optimizer = AdamW(model.parameters(), lr = 5e-5)

# training

model.train()

epoch = 100

# # this is for debug
# with torch.autograd.detect_anomaly(): 

df_train_true = df_train[df_train['flag'] == 1]
print(df_train_true)

for i in range(epoch):
    for _, item in df_train_true.iterrows():
        question = item['question']
        answer = item['answer']

        tokenizer_source = tokenizer(question,return_tensors = 'pt')
        tokenizer_target = tokenizer(answer,return_tensors = 'pt')


        # exit() 
        outputs = model(input_ids=tokenizer_source['input_ids'],
        attention_mask = tokenizer_source['attention_mask'],
        labels = tokenizer_target['input_ids'],
        decoder_attention_mask = tokenizer_target['attention_mask']
        )

        loss = outputs['loss']
        logits = outputs['logits']

        m = torch.nn.Softmax(dim = 2)
        softmax_score = m(logits)
        print(softmax_score)
        
        scores = torch.prod(torch.max(softmax_score,2).values,1)

        print(scores)
        print("bart loss",loss)







# for i in range (epoch):
#     for question in df_train['question'].unique()[:2]:
#         # get the group

#         group = df_train[df_train['question'] == question].reset_index()
#         querys,sequences_scores = query_expansion(question)
#         print(querys)
#         map_scores = []
        

#         for query in querys:
#             map_score = get_bm25_score(query,group)
#             map_scores.append(map_score)
#         # 注意这里要将tensor 转为float32
#         target_scores = torch.tensor(map_scores).to(torch.float32).to(model.device)
#             # 清空grad值
#         optimizer.zero_grad()
        
#         loss = loss_fn(sequences_scores,target_scores)
#         print("sequences_scores",sequences_scores)
#         print("target_score",target_scores)
#         print("loss",loss)

#         # 反向传播计算梯度
#         loss.backward()

#         # 根据梯度更新参数
#         optimizer.step()
       