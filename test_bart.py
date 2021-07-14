'''
Author: your name
Date: 2021-07-13 20:14:35
LastEditTime: 2021-07-14 10:49:04
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /bart-fintune/test_bart.py
'''
from modeling_bart import BartForConditionalGeneration
from tokenization_bart import BartTokenizer

from transformers import AdamW
import torch 

import pandas as pd 
import sklearn

from gensim import corpora
from gensim.summarization import bm25

df_train = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/wiki/train.tsv",sep = '\t',quoting = 3,names = ['question','answer','flag'])
print(df_train.head())

model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

def query_expansion(query):
    querys = []
    inputs = tokenizer([query], max_length=1024, return_tensors='pt')
    outputs = model.generate(inputs['input_ids'], num_return_sequences = 5, num_beams=5, max_length=50, early_stopping=True,output_scores = True,return_dict_in_generate =True)
    sequences = outputs['sequences']
    sequences_scores = outputs['sequences_scores']
    print(sequences_scores)
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
    group = sklearn.utils.shuffle(group,random_state =132)
    candidates = group.sort_values(by = 'score',ascending = False).reset_index()
    correct_candidates = candidates[candidates['flag'] == 1]

    if len(correct_candidates) == 0:
        return 0
    
    for i,index in enumerate(correct_candidates.index):
        ap += 1.0 * (i + 1) / (index + 1)
    return ap / len(correct_candidates)

# ARTICLE_TO_SUMMARIZE = "My friends are cool but they eat too many carbs."
# query_output = query_expansion(ARTICLE_TO_SUMMARIZE)

# for query,score in zip(*query_output):
#     print(query,score)

# dataset

# 设置loss
loss_fn = torch.nn.MSELoss(reduce=False, size_average = False)

# optimizer

optimizer = AdamW(model.parameters(), lr = 5e-5)

# training

model.train()

for question in df_train['question'].unique()[:5]:
    # get the group

    group = df_train[df_train['question'] == question].reset_index()

    query_output = query_expansion(question)

    scores = []
    map_scores = []
    

    for query,score in zip(*query_output):
        map_score = get_bm25_score(query,group)
        scores.append(score)
        map_scores.append(map_score)

    scores = torch.tensor(scores)
    print(scores)
    loss = loss_fn(scores, torch.tensor(map_scores))

    # 反向传播计算梯度
    loss.backward()

    # 根据梯度更新参数
    optimizer.step()

    # 清空grad值
    optimizer.zero_grad()