import os

from IR_transformers.modeling_t5 import T5ForConditionalGeneration
from IR_transformers.tokenization_t5 import T5Tokenizer
from transformers import BartForConditionalGeneration, BartTokenizer

import torch
import logging
import pyterrier as pt
from pyterrier.measures import * 
pt.init() 

m = "bart-large"

if m == "t5-small":
    model = T5ForConditionalGeneration.from_pretrained("/data/lxk/DownloadPretrainedModel/t5-small")
    tokenizer = T5Tokenizer.from_pretrained("/data/lxk/DownloadPretrainedModel/t5-small")
elif m == "bart-large":
    model = BartForConditionalGeneration.from_pretrained("/data/lxk/DownloadPretrainedModel/bart-large")
    tokenizer = BartTokenizer.from_pretrained("/data/lxk/DownloadPretrainedModel/bart-large")


labels = tokenizer('A glacier cave is a cave formed within the ice of a glacier .', return_tensors='pt').input_ids

# dataset = pt.datasets.get_dataset("trec-deep-learning-passages")
# index = pt.IndexFactory.of("./passage_index_50")
# print(index.getCollectionStatistics().toString())

# BM25_br = pt.BatchRetrieve(index, metadata = ["docno","text"], wmodel="BM25") % 10

# result = pt.Experiment(
#     [BM25_br],
#     dataset.get_topics("dev.small"),
    
#     dataset.get_qrels("dev.small"),
#     eval_metrics=[RR(rel = 1)])


# # print(len(dataset.get_topics("dev")))
# print(len(dataset.get_topics("dev.small")))
# print(result)


input_ids = tokenizer('how are glacier caves formed ?', return_tensors='pt').input_ids
labels = tokenizer('A glacier cave is a cave formed within the ice of a glacier .', return_tensors='pt').input_ids
# the forward function automatically creates the correct decoder_input_ids

FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level = logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

# beam search
beam_output = model.generate(
    input_ids,
    max_length=50,
    num_beams=5,
    early_stopping=True
)

print(tokenizer.decode(beam_output[0], skip_special_tokens=True))



# print("T5 loss",loss)

# from IR_transformers.modeling_bart import BartForConditionalGeneration
# from IR_transformers.tokenization_bart import BartTokenizer

# model_bart = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
# tokenizer_bart = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

# loss_bart = model_bart(input_ids=input_ids,labels = labels).loss
# print("Bart loss",loss_bart)



