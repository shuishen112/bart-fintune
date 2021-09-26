import os
os.environ['http_proxy']="http://star-proxy.oa.com:3128"
os.environ['https_proxy']="http://star-proxy.oa.com:3128"

import pyterrier as pt
import pandas as pd 
from pyterrier.measures import * 
import argparse
from gensim.utils import tokenize as gensim_tokenize

import nltk
# nltk.download("stopwords")

pt.init(version = 5.5,helper_version = "0.0.6", home_dir = "/data/ceph/zhansu/data/msmarco")
# pt.logging("INFO")

# from nltk.corpus import stopwords
# en_stop_words = stopwords.words("english")

# *************************获得trec-covid 数据集合


# indexer = pt.index.IterDictIndexer('./cord19-index')
# indexref = indexer.index(dataset.get_corpus_iter(), fields=('title', 'abstract'))
# index = pt.IndexFactory.of(indexref)
# # 输出最终的数据统计
# print(index.getCollectionStatistics().toString())

# 也可以直接进行文件的索引
# index = pt.IndexFactory.of("./cord19-index")
# print(index.getCollectionStatistics().toString())

# 构建检索模型
# DPH_br = pt.BatchRetrieve(index, wmodel="DPH") % 100
# BM25_br = pt.BatchRetrieve(index, wmodel="BM25") % 100
# # this runs an experiment to obtain results on the TREC COVID queries and qrels
# print(pt.Experiment(
#     [DPH_br, BM25_br],
#     dataset.get_topics('title'),
#     dataset.get_qrels(),
#     eval_metrics=["P.5", "P.10", "ndcg_cut.10", "map"]))



#************************ 进行textscorer的测试 ******************************

df = pd.DataFrame(
    [
        ["q1", "chemical reactions", "d1", "professor protor poured the chemicals"],
        ["q1", "chemical reactions",  "d2", "chemical brothers turned up the beats"],
    ], columns=["qid", "query", "docno","text"])

print(df)
textscorer = pt.batchretrieve.TextScorer(takes="docs", body_attr="text", wmodel="BM25")
rtr = textscorer.transform(df)
print(rtr)

exit()
# data = pd.read_csv("/root/program/WikiQACorpus/WikiQA-train.tsv",sep = '\t',quoting = 3)
# print(data.head)
# train_data = data[['QuestionID','Question','SentenceID','Sentence']]

# train_data.columns = ['qid','query','docno','text']

# textscorer = pt.batchretrieve.TextScorer(takes="docs", body_attr="text", wmodel="BM25")
# retrieval = train_data[train_data['qid'].isin(['Q2','Q5'])]
# print(retrieval)

# lsrtr = textscorer.transform(train_data)
# print(lsrtr.columns)


# ************************ 对msmarco doc ranking 进行数据集合审核 ******************

# # 对msmarco 数据集合训练
# dataset = pt.datasets.get_dataset("trec-deep-learning-docs")

# props = {
#   'indexer.meta.reverse.keys':'docno',
#   'termpipelines' : '',
# }
# pt.logging('INFO')

# indexer = pt.TRECCollectionIndexer("./msmarco_passage_index")
# indexer.setProperties(**props)
# # this downloads the file msmarco-docs.trec.gz
# indexref = indexer.index(dataset.get_corpus())
# index = pt.IndexFactory.of(indexref)
# print(index.getCollectionStatistics().toString())


# index = pt.IndexFactory.of("./msmarco_index")
# print(index.getCollectionStatistics().toString())

# BM25_br = pt.BatchRetrieve(index, wmodel="BM25") % 100

# # this runs an experiment to obtain results on the TREC 2019 Deep Learning track queries and qrels
# result = pt.Experiment(
#     [BM25_br],
#     dataset.get_topics("leaderboard-2020"),
    
#     dataset.get_qrels("test-2020"),
#     eval_metrics=["recip_rank", "ndcg_cut_10", "map"])
# print(result)

# ************************对单个query进行检索******************
print(BM25_br.search("Light"))


print(dataset.get_qrels("train"))

topics = dataset.get_topics("test-2020")

res = BM25_br.transform(topics)
print(res.head())


qrels = dataset.get_qrels("test-2020")
eval = pt.Utils.evaluate(res,qrels,metrics = ['map'], perquery = True)
print(len(eval))
print(len(topics))


# ************************对Msmarco Passage Ranking ******************

dataset = pt.datasets.get_dataset("trec-deep-learning-passages")
def msmarco_generate():
    with pt.io.autoopen(dataset.get_corpus()[0], "rt") as corpusfile:
        for l in corpusfile:
            docno, passage = l.split("\t")
            yield {"docno": docno, "text": passage}
# iter_indexer = pt.IterDictIndexer("./passage_index_50", threads = 50)
# indexref3 = iter_indexer.index(msmarco_generate(), meta=["docno","text"], meta_lengths = [20,4096])

index = pt.IndexFactory.of("./passage_index_50")
print(index.getCollectionStatistics().toString())

BM25_br = pt.BatchRetrieve(index, metadata = ["docno","text"], wmodel="BM25") % 10


querys = dataset.get_topics("train")[:10]
qrels = dataset.get_qrels("train")


# # 测试单个query
# res = BM25_br.transform(querys)
# print(res.head())

# eval = pt.Utils.evaluate(res,qrels,metrics = ['map'], perquery = True)
# print(eval)


# for _, row in querys.iterrows():
#     query = row['query']
#     res = BM25_br.search("query")
#     print(res)


# qrels = dataset.get_qrels("train")
# print("len qrels",len(qrels))
# print(qrels.head())

# eval_result = pt.Utils.evaluate(res,qrels,metrics = ['map'])
# print(eval_result)

# 测试数据

# qrels = dataset.get_qrels("dev.small")

# querys = dataset.get_topics("dev.small")
# # print(querys)
# result = pt.Experiment(
#     [BM25_br],
#     querys,
#     qrels,
#     eval_metrics=[RR(rel = 1)])


# # print(len(dataset.get_topics("dev")))
# print(len(dataset.get_topics("dev.small")))

######################### 载入T5模型，用T5模型生成query################

from IR_transformers.modeling_t5 import T5ForConditionalGeneration
from IR_transformers.tokenization_t5 import T5Tokenizer

from torch.utils.data import Dataset, DataLoader
from nlp import Dataset as nlp_dataset

model = T5ForConditionalGeneration.from_pretrained("./msmarco_finetune_t5")
tokenizer = T5Tokenizer.from_pretrained("./msmarco_finetune_t5")

querys = dataset.get_topics("dev.small")[:100]

class msmarco_test(Dataset):
    def __init__(self, tokenizer, input_length, num_samples):
        self.input_length = input_length
        self.dataset = nlp_dataset.from_pandas(querys)
        self.tokenizer = tokenizer

        if num_samples:
            self.dataset = self.dataset.select(list(range(0,num_samples)))
    def __len__(self):
        return self.dataset.shape[0]

    def clean_text(self, text):
        
        text = text.replace('Example of text:', '')
        text = text.replace('Example of Summary:', '')
        text = text.replace('\n', '')
        text = text.replace('``', '')
        text = text.replace('"', '')
        return text

    def convert_to_features(self, example_batch):

        input_ = self.clean_text(example_batch['query'])

        source = self.tokenizer.batch_encode_plus([input_], max_length=self.input_length,
                                                  padding="max_length", truncation=True, return_tensors="pt")
        return input_, source, example_batch['qid']

    def __getitem__(self, index):
        input_, source, qid = self.convert_to_features(self.dataset[index])
        source_ids = source["input_ids"].squeeze()
        src_mask = source["attention_mask"].squeeze()
        return {"input":input_,"source_ids": source_ids, "source_mask": src_mask,"qid":qid}

args_dict = dict(
    max_input_length = 15,
    test_batch_size = 5
)

args = argparse.Namespace(**args_dict)

test_dataset = msmarco_test(tokenizer = tokenizer,input_length = args.max_input_length, num_samples = 20)
test_dataloader = DataLoader(test_dataset, batch_size = args.test_batch_size)


def clean(text):
    text = text.lower()
    tokens = gensim_tokenize(text)
    text = " ".join(tokens)
    return text
def lmap(f, x):
    """list(map(f, x))"""
    return list(map(f, x))

def ids_to_clean_text(generated_ids):
    gen_text = tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    
    return lmap(clean, gen_text)

qids = []
original_querys = []
query_rewrite = []

for batch in test_dataloader:

    generated_ids = model.generate(
            batch["source_ids"],
            attention_mask=batch["source_mask"],
            use_cache=True,
            max_length=15,
            num_beams=2,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True
        )
    pred = ids_to_clean_text(generated_ids)

    for i in range(len(batch)):
        print("原始query",batch['input'][i])
        print("重写query",pred[i])

    print(["*"]*20)

    qids.extend(batch['qid'])
    original_querys.extend(batch['input'])
    query_rewrite.extend(pred)

df_pred = pd.DataFrame({"qid":qids,"query_rewrite":query_rewrite,"original_querys":original_querys})

df_pred["merge_query"] = df_pred['original_querys'] + " " + df_pred["query_rewrite"]


df_query_test_origin = df_pred[['qid',"original_querys"]].rename(columns = {"qid":"qid", "original_querys":"query"})

df_query_test_rewirte = df_pred[['qid',"merge_query"]].rename(columns = {"qid":"qid","merge_query":"query"})

print(df_query_test_origin)
print(df_query_test_rewirte)


# def generate_query(row):
#     original_query = row['query']
#     input_ids = tokenizer(original_query,return_tensors = "pt").input_ids
#     beam_output = model.generate(
#     input_ids,
#     max_length=10,
#     num_beams=5,
#     early_stopping=True
#     )

#     text = tokenizer.decode(beam_output[0], skip_special_tokens=True)
#     tokens = gensim_tokenize(text)
#     # tokens = [t for t in tokens if t not in en_stop_words]
#     return " ".join(tokens)




# querys["query_write"] = querys.apply(generate_query,axis = 1)
# querys.columns = ["qid","original_query","query"]

# for e,row in querys.iterrows():
#     print("原始query",row['query'])
#     print("重写query",row["query_write"])


qrel = dataset.get_qrels("dev.small")

res_origin = BM25_br.transform(df_query_test_origin)
print(res_origin.head())



eval_origin = pt.Utils.evaluate(res_origin,dataset.get_qrels("dev.small"),metrics = ['map'], perquery = True)
print("eval_origin",len(eval_origin))

rewrite_origin =  BM25_br.transform(df_query_test_rewirte)
eval_rewrite = pt.Utils.evaluate(rewrite_origin,dataset.get_qrels("dev.small"),metrics = ['map'], perquery = True)
print("eval_rewrite",len(eval_rewrite))

for e,qid in enumerate(df_pred['qid'].to_list()):
    print(e, eval_origin[qid],eval_rewrite[qid])

result = pt.Experiment(
    [BM25_br],
    df_query_test_origin,
    dataset.get_qrels("dev.small"),
    eval_metrics=[RR(rel = 1)])

print(result)

result = pt.Experiment(
    [BM25_br],
    df_query_test_rewirte,
    dataset.get_qrels("dev.small"),
    eval_metrics=[RR(rel = 1)])

print(result)
