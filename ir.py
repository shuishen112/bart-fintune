import pyterrier as pt
import pandas as pd 
pt.init()
# pt.logging("INFO")

# *************************获得trec-covid 数据集合

# dataset = pt.datasets.get_dataset('irds:cord19/trec-covid')

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

# df = pd.DataFrame(
#     [
#         ["q1", "chemical reactions", "d1", "professor protor poured the chemicals"],
#         ["q1", "chemical reactions",  "d2", "chemical brothers turned up the beats"],
#     ], columns=["qid", "query", "docno","text"])

# print(df)
# textscorer = pt.batchretrieve.TextScorer(takes="docs", body_attr="text", wmodel="BM25")
# rtr = textscorer.transform(df)
# print(rtr)

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
# print(BM25_br.search("Light"))


# print(dataset.get_qrels("train"))

# topics = dataset.get_topics("test-2020")

# res = BM25_br.transform(topics)
# print(res.head())


# qrels = dataset.get_qrels("test-2020")
# eval = pt.Utils.evaluate(res,qrels,metrics = ['map'], perquery = True)
# print(len(eval))
# print(len(topics))


# ************************对Msmarco Passage Ranking ******************

dataset = pt.datasets.get_dataset("trec-deep-learning-passages")
def msmarco_generate():
    with pt.io.autoopen(dataset.get_corpus()[0], "rt") as corpusfile:
        for l in corpusfile:
            docno, passage = l.split("\t")
            yield {"docno": docno, "text": passage}
# iter_indexer = pt.IterDictIndexer("./passage_index_8", threads = 8)
# indexref3 = iter_indexer.index(msmarco_generate(), meta=["docno","text"], meta_lengths = [20,4096])

index = pt.IndexFactory.of("./passage_index_8")
print(index.getCollectionStatistics().toString())

BM25_br = pt.BatchRetrieve(index, wmodel="BM25") % 10

# 测试单个query
# res = BM25_br.search("are you ok")
# print(res)

# qrels = dataset.get_qrels("train")
# print("len qrels",len(qrels))
# print(qrels.head())

# eval_result = pt.Utils.evaluate(res,qrels,metrics = ['map'])
# print(eval_result)

# 测试数据

# from pyterrier.measures import * 
# result = pt.Experiment(
#     [BM25_br],
#     dataset.get_topics("dev.small"),
    
#     dataset.get_qrels("dev.small"),
#     eval_metrics=[RR(rel = 1)])


# # print(len(dataset.get_topics("dev")))
# print(len(dataset.get_topics("dev.small")))
# print(result)

######################### 载入T5模型，用T5模型生成query################



from IR_transformers.modeling_t5 import T5ForConditionalGeneration
from IR_transformers.tokenization_t5 import T5Tokenizer

model = T5ForConditionalGeneration.from_pretrained("./t5_finetune_model")
tokenizer = T5Tokenizer.from_pretrained("./t5_finetune_model")

input_ids = tokenizer('how are glacier caves formed ?', return_tensors='pt').input_ids
labels = tokenizer('A glacier cave is a cave formed within the ice of a glacier .', return_tensors='pt').input_ids

# loss = model(input_ids = input_ids, labels = labels).loss
# print("loss:", loss)

def generate_query(row):
    original_query = row['query']
    print(original_query)
    input_ids = tokenizer(original_query,return_tensors = "pt").input_ids
    beam_output = model.generate(
    input_ids,
    max_length=10,
    num_beams=5,
    early_stopping=True
    )

    return tokenizer.decode(beam_output[0], skip_special_tokens=True)

querys = dataset.get_topics("dev.small")[:10]

querys["query_write"] = querys.apply(generate_query,axis = 1)
querys.columns = ["qid","original_query","query"]
print(querys.head())
print(len(querys))

from pyterrier.measures import * 
result = pt.Experiment(
    [BM25_br],
    querys,
    dataset.get_qrels("dev.small"),
    eval_metrics=[RR(rel = 1)])

print(result)