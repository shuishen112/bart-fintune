# 对wiki的结果进行评估

import pandas as pd
import torchtext
from torch.utils.data import Dataset, DataLoader
from nlp import Dataset as nlp_dataset
from torchtext.data import get_tokenizer
from gensim.utils import tokenize as gensim_tokenize
import evaluation
import argparse

import pyterrier as pt
# pt.init(version = 5.5,helper_version = "0.0.6", home_dir = "/data/ceph/zhansu/data/msmarco") 
pt.init()


# 载入wiki数据集

def remove_the_unanswered_sample(df):
    """
    clean the dataset
            :param df: dataframe
    """
    counter = df.groupby("Question").apply(lambda group: sum(group["Label"]))
    questions_have_correct = counter[counter > 0].index
    counter = df.groupby("Question").apply(
        lambda group: sum(group["Label"] == 0))
    questions_have_uncorrect = counter[counter > 0].index
    counter = df.groupby("Question").apply(lambda group: len(group["Label"]))
    questions_multi = counter[counter > 1].index

    return df[df["Question"].isin(questions_have_correct) & df["Question"].isin(questions_have_uncorrect)].reset_index()
def load_dataset(data_name):
    
    train_data = pd.read_csv("/data/zhansu/data/WikiQACorpus/WikiQA-{}.tsv".format(data_name),sep = '\t',quoting = 3)
    train_data = remove_the_unanswered_sample(train_data)

    train_qrel = train_data[['QuestionID',"SentenceID","Label"]]
    train_qrel.columns = ['qid','docno','label']



    train_data = train_data[['QuestionID','Question','SentenceID','Sentence','Label']]

    train_data.columns = ['qid','query','docno','text','label']

    # 对query进行预处理
    def clean(row):
        text = row['query'].lower()
        tokens = list(gensim_tokenize(text))
        text = " ".join(tokens)
        return text

    train_data['query'] = train_data.apply(clean,axis = 1)

    return train_data, train_qrel




train_data, train_qrel = load_dataset("train")
test_data, test_qrel = load_dataset("test")

textscorer = pt.batchretrieve.TextScorer(takes="docs", body_attr="text", wmodel="BM25")
# 测试一下训练数据集的map
# res_train = textscorer.transform(train_data)


# train_result = pt.Utils.evaluate(res_train,train_qrel,metrics = ['map'])
# print("train_map:{}".format(train_result))

# 测试一下测试数据集的map
# res_test = textscorer.transform(test_data)
# test_result = pt.Utils.evaluate(res_test,test_qrel,metrics = ['map'])

# print("test_map:{}".format(test_result))

# print(evaluation.evaluationBypandas(res_test))


class wiki_test(Dataset):
    def __init__(self, tokenizer, input_length, num_samples):
        self.input_length = input_length
        self.dataset = nlp_dataset.from_pandas(test_data)
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
    
        return input_, source, example_batch['qid'], example_batch['docno'], example_batch['text']

    def __getitem__(self, index):
        input_, source, qid, docno, text = self.convert_to_features(self.dataset[index])
        source_ids = source["input_ids"].squeeze()
        src_mask = source["attention_mask"].squeeze()
        return {"input":input_,"source_ids": source_ids, "source_mask": src_mask, "qid":qid, "docno": docno, "text":text}

######################### 载入T5模型，用T5模型生成query################

from IR_transformers.modeling_t5 import T5ForConditionalGeneration
from IR_transformers.tokenization_t5 import T5Tokenizer

from torch.utils.data import Dataset, DataLoader
from nlp import Dataset as nlp_dataset

pre_trained_bert = "./wiki_finetune_t5"

model = T5ForConditionalGeneration.from_pretrained(pre_trained_bert)
tokenizer = T5Tokenizer.from_pretrained(pre_trained_bert)

args_dict = dict(
    max_input_length = 15,
    test_batch_size = 20
)

args = argparse.Namespace(**args_dict)

test_dataset = wiki_test(tokenizer = tokenizer,input_length = args.max_input_length, num_samples = None)
test_dataloader = DataLoader(test_dataset, batch_size = args.test_batch_size)

def clean(text):
    text = text.lower()
    tokens = list(gensim_tokenize(text))
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

original_querys = []
query_rewrite = []
qids = []
docnos = []
texts = []

the_length_of_generation = 30

for batch in test_dataloader:

    try:

        generated_ids = model.generate(
                batch["source_ids"],
                attention_mask=batch["source_mask"],
                use_cache=True,
                max_length=the_length_of_generation,
                num_beams=2,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True
            )
        pred = ids_to_clean_text(generated_ids)

        # for i in range(len(batch)):
        #     print("原始query",batch['input'][i])
        #     print("重写query",pred[i])

        # print(["*"]*20)

        original_querys.extend(batch['input'])
        query_rewrite.extend(pred)
        qids.extend(batch['qid'])
        docnos.extend(batch['docno'])
        texts.extend(batch['text'])
    except Exception as e:
        print(e)

df_pred = pd.DataFrame({"qid":qids,"query_rewrite":query_rewrite,"original_querys":original_querys,"docno":docnos,"text":texts})

df_pred["merge_query"] = df_pred['original_querys'] + " " + df_pred["query_rewrite"]


df_query_test_origin = df_pred[['qid',"original_querys","docno","text"]]

df_query_test_origin.to_csv("original_query_{}.csv".format(the_length_of_generation),index = None)
df_query_test_origin.columns = ['qid',"query","docno","text"]

df_query_test_rewirte = df_pred[['qid',"merge_query","docno","text"]]

df_query_test_rewirte.to_csv("rewrite_query_{}.csv".format(the_length_of_generation),index = None)
df_query_test_rewirte.columns = ['qid','query',"docno","text"]

print(df_query_test_origin)
print(df_query_test_rewirte)

res_origin = textscorer.transform(df_query_test_origin)
eval_origin = pt.Utils.evaluate(res_origin,test_qrel,metrics = ['map'])
print("eval_origin",eval_origin)

rewrite_origin =  textscorer.transform(df_query_test_rewirte)
eval_rewrite = pt.Utils.evaluate(rewrite_origin,test_qrel,metrics = ['map'])
print("eval_rewrite",eval_rewrite)