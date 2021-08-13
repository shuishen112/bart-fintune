# 这代码的主要目的是finetune T5, msmarco_passage_ranking 的 query document pair 是我们的原始数据集
# 我们用query去生成document

from nlp import load_dataset
from nlp import Dataset as nlp_dataset
import pandas as pd 
from IR_transformers.modeling_t5 import T5ForConditionalGeneration
from IR_transformers.tokenization_t5 import T5Tokenizer
import argparse
from torch.utils.data import Dataset, DataLoader,random_split
import torch
import numpy as np 
import logging

FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level = logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

# dataset = load_dataset('squad', split='train')
# print(dataset)
df = pd.read_csv("/data/ceph/zhansu/data/msmarco/doc_query_pairs.train.tsv",sep = '\t',names = ["doc","query"])

msk = np.random.rand(len(df)) < 0.8

df_train = df[msk]
df_test = df[~msk]

train_dataset = nlp_dataset.from_pandas(df_train)
test_dataset = nlp_dataset.from_pandas(df_test)

print("train_number",len(train_dataset))
print("test_number",len(test_dataset))




model = T5ForConditionalGeneration.from_pretrained("/data/ceph/zhansu/embedding/t5-small")
tokenizer = T5Tokenizer.from_pretrained("/data/ceph/zhansu/embedding/t5-small")

class msmarco(Dataset):
    def __init__(self,tokenizer, type_path, num_samples, input_length, output_length, print_text = False):
        # df = pd.read_csv("/root/program/ir_data/doc_query_pairs.train.tsv",sep = '\t',names = ["doc","query"])
        # self.dataset = nlp_dataset.from_pandas(df)

        if type_path == 'train':
            self.dataset = nlp_dataset.from_pandas(df_train)
        elif type_path == "test":
            self.dataset = nlp_dataset.from_pandas(df_test)

        if num_samples:
            self.dataset = self.dataset.select(list(range(0,num_samples)))

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
        if self.print_text:
           print("Input Text: ", self.clean_text(example_batch['doc'])) 

        input_ = self.clean_text(example_batch['query'])
        target_ = self.clean_text(example_batch['doc'])

        source = self.tokenizer.batch_encode_plus([input_],max_length = self.input_length,
        padding = "max_length",truncation = True, return_tensors = "pt")

        targets = self.tokenizer.batch_encode_plus([target_],max_length = self.output_length,
        padding = "max_length", truncation = True, return_tensors = "pt")

        return source, targets

    def __getitem__(self,index):
        
        source, targets = self.convert_to_features(self.dataset[index])

        source_ids = source["input_ids"].squeeze()
        target_ids = targets["input_ids"].squeeze()

        src_mask = source["attention_mask"].squeeze()
        target_mask = targets["attention_mask"].squeeze()

        return {"source_ids": source_ids,
        "source_mask":src_mask,
        "target_ids":target_ids,
        "target_mask":target_mask}

def get_dataset(tokenizer, type_path, num_samples, args):
    return msmarco(tokenizer = tokenizer, type_path = "train", 
    num_samples=num_samples, input_length=args.max_input_length,
    output_length = args.max_output_length)

args_dict = dict(
    max_input_length = 150,
    max_output_length = 512,
    train_batch_size = 25,
    test_batch_size = 25,
    num_train_epochs = 2
)

args = argparse.Namespace(**args_dict)


train_dataset = get_dataset(tokenizer = tokenizer,type_path= "train",num_samples = None, args = args)
test_dataset = get_dataset(tokenizer = tokenizer,type_path = "test", num_samples = None, args = args)

train_dataloader = DataLoader(train_dataset, batch_size = args.train_batch_size)
test_dataloader = DataLoader(test_dataset, batch_size = args.test_batch_size)

# 构建optimizer
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

model.train()

if torch.cuda.is_available():
    model.cuda()


for epoch in range(args.num_train_epochs):
    logger.info("epoch:{}".format(epoch))
    model.train()
    # train loop
    for e, train_batch in enumerate(train_dataloader):
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
        logger.info("train loss:{}".format(loss.item()))

    # eval model
    model.eval()
    with torch.no_grad():
        test_loss = []
        for test_batch in test_dataloader:
            lm_labels = test_batch["target_ids"]
            lm_labels[lm_labels[:,:] == tokenizer.pad_token_id] = -100


            outputs = model(input_ids = test_batch["source_ids"].cuda(),
            attention_mask = test_batch['source_mask'].cuda(),
            labels = lm_labels.cuda(),
            decoder_attention_mask = test_batch['target_mask'].cuda()
            )

            loss = outputs[0]
            test_loss.append(loss)

        logger.info("test loss:{}".format(torch.mean(torch.tensor(test_loss))))
    model.save_pretrained("./t5_finetune_model")
        