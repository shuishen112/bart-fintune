import wandb
import pytorch_lightning as pl
from transformers import get_linear_schedule_with_warmup
from nlp import list_datasets
from nlp import load_dataset
from nlp import Dataset as nlp_dataset
# from IR_transformers.modeling_t5 import T5ForConditionalGeneration
# from IR_transformers.tokenization_t5 import T5Tokenizer
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import T5ForConditionalGeneration,T5Tokenizer
from nlp import load_metric
from torch.utils.data import Dataset, DataLoader

from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

import torch.nn as nn
import torch
import numpy as np
import pandas as pd
from gensim.utils import tokenize as gensim_tokenize
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
import wandb
import pyterrier as pt
pt.init(version = 5.6,helper_version="0.0.6") 
pt.set_property("termpipelines", "")
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# nltk.download('punkt')

wandb_logger = WandbLogger(project="wiki_map")
tb_logger = TensorBoardLogger("logs/")

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
dev_data, dev_qrel = load_dataset("dev")
test_data, test_qrel = load_dataset("test")


class wiki_dataset(Dataset):
    def __init__(self, tokenizer, type_path, num_samples, input_length, print_text=False):

        if type_path == 'train':
            self.dataset = nlp_dataset.from_pandas(train_data)
        elif type_path == "validation":
            self.dataset = nlp_dataset.from_pandas(dev_data)
        elif type_path == "test":
            self.dataset = nlp_dataset.from_pandas(test_data)
        if num_samples:
            self.dataset = self.dataset.select(list(range(0, num_samples)))
        self.input_length = input_length
        self.tokenizer = tokenizer
        self.print_text = print_text

    def __len__(self):
        return self.dataset.shape[0]

    def clean_text(self, text):
        text = text.replace('?', '')
        return text

    def convert_to_features(self, example_batch):
        if self.print_text:
            print("Input Text: ", self.clean_text(example_batch['query']))

        input_ = example_batch['query']
    
        source = self.tokenizer.batch_encode_plus([input_], max_length=self.input_length,
                                                  padding="max_length", truncation=True, return_tensors="pt")
        qid = example_batch['qid']
        docno = example_batch['docno']
        text = example_batch['text']
        label = example_batch['label']

        return input_, qid, docno, text, label, source
    def __getitem__(self, index):
        input_, qid, docno, text, label, source = self.convert_to_features(self.dataset[index])

        source_ids = source["input_ids"].squeeze()

        src_mask = source["attention_mask"].squeeze()


        return {"input":input_, "source_ids": source_ids, "source_mask": src_mask, "qid": qid,
        "docno":docno,"text":text, "label":label}

# 对初始的文档进行BM25检索后看一下结果
textscorer = pt.batchretrieve.TextScorer(takes="docs", body_attr="text", wmodel="BM25")
res_test = textscorer.transform(test_data)
test_result = pt.Utils.evaluate(res_test,test_qrel,metrics = ["map"])

print(test_result)

# 获取arguments

args_dict = dict(
    output_dir="",  # path to save the checkpoints
    model_name_or_path='/data/lxk/DownloadPretrainedModel/t5-small',
    tokenizer_name_or_path='/data/lxk/DownloadPretrainedModel/t5-small',
    max_input_length=15,
    max_output_length=150,
    freeze_encoder=False,
    freeze_embeds=True,
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
    val_check_interval=0.05,
    n_val=10,
    n_train=10,
    n_test=10,
    early_stop_callback=False,
    fp_16=False,  # if you want to enable 16-bit training then install apex and set this to true
    opt_level='O1',  # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
    # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
    max_grad_norm=1.0,
    seed=42,
)

# 获取数据
args_dict.update({'output_dir': 'wiki_reinforce', 'num_train_epochs':20,
                 'train_batch_size': 10, 'eval_batch_size': 10})
args = argparse.Namespace(**args_dict)
print(args_dict)


def get_dataset(tokenizer, type_path, num_samples, args):
    return wiki_dataset(tokenizer=tokenizer, type_path=type_path, num_samples=num_samples,  input_length=args.max_input_length)


################### Define Model ############################


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)

def clean_text(text):
    tokens = list(gensim_tokenize(text))
    text = " ".join(tokens)
    return text

class T5FineTuner(pl.LightningModule):
    def __init__(self, hparams):
        super(T5FineTuner, self).__init__()
        self.hparams_tmp = hparams
        self.model = T5ForConditionalGeneration.from_pretrained(
            hparams.model_name_or_path)
        self.tokenizer = T5Tokenizer.from_pretrained(
            hparams.tokenizer_name_or_path)
        # self.rouge_metric = load_metric('rouge')

        if self.hparams_tmp.freeze_embeds:
            self.freeze_embeds()
        if self.hparams_tmp.freeze_encoder:
            self.freeze_params(self.model.get_encoder())
            # assert_all_frozen(self.model.get_encoder())

        n_observations_per_split = {
            "train": self.hparams_tmp.n_train,
            "validation": self.hparams_tmp.n_val,
            "test": self.hparams_tmp.n_test,
        }
        self.n_obs = {k: v if v >= 0 else None for k,
                      v in n_observations_per_split.items()}

    def freeze_params(self, model):
        for par in model.parameters():
            par.requires_grad = False

    def freeze_embeds(self):
        """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
        try:
            self.freeze_params(self.model.model.shared)
            for d in [self.model.model.encoder, self.model.model.decoder]:
                self.freeze_params(d.embed_positions)
                self.freeze_params(d.embed_tokens)
        except AttributeError:
            self.freeze_params(self.model.shared)
            for d in [self.model.encoder, self.model.decoder]:
                self.freeze_params(d.embed_tokens)

    def lmap(self, f, x):
        """list(map(f, x))"""
        return list(map(f, x))

    def is_logger(self):
        return self.trainer.global_rank <= 0

    def parse_score(self, result):
        return {k: round(v.mid.fmeasure * 100, 4) for k, v in result.items()}

    def generation_decode(self, batch):
        # 训练集合中的生成
        t0 = time.time()

        generated_ids = self.model.generate(
            batch["source_ids"],
            attention_mask=batch["source_mask"],
            use_cache=True,
            max_length=15,
            num_beams=2,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True
        )
        preds = self.ids_to_clean_text(generated_ids)

        gen_time = (time.time() - t0) / batch["source_ids"].shape[0]

        return preds

    def forward(
        self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, lm_labels=None
    ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=lm_labels,
        )

    def _step(self, batch):
        

        qid = batch['qid']
        docno = batch['docno']
        text = batch['text']
        label = batch['label'].detach().cpu().numpy()

        # 根据source_id 生成querys,从而获得每个生成query在检索中的map值
        generated_querys = self.generation_decode(batch)


         # 根据生成的query获得其概率值
        target_tokenize = self.tokenizer.batch_encode_plus(generated_querys, max_length=args.max_input_length,
                                                  padding="max_length", truncation=True, return_tensors="pt")

        outputs = self(
            input_ids = batch['source_ids'],
            attention_mask = batch['source_mask'],
            lm_labels = target_tokenize['input_ids'].type_as(batch['source_ids']),
            decoder_attention_mask = target_tokenize['attention_mask'].type_as(batch['source_ids'])
        )

        logits = outputs['logits']

        # print("len generated_querys",len(generated_querys))
        topic_train = pd.DataFrame({"qid":qid, "query":generated_querys,
        "docno":docno,"text":text,'label':label})

        # 将label等于1的text 直接作为生成的query作为一个强正向的reward

        topic_train.loc[topic_train.label == 1, "query"] = topic_train.loc[topic_train.label == 1,"text"]

        topic_train['query'] = topic_train['query'].apply(clean_text)
        res = textscorer.transform(topic_train)

        eval_result = pt.Utils.evaluate(res,train_qrel,metrics = ['map'], perquery = True)

        map_scores = [eval_result[q]['map'] if q in eval_result else 0.0 for q in qid]

        target_scores = torch.tensor(map_scores)
                
        m = torch.nn.Softmax(dim = 2)
        softmax_score = m(logits)
        # 这个scores是生成的概率
        scores = torch.prod(torch.max(softmax_score,2).values,1)
        target_scores = target_scores.type_as(scores)

        print("scores:{}---- target_scores:{}".format(scores,target_scores))
        # 生成loss,我们用map作为reward来训练我们的生成模型
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(target_scores,scores)
        
        return loss,np.mean(map_scores)

    def ids_to_clean_text(self, generated_ids):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        gen_text_clean = self.lmap(clean_text,gen_text)
        return self.lmap(str.strip, gen_text_clean)

    def _generative_step(self, batch):

        # preds = self.generation_decode(batch)
        # target = self.ids_to_clean_text(batch['target_ids'])
        loss, map_score = self._step(batch)
        base_metrics = {'val_loss': loss}

        print("mapscore dev:{}".format(map_score))
        # self.rouge_metric.add_batch(preds,target)
        # rouge_results = self.rouge_metric.compute()
        # rouge_dict = self.parse_score(rouge_results)

        # self.log("rouge1",rouge_results['rouge1'],on_epoch=True, prog_bar=True, logger=True)
        # self.log("rougeL",rouge_results['rougeL'],on_epoch=True, prog_bar=True, logger=True)

        # base_metrics.update(rouge1=rouge_dict['rouge1'], rougeL=rouge_dict['rougeL'])

        return base_metrics

    def training_step(self, batch, batch_idx):
        loss, map_score = self._step(batch)
        print("map_score train:{}".format(map_score))
        tensorboard_logs = {"train_loss": loss}
        self.log("train_loss", loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_train_loss": avg_train_loss}
        self.log("avg_train_loss", avg_train_loss,
                 on_epoch=True, prog_bar=True, logger=True)
        # self.logger.experiment.add_scalar("avg_train_loss", avg_train_loss, self.current_epoch)

        # return {"avg_train_loss": avg_train_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        return self._generative_step(batch)

    def validation_epoch_end(self, outputs):

        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log("val_loss", avg_loss, on_epoch=True, prog_bar=True, logger=True)
        tensorboard_logs = {"val_loss": avg_loss}

        # summ_len = np.mean(self.lmap(len, generated_ids))
        # base_metrics.update(gen_time=gen_time,
        #                     gen_len=summ_len, preds=preds, target=target)
        # self.rouge_metric.add_batch(preds, target)

        # rouge_results = self.rouge_metric.compute()
        # rouge_dict = self.parse_score(rouge_results)
        # base_metrics.update(rouge1=rouge_dict['rouge1'], rougeL=rouge_dict['rougeL'])

        # rouge_results = self.rouge_metric.compute()
        # rouge_dict = self.parse_score(rouge_results)

        # tensorboard_logs.update(
        #     rouge1=rouge_dict['rouge1'], rougeL=rouge_dict['rougeL'])

        # Clear out the lists for next epoch
        self.target_gen = []
        self.prediction_gen = []
        # self.log("rouge1",rouge_results['rouge1'],on_epoch=True, prog_bar=True, logger=True)
        # self.log("rougeL",rouge_results['rougeL'],on_epoch=True, prog_bar=True, logger=True)

        # return {"avg_val_loss": avg_loss,
        #         "rouge1": rouge_results['rouge1'],
        #         "rougeL": rouge_results['rougeL'],
        #         "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        model = self.model
        # no_decay = ["bias", "LayerNorm.weight"]
        # optimizer_grouped_parameters = [
        #     {
        #         "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        #         "weight_decay": self.hparams_tmp.weight_decay,
        #     },
        #     {
        #         "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        #         "weight_decay": 0.0,
        #     },
        # ]
        # optimizer = torch.optim.AdamW(
        #     optimizer_grouped_parameters, lr=self.hparams_tmp.learning_rate, eps=self.hparams_tmp.adam_epsilon)
        # self.opt = optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

        return optimizer

    # def optimizer_step(self, epoch, batch_idx,
    #                    optimizer, optimizer_idx, second_order_closure=None, on_tpu=False, using_native_amp=False, using_lbfgs=False):

    #     optimizer.step(second_order_closure)
    #     torch.cuda.empty_cache()
    #     # optimizer.zero_grad()
    #     # self.lr_scheduler.step()

    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.3f}".format(
            self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}

        return tqdm_dict

    def train_dataloader(self):
        n_samples = self.n_obs['train']
        train_dataset = get_dataset(
            tokenizer=self.tokenizer, type_path="train", num_samples=n_samples, args=self.hparams_tmp)
        dataloader = DataLoader(train_dataset, batch_size=self.hparams_tmp.train_batch_size,
                                drop_last=True, shuffle=True, num_workers=4)
        # t_total = (
        #     (len(dataloader.dataset) // (self.hparams_tmp.train_batch_size *
        #      max(1, self.hparams_tmp.n_gpu)))
        #     // self.hparams_tmp.gradient_accumulation_steps
        #     * float(self.hparams_tmp.num_train_epochs)
        # )
        # scheduler = get_linear_schedule_with_warmup(
        #     self.opt, num_warmup_steps=self.hparams_tmp.warmup_steps, num_training_steps=t_total
        # )
        # self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        n_samples = self.n_obs['validation']
        validation_dataset = get_dataset(
            tokenizer=self.tokenizer, type_path="validation", num_samples=n_samples, args=self.hparams_tmp)

        return DataLoader(validation_dataset, batch_size=self.hparams_tmp.eval_batch_size, num_workers=4)

    def test_dataloader(self):
        n_samples = self.n_obs['test']
        test_dataset = get_dataset(
            tokenizer=self.tokenizer, type_path="test", num_samples=n_samples, args=self.hparams_tmp)

        return DataLoader(test_dataset, batch_size=self.hparams_tmp.eval_batch_size, num_workers=4)


logger = logging.getLogger(__name__)


class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        logger.info("***** Validation results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
            # Log results
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    logger.info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer, pl_module):
        logger.info("***** Test results *****")

        if pl_module.is_logger():
            metrics = trainer.callback_metrics

            # Log and save results to file
            output_test_results_file = os.path.join(
                pl_module.hparams.output_dir, "test_results.txt")
            with open(output_test_results_file, "w") as writer:
                for key in sorted(metrics):
                    if key not in ["log", "progress_bar"]:
                        logger.info("{} = {}\n".format(key, str(metrics[key])))
                        writer.write("{} = {}\n".format(
                            key, str(metrics[key])))


# Define Checkpoint function
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath=args.output_dir, monitor="val_loss", mode="min", save_top_k=3
)

# If resuming from checkpoint, add an arg resume_from_checkpoint
train_params = dict(
    accumulate_grad_batches=args.gradient_accumulation_steps,
    gpus=args.n_gpu,
    max_epochs=args.num_train_epochs,
    precision=16 if args.fp_16 else 32,
    # amp_level=args.opt_level,
    resume_from_checkpoint=args.resume_from_checkpoint,
    gradient_clip_val=args.max_grad_norm,
    checkpoint_callback=True,
    val_check_interval=args.val_check_interval,
    # logger=wandb_logger,
    # logger = tb_logger,
    callbacks=[LoggingCallback()],
    accelerator = "dp",
)

##################### Train Model ###################

if __name__ == "__main__":

    model = T5FineTuner(args)

    trainer = pl.Trainer(**train_params)
    trainer.fit(model)
    wandb.finish()