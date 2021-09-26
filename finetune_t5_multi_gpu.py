import wandb
import pytorch_lightning as pl
import os
from transformers import get_linear_schedule_with_warmup
from nlp import list_datasets
from nlp import load_dataset
from nlp import Dataset as nlp_dataset
from IR_transformers.modeling_t5 import T5ForConditionalGeneration
from IR_transformers.tokenization_t5 import T5Tokenizer
from nlp import load_metric


from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from torch.utils.data import Dataset, DataLoader

from pytorch_lightning.plugins import DDPPlugin
import torch.nn as nn
import torch
import numpy as np
import pandas as pd
# from nltk.tokenize import sent_tokenize
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
# import nltk
my_num_workers = os.cpu_count()

FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level = logging.INFO, format=FORMAT)

logger = logging.getLogger(__name__)

wandb_logger = WandbLogger(project="msmarco")
tb_logger = TensorBoardLogger("/data/ceph/zhansu/logs/",name = "msmarco")

df = pd.read_csv("/data/ceph/zhansu/data/msmarco/doc_query_pairs.train.tsv",
                 sep='\t', names=["doc", "query"])

df_train, df_validate, df_test = \
              np.split(df.sample(frac=1, random_state=42), 
                       [int(.6*len(df)), int(.8*len(df))])


class msmarco(Dataset):
    def __init__(self, tokenizer, type_path, num_samples, input_length, output_length, print_text=False):
        # self.dataset = load_dataset(
        #     'wikihow', 'all', data_dir='../summary_dataset/', split=type_path)

        if type_path == 'train':
            self.dataset = nlp_dataset.from_pandas(df_train)
        elif type_path == "validation":
            self.dataset = nlp_dataset.from_pandas(df_validate)
        elif type_path == "test":
            self.dataset = nlp_dataset.from_pandas(df_test)
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
        text = text.replace('\n', '')
        text = text.replace('``', '')
        text = text.replace('"', '')
        text = text.replace("?", "")
        
        return text

    def convert_to_features(self, example_batch):
        if self.print_text:
            print("Input Text: ", self.clean_text(example_batch['doc']))

        input_ = self.clean_text(example_batch['query'])
        target_ = self.clean_text(example_batch['doc'])

        source = self.tokenizer.batch_encode_plus([input_], max_length=self.input_length,
                                                  padding="max_length", truncation=True, return_tensors="pt")

        targets = self.tokenizer.batch_encode_plus([target_], max_length=self.output_length,
                                                   padding="max_length", truncation=True, return_tensors="pt")

        return source, targets

    def __getitem__(self, index):
        source, targets = self.convert_to_features(self.dataset[index])

        source_ids = source["input_ids"].squeeze()
        target_ids = targets["input_ids"].squeeze()

        src_mask = source["attention_mask"].squeeze()
        target_mask = targets["attention_mask"].squeeze()

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}


# Load dataset using NLP

# 获取arguments

args_dict = dict(
    output_dir="./",  # path to save the checkpoints
    model_name_or_path='/data/ceph/zhansu/embedding/t5-small',
    tokenizer_name_or_path='/data/ceph/zhansu/embedding/t5-small',
    max_input_length=150,
    max_output_length=150,
    freeze_encoder=False,
    freeze_embeds=False,
    learning_rate=3e-4,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    warmup_steps=0,
    train_batch_size=64,
    eval_batch_size=64,
    num_train_epochs=3,
    gradient_accumulation_steps=8,
    n_gpu=1,
    resume_from_checkpoint=None,
    val_check_interval=0.5,
    n_val=100,
    n_train=100,
    n_test=100,
    early_stop_callback=False,
    fp_16=False,  # if you want to enable 16-bit training then install apex and set this to true
    opt_level='O1',  # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
    # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
    max_grad_norm=1.0,
    seed=42,
)

# tokenizer = T5Tokenizer.from_pretrained('t5-small')

# 获取数据
args_dict.update({'output_dir': 't5_msmarco', 'num_train_epochs': 5,
                 'train_batch_size': 32, 'eval_batch_size':32})
args = argparse.Namespace(**args_dict)
# print(args_dict)


def get_dataset(tokenizer, type_path, num_samples, args):
    return msmarco(tokenizer=tokenizer, type_path=type_path, num_samples=num_samples,  input_length=args.max_input_length,
                   output_length=args.max_output_length)


################### Define Model ############################


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


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
                freeze_params(d.embed_positions)
                freeze_params(d.embed_tokens)
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
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            lm_labels=lm_labels,
            decoder_attention_mask=batch['target_mask']
        )

        loss = outputs[0]

        return loss

    def ids_to_clean_text(self, generated_ids):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return self.lmap(str.strip, gen_text)

    def _generative_step(self, batch):

        # t0 = time.time()

        # generated_ids = self.model.generate(
        #     batch["source_ids"],
        #     attention_mask=batch["source_mask"],
        #     use_cache=True,
        #     decoder_attention_mask=batch['target_mask'],
        #     max_length=150,
        #     num_beams=2,
        #     repetition_penalty=2.5,
        #     length_penalty=1.0,
        #     early_stopping=True
        # )
        # preds = self.ids_to_clean_text(generated_ids)
        # target = self.ids_to_clean_text(batch["target_ids"])

        # gen_time = (time.time() - t0) / batch["source_ids"].shape[0]

        loss = self._step(batch)
        base_metrics = {'val_loss': loss}
        # self.log("val_loss", loss,
        #          on_step=True, prog_bar=True, logger=True)
#         rouge: Dict = self.calc_generative_metrics(preds, target)
        # summ_len = np.mean(self.lmap(len, generated_ids))
        # base_metrics.update(gen_time=gen_time,
        #                     gen_len=summ_len, preds=preds, target=target)
        # self.rouge_metric.add_batch(preds, target)

#         rouge_results = self.rouge_metric.compute()
#         rouge_dict = self.parse_score(rouge_results)
#         base_metrics.update(rouge1=rouge_dict['rouge1'], rougeL=rouge_dict['rougeL'])

        return base_metrics

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        # logging.info("train step batch_idx:{}".format(batch_idx))
        
        return {"loss": loss}
    # if our trainning using a accelerator the splits data from each batch GPU, we need to implement the training_step_end method

    def training_step_end(self, batch_parts):
        # predictions from each GPU
        # losses from each GPU
        # logging.info("the batch loss from gpu:{}".format(batch_parts['loss']))
        losses = batch_parts["loss"]
        mean_loss = torch.mean(losses)

        logging.info("train loss step each gpu:{}".format(losses))
        print("train loss step mean {}".format(mean_loss.grad))
        self.log("train_loss", mean_loss, on_step=True, on_epoch = True, prog_bar=True, logger=True)
        # do something with both outputs
        return {"loss":mean_loss}

    def training_epoch_end(self, training_step_outputs):
        logging.info("train loss each epoch{}".format(training_step_outputs))
        # print(training_step_outputs)
        avg_train_loss = torch.stack([x["loss"] for x in training_step_outputs]).mean()
        # tensorboard_logs = {"avg_train_loss": avg_train_loss}
        self.log("avg_train_loss", avg_train_loss,
                 on_epoch=True, prog_bar=True, logger=True)
        # self.logger.experiment.add_scalar("avg_train_loss", avg_train_loss, self.current_epoch)

        # return {"avg_train_loss": avg_train_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        return self._generative_step(batch)
    def validation_step_end(self, batch_parts):
        losses = batch_parts["val_loss"]
        mean_loss = torch.mean(losses)
        self.log("val_loss", mean_loss, on_step=True, on_epoch = True, prog_bar=True, logger=True)

        return {"val_loss":mean_loss}

    def test_step(self,batch,batch_idx):
        loss = self._step(batch)
        return {"loss": loss}
    
    def test_step_end(self, batch_parts):
        # predictions from each GPU
        # losses from each GPU
        # logging.info("the batch loss from gpu:{}".format(batch_parts['loss']))
        losses = batch_parts["loss"]
        # do something with both outputs
        return {"loss":torch.mean(losses)}

    def validation_epoch_end(self, outputs):

        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}

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
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams_tmp.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=self.hparams_tmp.learning_rate, eps=self.hparams_tmp.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self, epoch, batch_idx,
                       optimizer, optimizer_idx, second_order_closure=None, on_tpu=False, using_native_amp=False, using_lbfgs=False):
        # if self.trainer.use_tpu:
        #     xm.optimizer_step(optimizer)
        # else:
        optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.3f}".format(
            self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}

        return tqdm_dict

    def train_dataloader(self):
        n_samples = self.n_obs['train']
        train_dataset = get_dataset(
            tokenizer=self.tokenizer, type_path="train", num_samples=n_samples, args=self.hparams_tmp)
        dataloader = DataLoader(train_dataset, batch_size=self.hparams_tmp.train_batch_size,
                                drop_last=True, shuffle=True, num_workers=my_num_workers)
        t_total = (
            (len(dataloader.dataset) // (self.hparams_tmp.train_batch_size *
             max(1, self.hparams_tmp.n_gpu)))
            // self.hparams_tmp.gradient_accumulation_steps
            * float(self.hparams_tmp.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams_tmp.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        n_samples = self.n_obs['validation']
        validation_dataset = get_dataset(
            tokenizer=self.tokenizer, type_path="validation", num_samples=n_samples, args=self.hparams_tmp)

        return DataLoader(validation_dataset, batch_size=self.hparams_tmp.eval_batch_size, num_workers=my_num_workers)

    def test_dataloader(self):
        n_samples = self.n_obs['test']
        test_dataset = get_dataset(
            tokenizer=self.tokenizer, type_path="test", num_samples=n_samples, args=self.hparams_tmp)

        return DataLoader(test_dataset, batch_size=self.hparams_tmp.eval_batch_size, num_workers=my_num_workers)

    
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
                pl_module.hparams_tmp.output_dir, "test_results.txt")
            with open(output_test_results_file, "w") as writer:
                for key in sorted(metrics):
                    if key not in ["log", "progress_bar"]:
                        logger.info("{} = {}\n".format(key, str(metrics[key])))
                        writer.write("{} = {}\n".format(
                            key, str(metrics[key])))


# Define Checkpoint function
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath = args.output_dir + "_model", monitor="val_loss", mode="min", save_top_k=3
)

# If resuming from checkpoint, add an arg resume_from_checkpoint
train_params = dict(
    accumulate_grad_batches=args.gradient_accumulation_steps,
    gpus=args.n_gpu,
    max_epochs=args.num_train_epochs,
    precision=16 if args.fp_16 else 32,
    amp_level=args.opt_level,
    resume_from_checkpoint=args.resume_from_checkpoint,
    gradient_clip_val=args.max_grad_norm,
    # checkpoint_callback=True,
    val_check_interval=args.val_check_interval,
    # logger=wandb_logger,
    logger = tb_logger,
    callbacks=[LoggingCallback(),checkpoint_callback],
    accelerator = "dp",
    # overfit_batches=3
    # fast_dev_run=100
)

##################### Train Model ###################

if __name__ == "__main__":

    model = T5FineTuner(args)

    trainer = pl.Trainer(**train_params)
    trainer.fit(model)
    trainer.test(model,ckpt_path="best")

    print(trainer.checkpoint_callback.best_k_models.items())
    # for i, (path, _) in enumerate(trainer.checkpoint_callback.best_k_models.items()):
    #     m = T5FineTuner.load_from_checkpoint(path,hparams = args)
        # m.model.save_pretrained("./finetune_t5")
    # wandb.finish()

