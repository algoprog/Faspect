import transformers
from torch.utils.data import DataLoader, TensorDataset, random_split, RandomSampler, Dataset
import pandas as pd
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
import math
import random
import re
import argparse
import wandb
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
from bert_score import score
from pytorch_lightning.callbacks import TQDMProgressBar
from transformers import BartTokenizer, BartForConditionalGeneration, AdamW, BartConfig, AutoTokenizer
from SummaryDataModule import SummaryDataModule
from LitModel_trainer import LitModel
import logging
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
from evaluate import load
torch.cuda.empty_cache()

bertscore = load("bertscore")

# model_name = "facebook/bart-large"
# model_path = "Models/facebook-bart-large.ckpt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def generate_prediction(seed_line, model_):
  # Put the model on eval mode
  model_.to(device)
  model_.eval()
  
  prompt_line_tokens = tokenizer(seed_line, max_length = 192, return_tensors = "pt", padding=True,truncation = True)

  line = model_.generate_text(prompt_line_tokens, eval_beams = 8)


  return line


def run_tests(testfile,model_loaded):
  df = pd.read_csv(testfile)
  df[['predicted', 'Blue', 'Blue_1gram', 'Blue_2gram', 'Blue_3gram']] = ''
  df[['rouge_1_r', 'rouge_1_p', 'rouge_1_f', 'rouge_2_r','rouge_2_p', 'rouge_2_f', 'rouge_l_r', 'rouge_l_p' ,'rouge_l_f']] = ''
  df[['bert_p', 'bert_r', 'bert_f1']] = ''
  for i in range(len(df)):
    line = df.iloc[i]['source']
    question_pred = generate_prediction(seed_line = line, model_ = model_loaded)
    question_true = df.iloc[i]['target']
    Blue_score = sentence_bleu([question_pred[0].split()], question_true.split())
    Blue_score_1n = sentence_bleu([question_pred[0].split()], question_true.split(), weights=(1, 0, 0, 0))
    Blue_score_2n = sentence_bleu([question_pred[0].split()], question_true.split(), weights=(0, 1, 0, 0))
    Blue_score_3n = sentence_bleu([question_pred[0].split()], question_true.split(), weights=(0, 0, 1, 0))
     
    rouge = Rouge()
    # logging.info(question_pred[0])
    # logging.info(question_true)
    # print(question_pred[0])
    # print(question_true)
    rouge_res = rouge.get_scores(question_pred[0], question_true)

    df.loc[i]['predicted'] = question_pred
    df.loc[i]['Blue'] = Blue_score
    df.loc[i]['Blue_1gram'] = Blue_score_1n
    df.loc[i]['Blue_2gram'] = Blue_score_2n
    df.loc[i]['Blue_3gram'] = Blue_score_3n
    df.loc[i]['rouge_1_r'] = rouge_res[0]["rouge-1"]['r']
    df.loc[i]['rouge_1_p'] = rouge_res[0]["rouge-1"]['p']
    df.loc[i]['rouge_1_f'] = rouge_res[0]["rouge-1"]['f']
    df.loc[i]['rouge_2_r'] = rouge_res[0]["rouge-2"]['r']
    df.loc[i]['rouge_2_p'] = rouge_res[0]["rouge-2"]['p']
    df.loc[i]['rouge_2_f'] = rouge_res[0]["rouge-2"]['f']
    df.loc[i]['rouge_l_r'] = rouge_res[0]["rouge-l"]['r']
    df.loc[i]['rouge_l_p'] = rouge_res[0]["rouge-l"]['p']
    df.loc[i]['rouge_l_f'] = rouge_res[0]["rouge-l"]['f']
    print("iteration")

  output_file = "facebook_bart-large"+'_metrics.csv'
  
  results = bertscore.compute(predictions =df['target'].values.tolist(), references = df['predicted'].values.tolist(), lang="en", verbose=True)
  df['bert_p'] = results['precision']
  df['bert_r'] = results['recall']
  df['bert_f1'] = results['f1']
  df.to_csv(output_file,index=False)


logging.basicConfig(filename = "logss.log", level =logging.INFO, filemode = "w")
base_dir = ''
wandb.init()
wandb.config = {
"learning_rate": 0.00002,
"epochs": 10,
"batch_size": 64
}
logging.info("The run name on wandb is {}".format(wandb.run.name))

hparams = argparse.Namespace()
hparams.freeze_encoder = True
hparams.freeze_embeds = True
hparams.eval_beams = 4


tokenizer = BartTokenizer.from_pretrained(model_name, add_prefix_space=True)
# tokenizer = BartTokenizer.from_pretrained(model_path)



bart_model = BartForConditionalGeneration.from_pretrained(model_name)


summary_data = SummaryDataModule(tokenizer, base_dir + '6-non_generic_train_snippets_cluster.csv',
                                 batch_size = 4)

model = LitModel(learning_rate = 2e-5, tokenizer = tokenizer, model = bart_model, hparams = hparams)


checkpoint = ModelCheckpoint(dirpath=base_dir)

trainer = pl.Trainer(gpus = 1,
                     max_epochs = 1,
                     min_epochs = 1,
                     auto_lr_find = True,
                     callbacks=[checkpoint,TQDMProgressBar(refresh_rate=100)])

trainer.fit(model, summary_data)

trainer.save_checkpoint(base_dir + "./Models/facebook-bart-large_2.ckpt")




#run_tests("test.csv",model)
# run_tests("7-openAI-clustered.csv",model)









line_pred = generate_prediction(seed_line = ["Samsung | Television , Smartphone, Soundbox , Computer , Vaccum ",
                                            "Samsung | Stockmarket, CEO, Devices, Headquarter",
                                            "mercedes cla class convertible | exterior , interior , engine , prices , competition",
                                            "Selena Gomez | Age , Birthday , Albums , Livingplace",
                                            "Weather | wind , temperature, precipitation, humidity , visibility | Weather is controlled by many factors, "], 
                                            model_ = model)

print(line_pred)







