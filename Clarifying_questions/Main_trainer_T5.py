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
from transformers import BartTokenizer, BartForConditionalGeneration, AdamW, BartConfig, T5ForConditionalGeneration, AutoTokenizer
from SummaryDataModule import SummaryDataModule
from LitModel_trainer import LitModel
import logging
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
from evaluate import load
from simpletransformers.t5 import T5Model, T5Args



def make_dataset(input_file):
  data_df = pd.read_csv(input_file)
  data_df.rename(columns = {'source':'input_text'}, inplace = True)
  data_df.rename(columns = {'target':'target_text'}, inplace = True)
  data_df['prefix'] = 'generate question'
  train_df, evaluation_df = np.split(data_df.sample(frac=1), [int(.7*len(data_df))])
  return (train_df,evaluation_df)


def generate_prediction(seed_line, model_):
  seedline =  ['generate question: {0}'.format(line) for line in seed_line]
  print(seed_line[:10])
  line = model_.predict(seed_line)



  return line


def run_tests(testfile,model_loaded):
  df = pd.read_csv(testfile)
  df[['predicted', 'Blue', 'Blue_1gram', 'Blue_2gram', 'Blue_3gram']] = ''
  df[['rouge_1_r', 'rouge_1_p', 'rouge_1_f', 'rouge_2_r','rouge_2_p', 'rouge_2_f', 'rouge_l_r', 'rouge_l_p' ,'rouge_l_f']] = ''
  df[['bert_p', 'bert_r', 'bert_f1']] = ''
  question_predictions = generate_prediction(seed_line =  df["source"].values.tolist(), model_ = model_loaded)
  for i in range(len(df)):
    question_pred = question_predictions[i]
  
    question_true = df.iloc[i]['target']
    Blue_score = sentence_bleu([question_pred[0].split()], question_true.split())
    Blue_score_1n = sentence_bleu([question_pred[0].split()], question_true.split(), weights=(1, 0, 0, 0))
    Blue_score_2n = sentence_bleu([question_pred[0].split()], question_true.split(), weights=(0, 1, 0, 0))
    Blue_score_3n = sentence_bleu([question_pred[0].split()], question_true.split(), weights=(0, 0, 1, 0))
     
    rouge = Rouge()
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

  output_file = 't5-base'+'_metrics.csv'
  
  results = bertscore.compute(predictions =df['target'].values.tolist(), references = df['predicted'].values.tolist(), lang="en", verbose=True)
  df['bert_p'] = results['precision']
  df['bert_r'] = results['recall']
  df['bert_f1'] = results['f1']
  df.to_csv(output_file,index=False)

def main(memory_limit):
  logging.basicConfig(filename = "logss.log", level =logging.INFO, filemode = "w")
  train_df,evaluation_df = make_dataset('6-non_generic_train_snippets_cluster.csv')
  model_args = T5Args()
  model_args.max_seq_length = 200
  model_args.train_batch_size = 8
  model_args.eval_batch_size = 8
  model_args.num_train_epochs = 3
  model_args.evaluate_during_training = True
  model_args.evaluate_during_training_steps = 1000
  model_args.use_multiprocessing = True
  model_args.save_eval_checkpoints = False
  model_args.gradient_checkpointing=True
  model_args.optimizer_class="AdamW"
  model_args.overwrite_output_dir = True
  model = T5Model("t5", "t5-base", args=model_args)

  model.train_model(train_df, eval_data=evaluation_df,output_dir='t5_base', show_running_loss=True)
  # model = T5Model("t5", 't5_base/checkpoint-910-epoch-1')
  run_tests("test.csv",model)

if __name__ == "__main__":
  torch.cuda.empty_cache()
  bertscore = load("bertscore")
  main(memory_limit=48000)











