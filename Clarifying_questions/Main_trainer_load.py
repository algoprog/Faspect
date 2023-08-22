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
from transformers import BartTokenizer, BartForConditionalGeneration, AdamW, BartConfig, AutoTokenizer,AutoConfig
from SummaryDataModule import SummaryDataModule
from LitModel_trainer import LitModel
import logging
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
from evaluate import load


torch.cuda.empty_cache()

bertscore = load("bertscore")

model_name = "facebook/bart-large"
model_path = "Models/facebook-bart-large.ckpt"
def generate_prediction(seed_line, model_):
  # Put the model on eval mode
  model_.to("cuda")
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



class Bart_model:
  def __init__(self, model_path=None):
    self.checkpoint = torch.load(model_path)
    self.model = BartForConditionalGeneration(AutoConfig.from_pretrained("facebook/bart-large"))
    self.model.load_state_dict(self.checkpoint["state_dict"])
    self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-large", add_prefix_space=True)
    self.device = ""
    if torch.cuda.is_available():
      self.device = torch.device("cuda")
    else:
      self.device = torch.device("cpu")
    self.model(device)
    self.model.eval()
  
  def generate_prediction(self,seed_line):
  # Put the model on eval mode
  # facets = ",".join(string_list)
    prompt_line_tokens = self.tokenizer(seed_line, max_length = 192, return_tensors = "pt", padding=True,truncation = True)

    generated_ids = self.model.generate(
              input_ids=  prompt_line_tokens["input_ids"].to('cuda'),
              num_beams= 8,
              max_length = 192
          )
    line =  [self.tokenizer.decode(w, skip_special_tokens=True, clean_up_tokenization_spaces=True) for w in generated_ids]
        


    return line


# # prepare the input data
# input_ids = ...
# attention_mask = ...

# # make predictions
# with torch.no_grad():
#     output = model(input_ids=input_ids, attention_mask=attention_mask)

# # process the output

model = Bart_model('Models/facebook-bart-large.ckpt')
model.generate_prediction(seed_line = ["Samsung | Television , Smartphone, Soundbox , Computer , Vaccum ",
                                            "Samsung | Stockmarket, CEO, Devices, Headquarter",
                                            "mercedes cla class convertible | exterior , interior , engine , prices , competition",
                                            "Selena Gomez | Age , Birthday , Albums , Livingplace",
                                            "Weather | wind , temperature, precipitation, humidity , visibility | Weather is controlled by many factors, "], 
                                            model_ = model)
predictions = output[0]



