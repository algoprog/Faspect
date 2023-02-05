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
from Clarifying_questions.SummaryDataModule import SummaryDataModule
from Clarifying_questions.LitModel_trainer import LitModel
from huggingface_hub import hf_hub_download
import logging
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
from evaluate import load


torch.cuda.empty_cache()

bertscore = load("bertscore")

model_name = "facebook/bart-base"
model_path = "Models/facebook-bart-large.ckpt"

class Clarifying_question:

    def __init__(self, model_path="umass/bart-base-mimics-question-generation"):
        self.tokenizer = BartTokenizer.from_pretrained(model_name, add_prefix_space=True)
        self.bart_model = BartForConditionalGeneration.from_pretrained(model_name)
        self.hparams = argparse.Namespace()
        self.hparams.freeze_encoder = True
        self.hparams.freeze_embeds = True
        self.hparams.eval_beams = 4
        # self.model_loaded = LitModel.load_from_checkpoint("Clarifying_questions/Models/facebook-bart-base.ckpt", learning_rate = 2e-5, tokenizer = self.tokenizer, model = self.bart_model, hparams = self.hparams)
        self.model_path = hf_hub_download(repo_id="umass/bart-base-mimics-question-generation", filename="facebook-bart-base.ckpt")
        self.model_loaded = LitModel.load_from_checkpoint(self.model_path, learning_rate = 2e-5, tokenizer = self.tokenizer, model = self.bart_model, hparams = self.hparams)
 
    def generate_prediction(self,seed_line):
      self.model_loaded.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
      self.model_loaded.eval()
      
      prompt_line_tokens = self.tokenizer(seed_line, max_length = 192, return_tensors = "pt", truncation = True)

      line = self.model_loaded.generate_text(prompt_line_tokens, eval_beams = 8)

      return line





model = Clarifying_question(model_path)
seed_line = "Samsung | Stockmarket, CEO, Devices, Headquarter"
line_pred = model.generate_prediction(seed_line = seed_line)

print(seed_line)
print(line_pred)