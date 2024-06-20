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
import logging

from transformers import BartTokenizer, BartForConditionalGeneration, AdamW, BartConfig
from Clarifying_questions.encode_sentences_noise_sentence import encode_sentences, noise_sentence
linebreak = "*"*100

class SummaryDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, data_file, batch_size, num_examples = 20000):
        super().__init__()
        self.tokenizer = tokenizer
        self.data_file = data_file
        self.batch_size = batch_size
        self.num_examples = num_examples
    
    # Loads and splits the data into training, validation and test sets with a 70/15/15 split
    def prepare_data(self):
        # self.data = pd.read_csv(self.data_file)[:self.num_examples]
        self.data = pd.read_csv(self.data_file)
        self.train, self.validate, self.test = np.split(self.data.sample(frac=1), [int(.7*len(self.data)), int(.85*len(self.data))])
        

    # encode the sentences using the tokenizer  
    def setup(self, stage):
        # see the inputs and the detail
        self.train = encode_sentences(self.tokenizer, self.train['source'], self.train['target'])
        self.validate = encode_sentences(self.tokenizer, self.validate['source'], self.validate['target'])
        self.test = encode_sentences(self.tokenizer, self.test['source'], self.test['target'])

    # Load the training, validation and test sets in Pytorch Dataset objects
    def train_dataloader(self):
        dataset = TensorDataset(self.train['input_ids'], self.train['attention_mask'], self.train['labels'])                          
        train_data = DataLoader(dataset, sampler = RandomSampler(dataset), batch_size = self.batch_size)
        return train_data

    def val_dataloader(self):
        dataset = TensorDataset(self.validate['input_ids'], self.validate['attention_mask'], self.validate['labels']) 
        val_data = DataLoader(dataset, batch_size = self.batch_size)                       
        return val_data

    def test_dataloader(self):
        dataset = TensorDataset(self.test['input_ids'], self.test['attention_mask'], self.test['labels']) 
        test_data = DataLoader(dataset, batch_size = self.batch_size)                   
        return test_data