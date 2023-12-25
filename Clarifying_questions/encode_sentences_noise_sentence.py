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
import logging

from transformers import BartTokenizer, BartForConditionalGeneration, AdamW, BartConfig
def encode_sentences(tokenizer, source_sentences, target_sentences, max_length=32, pad_to_max_length=True, return_tensors="pt"):
    ''' Function that tokenizes a sentence 
        Args: tokenizer - the BART tokenizer; source and target sentences are the source and target sentences
        Returns: Dictionary with keys: input_ids, attention_mask, target_ids
    '''

    input_ids = []
    attention_masks = []
    target_ids = []
    tokenized_sentences = {}

    for sentence in source_sentences:
        encoded_dict = tokenizer(
            sentence,
            max_length=max_length,
            padding="max_length" if pad_to_max_length else None,
            truncation=True,
            return_tensors=return_tensors,
            add_prefix_space = True
        )
        
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim = 0)
    attention_masks = torch.cat(attention_masks, dim = 0)
    for sentence in target_sentences:
        encoded_dict = tokenizer(
            sentence,
            max_length=max_length,
            padding="max_length" if pad_to_max_length else None,
            truncation=True,
            return_tensors=return_tensors,
            add_prefix_space = True
        )
        # Shift the target ids to the right
        # shifted_target_ids = shift_tokens_right(encoded_dict['input_ids'], tokenizer.pad_token_id)
        target_ids.append(encoded_dict['input_ids'])

    target_ids = torch.cat(target_ids, dim = 0)
    

    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": target_ids,
    }

    return batch


def noise_sentence(sentence_, percent_words, replacement_token = "<mask>"):
    '''
    Function that noises a sentence by adding <mask> tokens
    Args: sentence - the sentence to noise
            percent_words - the percent of words to replace with <mask> tokens; the number is rounded up using math.ceil
    Returns a noised sentence
    '''
    # Create a list item and copy
    sentence_ = sentence_.split(' ')
    sentence = sentence_.copy()
    
    num_words = math.ceil(len(sentence) * percent_words)
    
    # Create an array of tokens to sample from; don't include the last word as an option because in the case of lyrics
    # that word is often a rhyming word and plays an important role in song construction
    sample_tokens = set(np.arange(0, np.maximum(1, len(sentence)-1)))
    
    words_to_noise = random.sample(sample_tokens, num_words)
    
    # Swap out words, but not full stops
    for pos in words_to_noise:
        if sentence[pos] != '.':
            sentence[pos] = replacement_token
    
    # Remove redundant spaces
    sentence = re.sub(r' {2,5}', ' ', ' '.join(sentence))
    
    # Combine concurrent <mask> tokens into a single token; this just does two rounds of this; more could be done
    sentence = re.sub(r'<mask> <mask>', "<mask>", sentence)
    sentence = re.sub(r'<mask> <mask>', "<mask>", sentence)
    return sentence