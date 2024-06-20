import json
import logging
import random
import re
from collections import defaultdict
from sys import exit

import torch

from typing import List, Optional
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer, AutoTokenizer
from syntok.tokenizer import Tokenizer

#this funciton return the input id and the attention mask
def convert_example_to_features(
        text: str,
        max_seq_length: int,
        tokenizer: PreTrainedTokenizer,
        cls_token_at_end=False,
        cls_token="[CLS]",
        use_segment_ids=False,
        cls_token_segment_id=0,
        sep_token="[SEP]",
        sep_token_extra=True,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=11,
        mask_padding_with_zero=True,
):
#this function return the 
    #it tokenizes the text
    tokens_ = tokenizer.tokenize(text)

    # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
    #getting the number of extra tokens, that is 2
    special_tokens_count = tokenizer.num_special_tokens_to_add()

    # changes the length to max sequence length
    if len(tokens_) > max_seq_length - special_tokens_count:
        tokens_ = tokens_[: (max_seq_length - special_tokens_count)]

    # replacing the || with the separating token and getting the arrray of the segment ids
    tokens = []
    segment_ids = []
    segment_id = 0
    for word in tokens_:
        segment_ids.append(segment_id)
        if word == "Ġ||":
            tokens.append(sep_token)
            segment_id += 1
        else:
            tokens.append(word)
    #adding the extra
    tokens += [sep_token]
    segment_ids.append(segment_id)

    #not doing anything
    if sep_token_extra:
        # roberta uses an extra separator b/w pairs of sentences
        tokens += [sep_token]
        segment_ids.append(segment_id)
  

    # adding the cls token 

    if cls_token_at_end:
        tokens += [cls_token]
        segment_ids += [cls_token_segment_id]
    else:
        tokens = [cls_token] + tokens
        segment_ids = [cls_token_segment_id] + segment_ids

    input_ids = tokenizer.convert_tokens_to_ids(tokens)


    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_seq_length - len(input_ids)

    if pad_on_left:
        input_ids = ([pad_token] * padding_length) + input_ids
        attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
        segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
    else:
        input_ids += [pad_token] * padding_length
        attention_mask += [0 if mask_padding_with_zero else 1] * padding_length
        segment_ids += [pad_token_segment_id] * padding_length
    if(len(input_ids)!=max_seq_length or len(attention_mask)!=max_seq_length):
        logging.info("The length of inputs is wrong from the convert function{} {}".format(len(input_ids),len(attention_mask)))
    assert len(input_ids) == max_seq_length
    assert len(attention_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    if "token_type_ids" not in tokenizer.model_input_names:
        segment_ids = None

    if use_segment_ids:
        return input_ids, attention_mask, segment_ids
    else:
        return input_ids, attention_mask


class PairwiseRankingDataset:
    def __init__(
            self,
            data_path: str,
            tokenizer: PreTrainedTokenizer,
            model_type: str,
            max_seq_length: Optional[int] = None,
            num_negatives=8,
            queries_per_batch=8,
            epochs=3
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.model_type = model_type
        self.num_negatives = num_negatives
        self.queries_per_batch = queries_per_batch
        self.epochs = epochs
        self.epoch = 0
        self.seg_tokenizer = Tokenizer()
        # logging.info("Loading examples from {}...".format(data_path))

        self.qid_index = 0
        self.qids = []
        self.qids_pos_index = []

        self.queries = []
        self.snippets = []
        self.positives = []
        self.negatives = []
        self.total_examples = 0
        with open(data_path, encoding="utf-8") as f:
            for query_id, line in enumerate(f):
                d = json.loads(line.rstrip("\n"))
                self.qids.append(query_id)
                self.queries.append(d["query"])
                self.snippets.append(" || ".join(d["snippets"]))
                self.positives.append(d["groundtruth"])
                self.negatives.append(d["negatives"])
                self.total_examples += len(d["groundtruth"])*min(num_negatives, len(d["negatives"]))

        self.shuffle_dataset()

    def shuffle_dataset(self):
        self.epoch += 1
        self.qid_index = 0
        logging.info("Shuffling dataset...")
        random.shuffle(self.qids)
        self.qids_pos_index = [-1] * len(self.qids)
        for i in range(len(self.qids)):
            random.shuffle(self.positives[i])
            random.shuffle(self.negatives[i])

    
    def generate_negative_facets(self,snippets):
        negatives = []
        if snippets=="":
            return negatives
        snippets_ = snippets.split(" || ")
        for itofNeg in range(5):
            n_length = random.randint(1, 5)  
            j = 0
            while j < 100:
                j += 1
                snippet = random.choice(snippets_)
                tokens = list(self.seg_tokenizer.tokenize(snippet))
                index = random.randint(0, max(0,len(tokens) - n_length))
                negative_tokens = tokens[index:index + n_length]
                negative = " ".join([t.value for t in negative_tokens]).lower()
                if re.match('^[a-z0-9\- ]+$', negative):
                    negatives.append(negative)
                    break
            if j == 100:
                return negatives
        return negatives

    def get_next_ranklist(self):
        self.qids_pos_index[self.qid_index] += 1
        while self.qids_pos_index[self.qid_index] == len(self.positives[self.qids[self.qid_index]]):
            self.qid_index += 1
            if self.qid_index == len(self.qids):
                self.shuffle_dataset()
        # negatives = self.negatives[self.qids[self.qid_index]]

        negatives = self.generate_negative_facets(self.snippets[self.qids[self.qid_index]])


        return self.qids[self.qid_index],self.positives[self.qids[self.qid_index]][self.qids_pos_index[self.qid_index]], random.sample(negatives, min(self.num_negatives, len(negatives)))

    def batch_generator(self):
        while self.epoch <= self.epochs:
            batch = []
            for _ in range(self.queries_per_batch):
                query_id, positive, negatives = self.get_next_ranklist()
                # if self.epoch > self.epochs:
                #     return
                query = self.queries[query_id]
                snippets = self.snippets[query_id]

                positive_text = "{} || {} || {}".format(query, positive, snippets)
                pos_features = convert_example_to_features(
                    positive_text,
                    self.max_seq_length,
                    self.tokenizer,
                    cls_token_at_end=bool(self.model_type in ["xlnet"]),
                    cls_token=self.tokenizer.cls_token,
                    cls_token_segment_id=2 if self.model_type in ["xlnet"] else 0,
                    sep_token=self.tokenizer.sep_token,
                    sep_token_extra=False,
                    pad_on_left=bool(self.tokenizer.padding_side == "left"),
                    pad_token=self.tokenizer.pad_token_id,
                    pad_token_segment_id=self.tokenizer.pad_token_type_id)

                for negative in negatives:
                    negative_text = "{} || {} || {}".format(query, negative, snippets)
                    neg_features = convert_example_to_features(
                        negative_text,
                        self.max_seq_length,
                        self.tokenizer,
                        cls_token_at_end=bool(self.model_type in ["xlnet"]),
                        cls_token=self.tokenizer.cls_token,
                        cls_token_segment_id=2 if self.model_type in ["xlnet"] else 0,
                        sep_token=self.tokenizer.sep_token,
                        sep_token_extra=False,
                        pad_on_left=bool(self.tokenizer.padding_side == "left"),
                        pad_token=self.tokenizer.pad_token_id,
                        pad_token_segment_id=self.tokenizer.pad_token_type_id)
                    features = pos_features + neg_features
                    batch.append(features)
            yield collate_batch(batch)


class PointwiseRankingDataset:
    def __init__(
            self,
            data_path: str,
            tokenizer: PreTrainedTokenizer,
            model_type: str,
            max_seq_length: Optional[int] = None,
            batch_size=8,
            random_split=0.5
    ):
        self.tokenizer = tokenizer
        self.seg_tokenizer = Tokenizer()
        self.max_seq_length = max_seq_length
        self.model_type = model_type
        self.batch_size = batch_size
        self.random_split = random_split

        logging.info("the random split is {}".format(random_split))
        self.qids = []
        self.pos_index = defaultdict(lambda: 0)
        self.neg_index = defaultdict(lambda: 0)

        self.queries = []
        self.snippets = []
        self.positives = []
        self.negatives = []
        self.total_examples = 0
        #just parsing the data of
        with open(data_path, encoding="utf-8") as f:
            for query_id, line in enumerate(f):
                d = json.loads(line.rstrip("\n"))
                self.qids.append(query_id)
                self.queries.append(d["query"])
                self.snippets.append(" || ".join(d["snippets"]))
                self.positives.append(d["groundtruth"])
                self.negatives.append(d["negatives"])
                self.total_examples += len(d["groundtruth"]) + len(d["negatives"])
    

    def batch_generator(self):
        while True:

            batch = []
            for i in range(self.batch_size):
                query_id = random.randint(0, len(self.queries) - 1)
                query = self.queries[query_id]
                snippets = ""
                r = random.random()
                if(r<self.random_split):
                    neg_qid = random.randint(0, len(self.queries) - 1)
                    while len(self.snippets[neg_qid]) < 5:
                        neg_qid = random.randint(0, len(self.queries) - 1)
                    snippets = self.snippets[neg_qid]
                else:
                    snippets = self.snippets[query_id]
                label = 0.0
                not_found = False
                facet = ""
                # if i % 2 == 0:
                #     r = random.random()
                #     if r >= self.random_split:
                #         if len(self.negatives[query_id]) == 0:
                #             not_found = True
                #         else:
                #             facet = self.negatives[query_id][self.neg_index[query_id]]
                #             self.neg_index[query_id] = (self.neg_index[query_id] + 1) % len(self.negatives[query_id])
                #     else:
                #         neg_qid = random.randint(0, len(self.queries) - 1)
                #         while len(self.negatives[neg_qid]) < 2:
                #             neg_qid = random.randint(0, len(self.queries) - 1)
                #         neg_id = random.randint(0, len(self.negatives[neg_qid]) - 1)
                #         facet = self.negatives[neg_qid][neg_id]
                if i % 2 == 0:
                    if snippets=="":
                        not_found = True
                    else:
                        n_length = random.randint(1, 5)
                        snippets_ = snippets.split(" || ")
                        j = 0
                        while j < 100:
                            j += 1
                            snippet = random.choice(snippets_)
                            tokens = list(self.seg_tokenizer.tokenize(snippet))
                            index = random.randint(0, max(0,len(tokens) - n_length))
                            negative_tokens = tokens[index:index + n_length]
                            negative = " ".join([t.value for t in negative_tokens]).lower()
                            if re.match('^[a-z0-9\- ]+$', negative):
                                facet = negative
                                break
                        if j == 100:
                            not_found = True
                if not_found or i % 2 == 1:
                    label = 1.0
                    facet = self.positives[query_id][self.pos_index[query_id]]
                    self.pos_index[query_id] = (self.pos_index[query_id] + 1) % len(self.positives[query_id])

                text = "{} || {} || {}".format(query, facet, snippets)
    
                features = convert_example_to_features(
                    text,
                    self.max_seq_length,
                    self.tokenizer,
                    cls_token_at_end=bool(self.model_type in ["xlnet"]),
                    cls_token=self.tokenizer.cls_token,
                    cls_token_segment_id=2 if self.model_type in ["xlnet"] else 0,
                    sep_token=self.tokenizer.sep_token,
                    sep_token_extra=False,
                    pad_on_left=bool(self.tokenizer.padding_side == "left"),
                    pad_token=self.tokenizer.pad_token_id,
                    pad_token_segment_id=self.tokenizer.pad_token_type_id)
                features = features + (label,)
                batch.append(features)
            #logging.info("the Batch length from the collate batch pointwise {}".format(len(batch)))
            yield collate_batch(batch, all_features=True)


class PairwiseRankingDatasetRandom(Dataset):
    def __init__(
            self,
            data_path: str,
            tokenizer: PreTrainedTokenizer,
            model_type: str,
            max_seq_length: Optional[int] = None,
            random_split=0.5,
            use_snippets=True
    ):
        self.tokenizer = tokenizer
        self.seg_tokenizer = Tokenizer()
        self.max_seq_length = max_seq_length
        self.model_type = model_type
        self.random_split = random_split
        self.use_snippets = use_snippets

        self.queries = []
        self.snippets = []
        self.positives = []
        self.negatives = []
        self.examples = []
        with open(data_path, encoding="utf-8") as f:
            for query_id, line in enumerate(f):
                d = json.loads(line.rstrip("\n"))
                self.queries.append(d["query"])
                self.snippets.append(" || ".join(d["snippets"]))
                self.positives.append(d["groundtruth"])
                self.negatives.append(d["negatives"])
                for i in range(len(self.positives[-1])):
                    for j in range(len(self.negatives[-1])):
                        self.examples.append((query_id, i, j))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        example = self.examples[i]
        query_id = example[0]
        pos_id = example[1]
        neg_id = example[2]

        query = self.queries[query_id]
        snippets = self.snippets[query_id]
        positive = self.positives[query_id][pos_id]

        r = random.random()
        if r < self.random_split:
            snippets_ = snippets.split(" || ")
            snippet = random.choice(snippets_)
            tokens = list(self.seg_tokenizer.tokenize(snippet))
            j = 0
            while j < 100:
                j += 1
                length = random.randint(1, 5)
                index = random.randint(0, max(0,len(tokens) - length))
                negative_tokens = tokens[index:index + length]
                negative = " ".join([t.value for t in negative_tokens]).lower()
                if re.match('^[a-z0-9\- ]+$', negative):
                    break
            if j == 100:
                negative = self.negatives[query_id][neg_id]
        else:
            negative = self.negatives[query_id][neg_id]

        if self.use_snippets:
            positive_text = "{} || {} || {}".format(query, positive, snippets)
        else:
            positive_text = "{} || {}".format(query, positive)

        pos_features = convert_example_to_features(
            positive_text,
            self.max_seq_length,
            self.tokenizer,
            cls_token_at_end=bool(self.model_type in ["xlnet"]),
            cls_token=self.tokenizer.cls_token,
            cls_token_segment_id=2 if self.model_type in ["xlnet"] else 0,
            sep_token=self.tokenizer.sep_token,
            sep_token_extra=False,
            pad_on_left=bool(self.tokenizer.padding_side == "left"),
            pad_token=self.tokenizer.pad_token_id,
            pad_token_segment_id=self.tokenizer.pad_token_type_id)

        if self.use_snippets:
            negative_text = "{} || {} || {}".format(query, negative, snippets)
        else:
            negative_text = "{} || {}".format(query, negative)

        neg_features = convert_example_to_features(
            negative_text,
            self.max_seq_length,
            self.tokenizer,
            cls_token_at_end=bool(self.model_type in ["xlnet"]),
            cls_token=self.tokenizer.cls_token,
            cls_token_segment_id=2 if self.model_type in ["xlnet"] else 0,
            sep_token=self.tokenizer.sep_token,
            sep_token_extra=False,
            pad_on_left=bool(self.tokenizer.padding_side == "left"),
            pad_token=self.tokenizer.pad_token_id,
            pad_token_segment_id=self.tokenizer.pad_token_type_id)

        features = pos_features + neg_features
        return features


class RankingDataset(Dataset):
    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            model_type: str,
            data_path: Optional[str] = None,
            data: Optional[List] = None,
            max_seq_length: Optional[int] = None,
            use_snippets=True
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.model_type = model_type
        self.use_snippets = use_snippets
        self.seg_tokenizer = Tokenizer()

        if data_path is not None:
            self.from_file = True
            self.queries = []
            self.facets = []
            self.snippets = []
            self.examples = []
            with open(data_path, encoding="utf-8") as f:
                for query_id, line in enumerate(f):
                    d = json.loads(line.rstrip("\n"))
                    self.queries.append(d["query"])
                    self.snippets.append(" || ".join(d["snippets"]))
                    #negatives = self.generate_negative_facets(self.snippets[-1])
                    query_facets = d["groundtruth"] + d['negatives']
                    pos_labels = len(d["groundtruth"])
                    self.facets.append(query_facets)
                    for facet_id, facet in enumerate(query_facets):
                        label = 1
                        if facet_id >= pos_labels:
                            label = 0
                        self.examples.append((query_id,facet_id,len(self.snippets) - 1,label))

        else:
            self.from_file = False
            self.examples = data
    
    def generate_negative_facets(self,snippets):
        negatives = []
        if snippets=="":
            return negatives
        snippets_ = snippets.split(" || ")
        for itofNeg in range(10):
            n_length = random.randint(1, 5)  
            j = 0
            while j < 100:
                j += 1
                snippet = random.choice(snippets_)
                tokens = list(self.seg_tokenizer.tokenize(snippet))
                index = random.randint(0, max(0,len(tokens) - n_length))
                negative_tokens = tokens[index:index + n_length]
                negative = " ".join([t.value for t in negative_tokens]).lower()
                if re.match('^[a-z0-9\- ]+$', negative):
                    negatives.append(negative)
                    break
            if j == 100:
                return negatives
        return negatives


    def __len__(self):
        return len(self.examples)
    
    # this will return the ith item of the development
    def __getitem__(self, i):
        example = self.examples[i]

        if not self.from_file:
            if self.use_snippets:
                text = "{} || {} || {}".format(example[0], example[1], example[2])
            else:
                text = "{} || {}".format(example[0], example[1])
        else:
            if self.use_snippets:
                text = "{} || {} || {}".format(self.queries[example[0]],
                                               self.facets[example[0]][example[1]],
                                               self.snippets[example[2]])

            else:
                text = "{} || {}".format(self.queries[example[0]],
                                         self.facets[example[1]])
    
        features = convert_example_to_features(
            text,
            self.max_seq_length,
            self.tokenizer,
            cls_token_at_end=bool(self.model_type in ["xlnet"]),
            cls_token=self.tokenizer.cls_token,
            cls_token_segment_id=2 if self.model_type in ["xlnet"] else 0,
            sep_token=self.tokenizer.sep_token,
            sep_token_extra=False,
            pad_on_left=bool(self.tokenizer.padding_side == "left"),
            pad_token=self.tokenizer.pad_token_id,
            pad_token_segment_id=self.tokenizer.pad_token_type_id)
        (input_ids, attention_mask) = features
        
        if self.from_file:
            return features + (example[0], example[3],)  # input ids, mask, qid, label
            
        else:
            return features  # input ids, mask


def collate_batch(batch, all_features=False):
    #this funtion get called from the iterators of the function, and returns tensors
    num_features = len(batch[0])
  
    coll_batch = [[] for _ in range(num_features)]

    for sample in batch:
        for i, x in enumerate(sample):
            coll_batch[i].append(x)

    for i in range(num_features):
        if all_features or isinstance(coll_batch[i][0], list):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            t = torch.tensor(coll_batch[i]).to(torch.device(device))
            coll_batch[i] = t


    return coll_batch

def run():
    logging.getLogger().setLevel(logging.INFO)
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    dataset = PointwiseRankingDataset(data_path="test.jsonl",
                                      max_seq_length=512,
                                      model_type="distilroberta-base",
                                      tokenizer=tokenizer,
                                      batch_size=4)
    dataloader = dataset.batch_generator()
    """
    dataloader = DataLoader(dataset=dataset,
                            batch_size=8,
                            shuffle=True,
                            collate_fn=collate_batch)
    """



    
 # if i % 2 == 0:
                #     r = random.random()
                #     if r >= self.random_split:
                #         if len(self.negatives[query_id]) == 0:
                #             not_found = True
                #         else:
                #             facet = self.negatives[query_id][self.neg_index[query_id]]
                #             self.neg_index[query_id] = (self.neg_index[query_id] + 1) % len(self.negatives[query_id])
                #     else:
                       
                #         r = random.random()
                #         if r < 0.5:
                #             neg_qid = random.randint(0, len(self.queries) - 1)
                #             while len(self.negatives[neg_qid]) < 2:
                #                 neg_qid = random.randint(0, len(self.queries) - 1)
                #             neg_id = random.randint(0, len(self.negatives[neg_qid]) - 1)
                #             facet = self.negatives[neg_qid][neg_id]
                #         else:
                #             snippets_ = snippets.split(" || ")
                #             j = 0
                #             while j < 100:
                #                 j += 1
                #                 snippet = random.choice(snippets_)
                #                 tokens = list(self.seg_tokenizer.tokenize(snippet))
                #                 length = random.randint(1, 5)
                #                 index = random.randint(0, len(tokens) - 1)
                #                 negative_tokens = tokens[index:index + length]
                #                 negative = " ".join([t.value for t in negative_tokens]).lower()
                #                 if re.match('^[a-z0-9\- ]+$', negative):
                #                     facet = negative
                #                     break
                #             if j == 100:
                #                 if len(self.negatives[query_id]) == 0:
                #                     not_found = True
                #                 else:
                #                     facet = self.negatives[query_id][self.neg_index[query_id]]
                #                     self.neg_index[query_id] = (self.neg_index[query_id] + 1) % len(self.negatives[query_id])