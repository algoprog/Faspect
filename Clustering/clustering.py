import logging
import math
import pickle
import json
import torch
import numpy as np
import random
import collections
from typing import List, Union
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.models import Pooling, Transformer
from sentence_transformers.util import dot_score
from torch.utils.data import DataLoader
from sklearn.cluster import DBSCAN, KMeans
from symbol import return_stmt
import wandb
import os
import numpy as np
os.environ["NCCL_DEBUG"] = "INFO"
from collections import defaultdict
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from facet_ranker import FacetRanker
from utils_ranking_model import PairwiseRankingDataset, RankingDataset, collate_batch, PairwiseRankingDatasetRandom, PointwiseRankingDataset
from utils_ranking_model import convert_example_to_features,collate_batch
from matplotlib import pyplot as plt
from scipy.special import softmax


model_path = 'weights_microsoft/mpnet-base'
line_break = '*'*50



class Clustering:

    def __init__(self, model_path=None):
        self.model = SentenceTransformer(model_path)
        self.model_type = "distilroberta-base"
        self.ranking_model = FacetRanker(model_path="weights_ranker_5",
                                        model_type=self.model_type,
                                        use_gpu=True,
                                        parallel=True,
                                        max_seq_length=512)
        self.tokenizer = self.ranking_model.tokenizer

 
    def generate_the_training_data(self,input_file):
        queryMap = {}
        with open(input_file, encoding="utf-8") as f:
                    for query_id, line in enumerate(f):
                        d = json.loads(line.rstrip("\n"))
                        queryMap[d['query']] = d['snippets']
        return queryMap
    
    def generate_the_training_data_2(self,input_file):
        queryMap = {}
        with open(input_file, encoding="utf-8") as f:
                    for query_id, line in enumerate(f):
                        d = json.loads(line.rstrip("\n"))
                        queryMap[d['query']] = {}
                        queryMap[d['query']]['snippets'] = d['snippets']
                        queryMap[d['query']]['groundtruth'] = d['groundtruth']
                        queryMap[d['query']]['negatives'] = d['negatives']
        return queryMap
    
    
    def extract_facets(self,facet_extractor,query,documents):
        facets = facet_extractor.extract_facets(query, 
                                        documents,
                                        aggregation="round-robin",
                                        mmr_lambda=0.5,
                                        classification_threshold=0.05,
                                        classification_topk=0)
        return facets


    def cluster_facets(self,facets_embeddings):
        clustering = DBSCAN(eps=0.485,min_samples=1).fit(facets_embeddings)
        return clustering.labels_

    def get_average_scores_positives(self,input_file):
        total_score = 0
        total_snippet = 0
        minscore = 300000
        max_score = -300000
        i = 0
        with open(input_file, encoding="utf-8") as f:
                    for query_id, line in enumerate(f):
                        d = json.loads(line.rstrip("\n"))
                        logging.info(line_break)
                        scores_ = self.ranking_cluster(d['query'],d['snippets'],d['groundtruth'])
                        for j, true in enumerate(d['groundtruth']):
                            logging.info(true+"  {}".format(scores_[j]))
                        logging.info(line_break)
                        scores_ = self.ranking_cluster(d['query'],d['snippets'],d['negatives'])
                        for j, true in enumerate(d['negatives']):
                            logging.info(true+"  {}".format(scores_[j]))
                        
                        for each in scores_:
                            total_score+=each
                            total_snippet+=1
                            if(each<minscore):
                                minscore=each
                            if(each>max_score):
                                max_score = each
                        i+=1
                        if(i>10):
                            break
        logging.info("the average score is {}".format(total_score/total_snippet))
        logging.info("the min score is {}".format(minscore))
        logging.info("the max score is {}".format(max_score))


    def batch_generator(self,query,snippets,facet_list):
        batch = []
        for facet in facet_list:
            text = "{} || {} || {}".format(query,facet,snippets)
            features = convert_example_to_features(
                        text,
                        512,
                        self.tokenizer,
                        cls_token_at_end=bool(self.model_type in ["xlnet"]),
                        cls_token=self.tokenizer.cls_token,
                        cls_token_segment_id=2 if self.model_type in ["xlnet"] else 0,
                        sep_token=self.tokenizer.sep_token,
                        sep_token_extra=False,
                        pad_on_left=bool(self.tokenizer.padding_side == "left"),
                        pad_token=self.tokenizer.pad_token_id,
                        pad_token_segment_id=self.tokenizer.pad_token_type_id)
            batch.append(features)
        return collate_batch(batch, all_features=True)

    # def precision(self,facet_extractor,input_file):
    def ranking_cluster_histogram(self,facet_extractor,input_file):
        query_snippets = self.generate_the_training_data_2(input_file)
        positives_scores = np.empty(0)
        negatives_scores = np.empty(0)
        for query in query_snippets:
            #for positives
            snippets = query_snippets[query]['snippets']
            groundtruth_facet_list = query_snippets[query]['groundtruth']
            data = self.batch_generator(query,snippets,groundtruth_facet_list)
            input_ids, attention_mask   = data
            scores_ = self.ranking_model.forward(input_ids, attention_mask).data.cpu().numpy().flatten()
            positives_scores = np.concatenate((positives_scores,1/(1 + np.exp(-scores_))))
            negatives_facet_list = query_snippets[query]['negatives']
            data = self.batch_generator(query,snippets,negatives_facet_list)
            input_ids, attention_mask   = data
            scores_ = self.ranking_model.forward(input_ids, attention_mask).data.cpu().numpy().flatten()

            negatives_scores = np.concatenate((negatives_scores,1/(1 + np.exp(-scores_))))

        bins = np.linspace(-10, 10, 100)
        plt.hist(positives_scores, bins = 100, alpha=0.5, label='groundtruth')
        plt.hist(negatives_scores, bins = 100, alpha=0.5, label='negatives')
        plt.legend(loc='upper right')
        plt.show()
        plt.savefig("graph.png")


    def merge(self,list1, list2,type_facet=True,checker=0.9):
        tuples = []
        for i in range(0,len(list1)):
            if(list2[i]>=0.9):
                tuples.append((list1[i],list2[i],type_facet))

        return  tuples

    def precision(self,facet_extractor,input_file):
        average_precision = 0
        query_snippets = self.generate_the_training_data_2(input_file)
        for query in query_snippets:
            snippets = query_snippets[query]['snippets']
            groundtruth_facet_list = query_snippets[query]['groundtruth']
            data = self.batch_generator(query,snippets,groundtruth_facet_list)
            input_ids, attention_mask   = data
            scores_ = self.ranking_model.forward(input_ids, attention_mask).data.cpu().numpy().flatten()
            scores_ = 1/(1 + np.exp(-scores_))
            positive_tuples = self.merge(groundtruth_facet_list,scores_,True)
            negatives_facet_list = query_snippets[query]['negatives']
            data = self.batch_generator(query,snippets,negatives_facet_list)
            input_ids, attention_mask   = data
            scores_ = self.ranking_model.forward(input_ids, attention_mask).data.cpu().numpy().flatten()
            scores_ = 1/(1 + np.exp(-scores_))
            negative_tuples = self.merge(negatives_facet_list,scores_,False)
            facets_tuple = positive_tuples+negative_tuples
            sorted_tuples = sorted(facets_tuple, key=lambda x: x[1], reverse=True)
            # logging.info(sorted_tuples)
            total=min(5,len(sorted_tuples))
            relevant = 0
            for i in range(min(5,len(sorted_tuples))):
                if(sorted_tuples[i][2]==True):
                    relevant+=1
            # logging.info(relevant/total)
            average_precision+=(relevant/total)
            
    def ranking_cluster_threshold(self,query,snippets,facet_list):
        batch = []
        data = self.batch_generator(query,snippets,facet_list)
        input_ids, attention_mask   = data
        scores_ = self.ranking_model.forward(input_ids, attention_mask).data.cpu().numpy()
        scores_ = 1/(1 + np.exp(-scores_))
        relevant_facets = []
        for i, facet in enumerate(facet_list):
            if(scores_[i]>=0.9):
                relevant_facets.append(facet)
        return relevant_facets


    def ranking_cluster_partition(self,query,snippets,facet_list):
        batch = []
        data = self.batch_generator(query,snippets,facet_list)
        input_ids, attention_mask   = data
        scores_ = self.ranking_model.forward(input_ids, attention_mask).data.cpu().numpy()

        kmeans = KMeans(n_clusters=2)
        kmeans.fit(scores_)
         # this is finding the dominant cluster
        cluster_scores=[[],[]]
        for i, score in enumerate(scores_):
             cluster_scores[kmeans.labels_[i]].append(score)
        relevant_cluster = 0 if (sum(cluster_scores[0])/len(cluster_scores[0])) >  (sum(cluster_scores[1])/len(cluster_scores[1])) else 1

        relevant_facets = []
        for i, facet in enumerate(facet_list):
            if(kmeans.labels_[i]==relevant_cluster):
                relevant_facets.append(facet)
        logging.info(relevant_facets)
        logging.info(facet_list)
        return relevant_facets



    def run_clustering(self,input_file,output_file):
        query_snippets = self.generate_the_training_data(input_file)
        query_clusters = {}
        j =0
        for query in query_snippets:
            facet_list = self.extract_facets(facet_extractor,query,query_snippets[query])
            relevant_facets = self.ranking_cluster_threshold(query,query_snippets[query],facet_list)
            cluster_labels= self.cluster_facets(self.model.encode(relevant_facets))
            clusters = [[] for _ in range(len(set(cluster_labels)))]
            for i,facet in enumerate(relevant_facets):
                clusters[cluster_labels[i]].append(facet)
            query_clusters[query] = clusters
            j+=1
        self.write(query_clusters,output_file)
    
    def cluster_facets_query(query,snippets,facets):
        cluster_labels = self.cluster_facets(self.model.encode(relevant_facets))
        clusters = [[] for _ in range(len(set(cluster_labels)))]
        for i,facet in enumerate(relevant_facets):
            clusters[cluster_labels[i]].append(facet)
        return clusters


    
    
    def write(self, query_clusters,output_file):
        with open(output_file, "w") as outfile:
            for query in  query_clusters:
                dictionary = {
                        "query": query,
                        "facet_clusters":  query_clusters[query],
                }
                json_string = json.dumps(dictionary)
                outfile.write(json_string+"\n")




# logging.basicConfig(filename = "logss2.log", level =logging.INFO, filemode = "w")
# cluster_object = Clustering(model_path)
# cluster_object.run_clustering('dev.jsonl','clusters.txt')
# cluster_object.ranking_cluster_histogram()
# cluster_object.get_average_scores_positives('dev.jsonl')


model = SentenceTransformer(model_path)