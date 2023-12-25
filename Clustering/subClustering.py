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
from matplotlib import pyplot as plt
from scipy.special import softmax


model_path = 'weights_microsoft/mpnet-base'
line_break = '*'*50



class Clustering:

    def __init__(self, model_path='umass/mpnet-base-mimics-query-facet-encoder'):
        self.model = SentenceTransformer(model_path)
    


    def cluster_facets(self,facets_embeddings):
        clustering = DBSCAN(eps=0.785,min_samples=1).fit(facets_embeddings)
        return clustering.labels_


            
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





    # def run_clustering(self,input_file,output_file):
    #     query_snippets = self.generate_the_training_data(input_file)
    #     query_clusters = {}
    #     j =0
    #     for query in query_snippets:
    #         facet_list = self.extract_facets(facet_extractor,query,query_snippets[query])
    #         relevant_facets = self.ranking_cluster_threshold(query,query_snippets[query],facet_list)
    #         cluster_labels= self.cluster_facets(self.model.encode(relevant_facets))
    #         clusters = [[] for _ in range(len(set(cluster_labels)))]
    #         for i,facet in enumerate(relevant_facets):
    #             clusters[cluster_labels[i]].append(facet)
    #         query_clusters[query] = clusters
    #         j+=1
    #     self.write(query_clusters,output_file)
    
    def cluster_facets_query(self,query,snippets,facets):
        adder = query + " "
        facets = [adder + s  for s in facets]
        cluster_labels = self.cluster_facets(self.model.encode(facets))
        clusters = [[] for _ in range(len(set(cluster_labels)))]
        for i,facet in enumerate(facets):
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




# cluster_object.ranking_cluster_histogram()
# cluster_object.get_average_scores_positives('dev.jsonl')

