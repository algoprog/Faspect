import logging
import math
import random
from symbol import return_stmt
import wandb
import torch
import os
import numpy as np
os.environ["NCCL_DEBUG"] = "INFO"

from collections import defaultdict
from torch.cuda.amp import GradScaler, autocast

from torch.utils.data import DatLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

from Prevfacet_ranker import FacetRanker
from utils_ranking_model import PairwiseRankingDataset, RankingDataset, collate_batch, PairwiseRankingDatasetRandom, PointwiseRankingDataset


def evaluate(model, dataloader):

    model.eval()
    scores = defaultdict(lambda: [])
    for input_ids, attention_mask, qids, labels in tqdm(dataloader, position=1):
        scores_ = model.forward(input_ids, attention_mask).data.cpu().numpy()
        for i, score in enumerate(scores_):
            r = random.random()
            scores[qids[i]].append((score, labels[i], r))
        
        # for i in range(len(qids)):
        #     r = random.random()
        #     scores[qids[i]].append((labels[i], r))

    avg_ndcg = 0
    avg_len = 0
    mrr = 0
    for _, scores_ in scores.items():
        sorted_scores = sorted(scores_, key=lambda x: x[0], reverse=True)
        relevant = 0
        dcg = 0
        mr = 0
        avg_len += len(sorted_scores)
        for i, (score,label,rscore) in enumerate(sorted_scores):
            if label == 1:
                relevant += 1
                dcg += 1 / math.log(2 + i)
                if mr == 0:
                    mr = 1 / (i+1)
        idcg = 0
        for i in range(relevant):
            idcg += 1 / math.log(2 + i)
        
        ndcg = dcg / idcg
        avg_ndcg += ndcg
        mrr += mr
    total_queries = len(scores.keys())

    avg_ndcg /= total_queries
    mrr /= total_queries
    avg_len /= total_queries

    logging.info("The ndcg and mrr are {},{} with batch size 32 and lr of 1e5 and eval steps of 1000 with generated negative facets".format(avg_ndcg,mrr))

    model.train()
    return avg_ndcg, mrr


def train(output_path="weights_ranker",
          model_type="distilroberta-base",
          use_snippets=True,
          train_batch_size=16,
          eval_batch_size=24,
          lr=1e-5,
          accumulation_steps=4,
          warmup_steps=1000,
          max_seq_length=512,
          epochs=3,
          eval_steps=1000,
          log_steps=20,
          train_path="train.jsonl",
          dev_path="dev.jsonl",
          test_path="test.jsonl",
          use_gpu=True,
          parallel=True,
          fp16=False,
          wandb_log=True):
   
    if wandb_log:
        wandb.init(project="facet-extraction")
        wandb.config = {
            "learning_rate": lr,
            "epochs": epochs,
            "batch_size": train_batch_size
        }
    
    model = FacetRanker(model_type=model_type,
                        use_gpu=True,
                        parallel=parallel,
                        max_seq_length=max_seq_length)
    
     
    
    # train_dataset = PairwiseRankingDataset(data_path=train_path,
    #                                        max_seq_length=max_seq_length,
    #                                        model_type=model_type,
    #                                        tokenizer=model.tokenizer,
    #                                        num_negatives=4,
    #                                        queries_per_batch=2)
    # train_dataloader = train_dataset.batch_generator()
    train_dataset = PointwiseRankingDataset(data_path=train_path,
                                            max_seq_length=max_seq_length,
                                            model_type=model_type,
                                            tokenizer=model.tokenizer,
                                            batch_size=train_batch_size,
                                            random_split=0.2)
                            
    train_dataloader = train_dataset.batch_generator()


    dev_dataset = RankingDataset(data_path=dev_path,
                                 use_snippets=use_snippets,
                                 max_seq_length=max_seq_length,
                                 model_type=model_type,
                                 tokenizer=model.tokenizer)

    dev_dataloader = DataLoader(dataset=dev_dataset,
                                batch_size=eval_batch_size,
                                shuffle=False,
                                collate_fn=collate_batch)

    test_dataset = RankingDataset(data_path=test_path,
                                  use_snippets=use_snippets,
                                  max_seq_length=max_seq_length,
                                  model_type=model_type,
                                  tokenizer=model.tokenizer)
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=eval_batch_size,
                                 shuffle=False,
                                 collate_fn=collate_batch)
   

    total_examples = train_dataset.total_examples
    optimizer = AdamW(model.parameters(), lr=lr, eps=1e-6, correct_bias=False)
    total_steps = math.ceil(total_examples / (train_batch_size * accumulation_steps)) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    writer = SummaryWriter()
    best_ndcg = 0
    steps = 0
    accumulated_steps = 0
    running_loss = 0.0 
    scaler = GradScaler()
    for epoch in range(epochs):
        iterator = tqdm(train_dataloader, position=0)
        theEvaluation =0
        for batch in iterator:
            if fp16:
                with autocast():
                    loss = model.pointwise_loss(batch)
                scaler.scale(loss).backward()
            else:
                
                loss = model.pointwise_loss(batch)
               
                loss.backward()
        
            
           
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            #accumulation steps is the number of steps, we are running it every time
            if ((steps + 1) % accumulation_steps == 0) or (steps + 1 == total_steps):
               
                batch_loss_value = loss.item()
                running_loss += batch_loss_value
                if fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()
                accumulated_steps += 1
                if accumulated_steps % eval_steps == 0:
                    with torch.no_grad():
                        dev_ndcg, dev_mrr = evaluate(model, dev_dataloader)
                        test_ndcg, test_mrr = evaluate(model, test_dataloader)
                    writer.add_scalar("dev_ndcg", dev_ndcg, accumulated_steps)
                    writer.add_scalar("dev_mrr", dev_mrr, accumulated_steps)
                    writer.add_scalar("test_ndcg", test_ndcg, accumulated_steps)
                    writer.add_scalar("test_mrr", test_mrr, accumulated_steps)
                    if wandb_log:
                        wandb.log({"dev_ndcg": dev_ndcg})
                        wandb.log({"dev_mrr": dev_mrr})
                        wandb.log({"test_ndcg": test_ndcg})
                        wandb.log({"test_mrr": test_mrr})
                    if dev_ndcg > best_ndcg:
                        best_ndcg = dev_ndcg
                        model.save_model(output_path=output_path)
                # every 10 batchees
                if accumulated_steps % log_steps == 0:
                    writer.add_scalar("loss", running_loss / log_steps, accumulated_steps)
                    if wandb_log:
                        wandb.log({"loss": running_loss / log_steps})
                    running_loss = 0.0

                iterator.set_description("loss: {}, acc_steps: {}/{}".format(batch_loss_value,
                                                                             accumulated_steps,
                                                                             total_steps))

            steps += 1


def eval(model_type="distilroberta-base",
         eval_batch_size=16,
         max_seq_length=512,
         dev_path="dev.jsonl",
         test_path="test_scoring.jsonl",
         use_gpu=True,
         parallel=True):
   
    model = FacetRanker(model_path="models/weights_ranker_5",
                        model_type=model_type,
                        use_gpu=use_gpu,
                        parallel=parallel,
                        max_seq_length=max_seq_length)

   
    # dev_dataset = RankingDataset(data_path=dev_path,
    #                              max_seq_length=max_seq_length,
    #                              model_type=model_type,
    #                              tokenizer=model.tokenizer)
    # #wrapping the dataset using the dataloader

    # dev_dataloader = DataLoader(dataset=dev_dataset,
    #                             batch_size=eval_batch_size,
    #                             shuffle=False,
    #                             collate_fn=collate_batch)
    
   
    test_dataset = RankingDataset(data_path=test_path,
                                  max_seq_length=max_seq_length,
                                  model_type=model_type,
                                  tokenizer=model.tokenizer)
  
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=eval_batch_size,
                                 shuffle=False,
                                 collate_fn=collate_batch)
    # dev_ndcg, dev_mrr = evaluate(model, dev_dataloader)
    test_ndcg, test_mrr = evaluate(model, test_dataloader)
    return test_ndcg, test_mrr

    

if __name__ == "__main__":
    wandb.init()
    logging.basicConfig(filename = "logss.log", level =logging.INFO, filemode = "w")
    logging.info("The run name on wandb is {}".format(wandb.run.name))
    
    # train(output_path="weights_ranker_5",
    #       model_type="distilroberta-base",
    #       use_snippets=True,
    #       train_batch_size=32,
    #       eval_batch_size=32,
    #       accumulation_steps=1,
    #       lr=1e-5,
    #       warmup_steps=100,
    #       max_seq_length=512,
    #       epochs=1000,
    #       eval_steps=1000,
    #       log_steps=10,
    #       wandb_log=True,
    #       train_path="train.jsonl",
    #       dev_path="dev.jsonl",
    #       test_path="test.jsonl",
    #       use_gpu=True,
    #       parallel=True,
    #       fp16=False)

    eval()
