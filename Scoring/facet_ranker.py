from multiprocessing.dummy import active_children
import os
import torch
import logging
from torch import nn
from torch.nn import DataParallel
import transformers
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from Scoring.utils_ranking_model import collate_batch, RankingDataset
from huggingface_hub import hf_hub_url, cached_download, hf_hub_download
import joblib


class FacetRanker(nn.Module):
    def __init__(self,
                 model_path="umass/roberta-base-mimics-facet-reranker",
                 model_type="roberta-base",
             use_gpu=True,
                 parallel=False,
                 debug=False, max_seq_length= 512):
        super(FacetRanker, self).__init__()

        self.model_type = model_type

        configuration = AutoConfig.from_pretrained(self.model_type)
        if model_path is None:
            # this is the code that will be executed
            self.bert = AutoModel.from_pretrained(self.model_type)
        else:
            self.bert = AutoModel.from_config(configuration)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_type)
        self.hidden_dim = configuration.hidden_size
        self.max_seq_length = max_seq_length
        self.score = nn.Linear(self.hidden_dim, 1)

        if parallel:
            self.bert = DataParallel(self.bert)
        if model_path is not None:
            model_path = hf_hub_download(repo_id=model_path, filename="model.state_dict")
            sdict = torch.load(model_path, map_location=lambda storage, loc: storage)
            self.load_state_dict(sdict, strict=False)
        self.device = torch.device("cuda") if use_gpu else torch.device("cpu")
        self.to(self.device)
        self.debug = debug

    
    def pairwise_loss(self, batch):
        pos_logits = self.forward(batch[0], batch[1])
        neg_logits = self.forward(batch[2], batch[3])
        loss = torch.mean(torch.log(1 + torch.exp(-torch.sub(pos_logits, neg_logits))), dim=0)
        return loss

    def pointwise_loss(self, batch):
        loss_fn = nn.BCELoss()
        scores = torch.sigmoid(self.forward(batch[0], batch[1]))
        loss = loss_fn(scores, batch[2].view(-1, 1))
        return loss

    def forward(self, input_ids, attention_mask):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        for x in range(len(input_ids)):
            if(len(input_ids[x])!=512 or len(attention_mask[x])!=512):
                logging.info("the length is not equal to 512 of the input ids or the attention mask in the foreward function {} {}".format(len(input_ids),len(attention_mask)))
        cls = self.bert(input_ids=input_ids, attention_mask=attention_mask)[1]
        scores = self.score(cls)
        return  scores

    def score_facets(self, query_facet_snippets, batch_size=8):
        dataset = RankingDataset(data=query_facet_snippets,
                                 max_seq_length=self.max_seq_length,
                                 model_type=self.model_type,
                                 tokenizer=self.tokenizer)
        dataloader = DataLoader(dataset=dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                collate_fn=collate_batch)
        scores = []
        for input_ids, attention_mask in dataloader:
            scores_ = self.forward(input_ids, attention_mask).data.cpu().numpy()
            #scores_ = [s[0] for s in scores_]
            scores.extend(scores_)
        return scores

    def load_model(self, sdict):
        self.load_state_dict(sdict)
        self.to(self.device)

    def save_model(self, output_path):
        model_name = 'model.state_dict'
        opath = os.path.join(output_path, model_name)
        torch.save(self.state_dict(), opath)
