import json
import logging

from itertools import cycle, islice
from flask import request, Flask
from flask_cors import CORS
from models.ranking import FacetDiversifier
from models.extractive.unsupervised.unsupervised import UnsupervisedFacetExtractor
from models.abstractive.seq2seq import SupervisedFacetExtractorSeq2seq
from models.extractive.tagging.tagging import SupervisedFacetExtractorTagging
from models.classification.facet_classification import FacetClassifier
from Scoring.facet_ranker import FacetRanker
from Scoring.utils_ranking_model import convert_example_to_features, collate_batch
from transformers import AutoConfig, AutoModel, AutoTokenizer
import numpy as np
from Clustering.subClustering import Clustering
from Clarifying_questions.Main_trainer_load_Faspect import Clarifying_question

def batch_generator(query,snippets,facet_list,model_type="distilroberta-base"):
        tokenizer = AutoTokenizer.from_pretrained(model_type)
        batch = []
        for facet in facet_list:
            text = "{} || {} || {}".format(query,facet,snippets)
            features = convert_example_to_features(
                        text,
                        400,
                        tokenizer,
                        cls_token_at_end=bool(model_type in ["xlnet"]),
                        cls_token=tokenizer.cls_token,
                        cls_token_segment_id=2 if model_type in ["xlnet"] else 0,
                        sep_token=tokenizer.sep_token,
                        sep_token_extra=False,
                        pad_on_left=bool(tokenizer.padding_side == "left"),
                        pad_token=tokenizer.pad_token_id,
                        pad_token_segment_id=tokenizer.pad_token_type_id)
            batch.append(features)
        return collate_batch(batch, all_features=True)

def threshold(threshold_model,query,facet_list,snippets):
    data = batch_generator(query,snippets,facet_list)
    input_ids, attention_mask   = data
    scores_ = threshold_model.forward(input_ids, attention_mask).data.cpu().numpy()
    scores_ = 1/(1 + np.exp(-scores_))
    relevant_facets = []
    for i, facet in enumerate(facet_list):
        if(scores_[i]>=0.9):
            relevant_facets.append(facet)
    return relevant_facets

    
def generate_prediction(seed_line, tokenizer, model_):
  # Put the model on eval mode
  model_.to("cuda")
  model_.eval()
  
  prompt_line_tokens = tokenizer(seed_line, max_length = 192, return_tensors = "pt", padding=True,truncation = True)

  line = model_.generate_text(prompt_line_tokens, eval_beams = 8)


  return line


def roundrobin(*iterables):
    "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
    # Recipe credited to George Sakkis
    num_active = len(iterables)
    nexts = cycle(iter(it).__next__ for it in iterables)
    while num_active:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            num_active -= 1
            nexts = cycle(islice(nexts, num_active))


def remove_duplicates(items):
    added = set()
    added_list = []
    for item in items:
        if item not in added:
            added.add(item)
            added_list.append(item)
    return added_list


class Faspect:
    def __init__(self):
        logging.getLogger().setLevel(logging.INFO)

        logging.info("Loading models...")

        self.abstractive_extractor = SupervisedFacetExtractorSeq2seq(model_name="algoprog/mimics-bart-base")
        self.abstractive_query_extractor = SupervisedFacetExtractorSeq2seq(model_name="algoprog/mimics-query-bart-base")
        self.extractive_extractor = SupervisedFacetExtractorTagging(model_name="algoprog/mimics-tagging-roberta-base")
        self.classifier = FacetClassifier(model_name="algoprog/mimics-multilabel-roberta-base-787",
                                          labels_path="models/classification/facets.json")
        self.unsupervised_extractor = UnsupervisedFacetExtractor()

        self.ranker = FacetDiversifier(model_name="algoprog/mimics-query-facet-encoder-mpnet-base")
        self.threshold_model = FacetRanker(model_path="models/Scoring/weights_ranker_5",
                        model_type="distilroberta-base",
                        use_gpu=True,
                        parallel=True,
                        max_seq_length=512)
        self.clustering_model = Clustering('umass/mpnet-base-mimics-query-facet-encoder')
        self.clarifying_question_model = Clarifying_question("Clarifying_questions/Models/facebook-bart-large.ckpt")

        logging.info("Finished loading.")

        self.app = Flask(__name__)
        CORS(self.app)

    def extract_facets(self, query, docs,
                       aggregation="threshold",
                       mmr_lambda=0.5,
                       classification_threshold=0.05,
                       classification_topk=0):
        """
        Extracts facets for a given query and documents
        :param query: 
        :param docs: a list of documents
        :param aggregation: "round-robin", "mmr" or "rank"
        :param mmr_lambda: the parameter used by mmr (relevance weight)
        :param classification_topk: the topk classes returned by the multi-label model
        :param classification_threshold: instead of topk classes, return classes based on a threshold
        :return: list of facet terms

        """
        facets_abstractive = self.abstractive_extractor.extract(batch_queries=query, batch_snippets=docs)
        facets_abstractive_query = self.abstractive_query_extractor.extract(batch_queries=query, batch_snippets=docs)
        facets_extractive = self.extractive_extractor.extract(batch_queries=query, batch_snippets=docs)
        facets_classifier = self.classifier.predict(query=query,
                                                    documents=docs,
                                                    threshold=classification_threshold,
                                                    topk=classification_topk)
        facets_unsupervised = [] #self.unsupervised_extractor.extract(batch_queries=query, batch_snippets=docs, limit=20)

        facets = list(roundrobin(facets_abstractive,
                                 facets_abstractive_query,
                                 facets_extractive,
                                 facets_classifier,
                                 facets_unsupervised))

        if aggregation == "mmr":
            facets = self.ranker.maximal_marginal_relevance(query, facets, lamda=mmr_lambda)
        elif aggregation == "rank":
            facets = self.ranker.maximal_marginal_relevance(query, facets, lamda=1.0)
        
        elif aggregation == "threshold":
            facets = threshold(self.threshold_model,query,facets,docs)

        facets = [f.lower() for f in facets]
        facets = remove_duplicates(facets)

        return facets

    def cluster_facets(self,query,snippets,facets):
        clusters = self.clustering_model.cluster_facets_query(query,snippets,facets)
        return clusters
    
    def generate_clarifying_questions(self,query,snippets,facets):
        facets = ", ".join(facets)
        snippets = " | ".join(snippets)
        seed_line = query + " | " + facets

        question_pred =  self.clarifying_question_model.generate_prediction(seed_line = seed_line)
        return  question_pred

    def build_endpoints(self):
        @self.app.route("/extract", methods=["GET", "POST"])
        def search_endpoint():
            params = request.json
            facets = self.extract_facets(params["query"], params["documents"])
            results = json.dumps({"facets": facets}, indent=4)
            return results

    def serve(self, port=80):
        self.build_endpoints()
        self.app.run(host='0.0.0.0', port=port)


if __name__ == "__main__":
    extractor = Faspect()
    extractor.serve(port=6000)
