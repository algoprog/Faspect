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

        logging.info("Finished loading.")

        self.app = Flask(__name__)
        CORS(self.app)

    def extract_facets(self, query, docs,
                       aggregation="round-robin",
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

        facets = [f.lower() for f in facets]
        facets = remove_duplicates(facets)

        return facets

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
