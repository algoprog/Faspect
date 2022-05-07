import argparse
import json
import torch

from typing import List
from simpletransformers.classification import (
    MultiLabelClassificationModel
)
from simpletransformers.config.model_args import MultiLabelClassificationArgs


class FacetClassifier:
    def __init__(self, model_name, labels_path):
        cuda_available = torch.cuda.is_available()

        model_args = MultiLabelClassificationArgs()
        model_args.use_multiprocessing = False
        model_args.use_multiprocessing_for_evaluation = False

        self.model = MultiLabelClassificationModel(
            "roberta",
            model_name,
            # "roberta-base",
            args=model_args,
            use_cuda=cuda_available)

        self.labels = []
        self.label_indx_map = {}
        self.get_labels(labels_path)

    def get_labels(self, file_path):
        # load and create a list of labels
        with open(f'{file_path}', 'r') as fobj:
            facet_dict = json.load(fobj)  # load the most frequent facets

        self.labels = sorted(list(facet_dict.keys()))
        self.label_indx_map = dict(zip(range(0, len(self.labels)), self.labels))

    def predict(self, query: str, documents: List[str], threshold=0.05, topk=5):
        sep_char = '</s>'
        doc_full = sep_char.join([query] + documents)
        predictions, raw_outputs = self.model.predict([doc_full])
        if topk > 0:
            pred_labels = [(self.labels[i], raw_outputs[0][i]) for i, x in enumerate(raw_outputs[0])]
            pred_labels = sorted(pred_labels, key=lambda x: x[1], reverse=True)[:topk]
            pred_labels = [x[0] for x in pred_labels]
        else:
            y_pred = (raw_outputs[0] >= threshold).astype(int)
            pred_labels = [self.labels[i] for i, x in enumerate(y_pred) if x == 1]

        return pred_labels


if __name__ == '__main__':
    #parser = argparse.ArgumentParser(description='facet model training')
    #parser.add_argument("--model_path", type=str, help="path to trained model")
    #parser.add_argument("--facet_path", type=str, help="path to list of facets")
    #args = parser.parse_args()

    fc_model = FacetClassifier("weights", "facets.json")

    print(f"Total labels: {len(fc_model.labels)}")

    query = "cars"
    docs = ["Shop new & used cars, research & compare models, find local dealers/sellers,calculate payments, value your car, sell/trade in your car & more at Cars.com."]

    pred_facets = fc_model.predict(query, docs)

    print(f"Query: {query}")
    print(f"Prediced: {pred_facets}")
