import json
import operator
import os

from collections import defaultdict
from importlib import import_module
from typing import Dict, Tuple, List, Optional

import numpy as np

from flask import Flask, request
from flask_cors import CORS
from torch import nn
from torch.utils.data import Dataset
from transformers import AutoConfig, AutoTokenizer, AutoModelForTokenClassification, HfArgumentParser, \
    TrainingArguments, Trainer, PreTrainedTokenizer

from models.extractive.tagging.train import ModelArguments, DataTrainingArguments
from models.extractive.tagging.utils_token_classification import TokenClassificationTask, InputFeatures, InputExample
from syntok.tokenizer import Tokenizer


class TaggingDataset(Dataset):
    features: List[InputFeatures]
    pad_token_label_id: int = nn.CrossEntropyLoss().ignore_index

    def __init__(
            self,
            token_classification_task: TokenClassificationTask,
            texts,
            tokenizer: PreTrainedTokenizer,
            labels: List[str],
            model_type: str,
            max_seq_length: Optional[int] = None
    ):
        examples = [InputExample(guid='f-{}'.format(i), words=words, labels=['O'] * len(words)) for i, words in
                    enumerate(texts)]
        self.features = [token_classification_task.convert_example_to_features(
            example,
            labels,
            max_seq_length,
            tokenizer,
            cls_token_at_end=bool(model_type in ["xlnet"]),
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if model_type in ["xlnet"] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=False,
            pad_on_left=bool(tokenizer.padding_side == "left"),
            pad_token=tokenizer.pad_token_id,
            pad_token_segment_id=tokenizer.pad_token_type_id,
            pad_token_label_id=self.pad_token_label_id,
        ) for example in examples]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]


def align_predictions(predictions: np.ndarray, label_ids: np.ndarray, label_map) -> Tuple[List[int], List[int]]:
    preds = np.argmax(predictions, axis=2)

    batch_size, seq_len = preds.shape

    out_label_list = [[] for _ in range(batch_size)]
    preds_list = [[] for _ in range(batch_size)]

    for i in range(batch_size):
        for j in range(seq_len):
            if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                out_label_list[i].append(label_map[label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])

    return preds_list, out_label_list


class SupervisedFacetExtractorTagging:
    def __init__(self, model_name='algoprog/mimics-tagging-roberta-base', model=None):
        self.labels = ['O', 'B-FACET', 'I-FACET']
        self.label_map: Dict[int, str] = {i: label for i, label in enumerate(self.labels)}
        num_labels = len(self.labels)

        self.config = AutoConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=self.label_map,
            label2id={label: i for i, label in enumerate(self.labels)}
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False,
        )
        if model is None:
            self.model = AutoModelForTokenClassification.from_pretrained(
                model_name,
                config=self.config
            )
        else:
            self.model = model

        self.model.eval()

        self.model_config = {
            "data_dir": "data",
            "model_name_or_path": "roberta-base",
            "output_dir": "weights",
            "max_seq_length": 400,
            "num_train_epochs": 50,
            "per_device_train_batch_size": 16,
            "per_device_eval_batch_size": 16,
            "gradient_accumulation_steps": 4,
            "learning_rate": 5e-5,
            "seed": 43,
            "eval_steps": 500,
            "logging_steps": 10,
            "do_train": True,
            "overwrite_output_dir": True
        }

        parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
        model_args, data_args, training_args = parser.parse_dict(self.model_config)
        module = import_module("models.extractive.tagging.tasks")
        token_classification_task_clazz = getattr(module, model_args.task_type)
        self.token_classification_task: TokenClassificationTask = token_classification_task_clazz()

        self.trainer = Trainer(
            model=self.model,
            args=training_args
        )

        self.word_tokenizer = Tokenizer()

        self.app = Flask(__name__)
        CORS(self.app)

    def predict(self, query_text_pairs):
        facets = []
        texts = []
        for query, text in query_text_pairs:
            model_input = '{} || {}'.format(query, text).lower()
            doc = self.word_tokenizer.tokenize(model_input)
            words = [word.value for word in doc]
            texts.append(words)
        test_dataset = TaggingDataset(
            token_classification_task=self.token_classification_task,
            texts=texts,
            tokenizer=self.tokenizer,
            labels=self.labels,
            model_type=self.config.model_type,
            max_seq_length=400
        )
        predictions, label_ids, metrics = self.trainer.predict(test_dataset)
        preds_list, _ = align_predictions(predictions, label_ids, self.label_map)
        preds_list = list(preds_list)

        for i, pred_labels in enumerate(preds_list):
            lstr = ' '.join(pred_labels)
            preds_list[i] = lstr.split(' ')
        snippet_facets = []

        for i, pred_labels in enumerate(preds_list):
            snippet_facets.append([])
            facet_text = ''
            for j, label in enumerate(pred_labels):
                if label != 'O':
                    if label == 'B-FACET' and facet_text != '':
                        snippet_facets[i].append(facet_text.strip())
                        facet_text = ''
                    facet_text += texts[i][j] + ' '
                else:
                    if facet_text != '':
                        snippet_facets[i].append(facet_text.strip())
                        facet_text = ''
            if facet_text != '':
                snippet_facets[i].append(facet_text.strip())

        for facet_list in snippet_facets:
            facets.append([e for e in facet_list])

        return facets

    def extract(self, batch_queries, batch_snippets, batch_size=32):
        is_batch = True
        if not isinstance(batch_queries, list):
            batch_queries = [batch_queries]
            batch_snippets = [batch_snippets]
            is_batch = False

        batch_facets = []

        for i, q in enumerate(batch_queries):
            facets_freq = defaultdict(lambda: 0)
            pairs = [(q, snippet) for snippet in batch_snippets[i]]
            facets = self.predict(pairs)
            for snippet_facets in facets:
                for facet in snippet_facets:
                    facets_freq[facet.lower()] += 1
            facets_sorted = sorted(facets_freq.items(), key=operator.itemgetter(1), reverse=True)
            facets_values = [f[0] for f in facets_sorted]
            batch_facets.append(facets_values)

        if not is_batch:
            return batch_facets[0]
        else:
            return batch_facets

    def build_endpoints(self):
        @self.app.route('/tag', methods=['POST', 'GET'])
        def search_endpoint():
            query = request.args.get('query')
            text = request.args.get('text')
            results = json.dumps(self.predict([(query, text)])[0], indent=4)
            return results

    def serve(self, port=80):
        self.build_endpoints()
        self.app.run(host='0.0.0.0', port=port)


if __name__ == "__main__":
    e = SupervisedFacetExtractorTagging()
    r = e.extract("2010 cadillac srx", [
        "Used 2010 Cadillac SRX Values & Cars for sale | Kelley... Learn more about used 2010 Cadillac SRX vehicles. Get 2010 Cadillac SRX values, consumer reviews, safety ratings, and find cars for sale near you."])
    print(r)
