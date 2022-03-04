import json
import logging
import torch

from flask import Flask, request
from flask_cors import CORS

from simpletransformers.seq2seq import (
            Seq2SeqModel,
            Seq2SeqArgs,
        )


class SupervisedFacetExtractorSeq2seq:
    def __init__(self, model_name='algoprog/mimics-bart-base', only_query=False):
        logging.basicConfig(level=logging.INFO)
        transformers_logger = logging.getLogger("transformers")
        transformers_logger.setLevel(logging.WARNING)

        model_args = Seq2SeqArgs()
        model_args.eval_batch_size = 1
        model_args.max_seq_length = 200
        model_args.use_multiprocessing = False
        model_args.use_multiprocessing_for_evaluation = False

        model_args.do_sample = False
        model_args.num_beams = 4
        model_args.num_return_sequences = 1
        model_args.top_k = 100
        model_args.top_p = 0.8
        model_args.max_length = 40
        model_args.early_stopping = True
        model_args.temperature = 0.7

        # Initialize model
        self.model = Seq2SeqModel(
            encoder_decoder_type="bart",
            encoder_decoder_name=model_name,
            args=model_args,
            use_cuda=torch.cuda.is_available()
        )

        self.only_query = only_query

        self.app = Flask(__name__)
        CORS(self.app)

    def extract(self, batch_queries, batch_snippets=None):
        is_batch = True
        if not isinstance(batch_queries, list):
            batch_queries = [batch_queries]
            if batch_snippets is not None:
                batch_snippets = [batch_snippets]
            is_batch = False

        texts = []
        for i, q in enumerate(batch_queries):
            text = q
            if batch_snippets is not None:
                if self.only_query:
                    text = q
                else:
                    text = " || ".join(batch_snippets[i])
                    text = "{} || {}".format(q, text)
            texts.append(text)
        facets = self.model.predict(texts)
        facets = [list(set(f.split(", "))) for f in facets]

        if not is_batch:
            return facets[0]
        else:
            return facets

    def build_endpoints(self):
        @self.app.route('/generate', methods=['POST', 'GET'])
        def search_endpoint():
            query = request.args.get('query')
            texts = [request.args.get('text')]
            results = json.dumps(self.extract(query, texts), indent=4)
            return results

    def serve(self, port=80):
        self.build_endpoints()
        self.app.run(host='0.0.0.0', port=port)


if __name__ == "__main__":
    e = SupervisedFacetExtractorSeq2seq()
    r = e.extract(batch_queries=["2010 cadillac srx"],
                  batch_snippets=[["Used 2010 Cadillac SRX Values & Cars for sale | Kelley... Learn more about used 2010 Cadillac SRX vehicles. Get 2010 Cadillac SRX values, consumer reviews, safety ratings, and find cars for sale near you."]])
    print(r)
