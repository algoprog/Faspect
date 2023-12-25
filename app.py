from flask import Flask, jsonify
from sentence_transformers import SentenceTransformer, InputExample
from huggingface_hub import utils,whoami,HfFolder,create_repo, HfApi
from huggingface_hub.utils import validate_repo_id
from huggingface_hub import HfApi
from faspect import Faspect
from trial import trial

app = Flask(__name__)





@app.route('/')
def hello():
    # ans = trial()
    # print(ans)
    return "jsonify(ans)"

@app.route('/predict', methods=['POST'])
def predict():
    ans = trial()
    print(ans)
    return jsonify(ans)