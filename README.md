# Faspect
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1IVaxmq574aaPqEIlUiE0d-OPxsjXZyjX?usp=sharing)
 
Faspect is a library with various model implementations for open domain query facet extraction and generation. For more details you can have a look at our ICTIR 2022 paper: [Revisiting Open Domain Query Facet Extraction and Generation](https://ciir-publications.cs.umass.edu/pub/web/getpdf.php?id=1455).

# Installation

```
pip install -r requirements.txt
```

Install the required nltk packages
```
python -m nltk.downloader stopwords averaged_perceptron_tagger
```

You can also try an online demo on [Colab](https://colab.research.google.com/drive/1IVaxmq574aaPqEIlUiE0d-OPxsjXZyjX?usp=sharing) before running locally.

# Usage

Run faspect.py and use the API to extract facets, for example you can post json data in this format to `127.0.0.1:6000/extract`:

```json
{
    "query": "cars",
    "documents": ["doc 1 text", "doc 2 text", "doc N text"]
}
```

and get results:

```json
{
    "facets": [
       "cars for rent",
       "animated movie",
       "cars for sale"
    ]
}
```

or use the classes from the *models* folder in your project. For evaluation of your own models, please use the metric implementations in `evaluation.py`.

# Citation

If you use any part of this code, including model weights, please cite our [paper](https://ciir-publications.cs.umass.edu/pub/web/getpdf.php?id=1455):

```
@article{Samarinas_2022_Faspect,
  title   =  {Revisiting Open Domain Query Facet Extraction and Generation},
  author  =  {Chris Samarinas and Arkin Dharawat and Hamed Zamani},
  journal =  {Proceedings of the 2022 ACM SIGIR International Conference on Theory of Information Retrieval},
  year    =  {2022}
}
```

For any problems or suggestions, feel free to open an issue or contact [Chris Samarinas](mailto:chris.samarinas@gmail.com).
