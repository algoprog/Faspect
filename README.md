# Faspect
 
Faspect is a library with various model implementations for open domain query facet extraction and generation.

# Installation

```
pip install -r requirements.txt
```

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

or use the classes from the *models* folder in your project.
