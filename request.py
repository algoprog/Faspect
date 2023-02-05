import requests

resp = requests.post("http://localhost:8000/predict")
print(resp.text)
print(resp.json())