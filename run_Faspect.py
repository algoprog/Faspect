# from faspect import Faspect
from sentence_transformers import SentenceTransformer, InputExample
from huggingface_hub import utils,whoami,HfFolder,create_repo, HfApi
from huggingface_hub.utils import validate_repo_id
from huggingface_hub import HfApi
from faspect import Faspect
# api = HfApi()
# api.upload_folder(
#     repo_id="umass/roberta-base-mimics-facet-reranker",
#     folder_path='weights_ranker_5/model.state_dict',
#     repo_type="model",
# )
# print("done")


# repo_name = "clustering-Model-3"


# model = SentenceTransformer('models/Clustering/weights_microsoft/mpnet-base')
# url = model.save_to_hub(repo_name = repo_name,private=None)
# print(url)
# facet_extractor = Faspect()
# def trial():
    
#     query = "cars"
#     documents = ["Shop new & used cars, research & compare models, find local dealers/sellers,calculate payments, value your car, sell/trade in your car & more at Cars.com.",
#              "Search over 48789 used Cars in Amherst, MA. TrueCar has over 861570 listings nationwide, updated daily. Come find a great deal on used Cars in Amherst today!",
#              "Cars is a 2006 American computer-animated sports comedy film produced by Pixar Animation Studios and released by Walt Disney Pictures.",
#              "Search for used cars at carmax.com. Use our car search or research makes and models with customer reviews, expert reviews, and more."]
#     facets = facet_extractor.extract_facets(query, 
#                                         documents,
#                                         aggregation="threshold", # mmr, round-robin
#                                         mmr_lambda=0.5,
#                                         classification_threshold=0.05,
#                                         classification_topk=0)
#     clusters = facet_extractor.cluster_facets(query,documents,facets)
#     thisDict = {}
#     for each in clusters:
#         thisDict[" ".join(each)] = facet_extractor.generate_clarifying_questions(query,documents,each)
#     return thisDict


facet_extractor = Faspect()

query = "cars"

documents = ["Shop new & used cars, research & compare models, find local dealers/sellers,calculate payments, value your car, sell/trade in your car & more at Cars.com.",
             "Search over 48789 used Cars in Amherst, MA. TrueCar has over 861570 listings nationwide, updated daily. Come find a great deal on used Cars in Amherst today!",
             "Cars is a 2006 American computer-animated sports comedy film produced by Pixar Animation Studios and released by Walt Disney Pictures.",
             "Search for used cars at carmax.com. Use our car search or research makes and models with customer reviews, expert reviews, and more."]
facets = facet_extractor.extract_facets(query, 
                                        documents,
                                        aggregation="threshold", # mmr, round-robin
                                        mmr_lambda=0.5,
                                        classification_threshold=0.05,
                                        classification_topk=0)
clusters = facet_extractor.cluster_facets(query,documents,facets)


print("facets:", facets)

print(clusters)

for each in clusters:
    print(facet_extractor.generate_clarifying_questions(query,documents,each))
