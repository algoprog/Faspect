import numpy

from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize


class FacetDiversifier:
    def __init__(self, model_name="algoprog/mimics-query-facet-encoder-mpnet-base"):
        self.text_encoder = SentenceTransformer(model_name)

    def maximal_marginal_relevance(self, query, results, lamda=0.5):
        results_ = [r.replace(query, "").replace("  ", " ") for r in results]
        embeddings = self.text_encoder.encode(results_)
        embeddings = normalize(embeddings)

        query_embedding = self.text_encoder.encode([query])
        query_embedding = normalize(query_embedding)[0]

        added_results = []
        docs = [(i, doc) for i, doc in enumerate(results)]

        added_results.append(docs[0]+(1., 1.))
        docs.pop(0)
        while len(docs) > 0:
            max_score = -1
            max_sim = -1
            max_id = -1
            for i, result in enumerate(docs):
                facet_relevance = numpy.dot(embeddings[result[0]], query_embedding)
                max_sim_ = -1
                for added_result in added_results:
                    sim = numpy.dot(embeddings[result[0]], embeddings[added_result[0]])
                    if sim > max_sim_:
                        max_sim_ = sim
                score = lamda * facet_relevance - (1 - lamda) * max_sim_
                if score > max_score:
                    max_score = score
                    max_id = i
                    max_sim = max_sim_
            added_results.append(docs[max_id]+(max_score, max_sim))
            docs.pop(max_id)

        added_results = [r[1] for r in added_results]

        return added_results


if __name__ == "__main__":
    dv = FacetDiversifier()
    res = ["city council", "fairfield city", "fairfield city council", "city of fairfield files",
           "departments city of fairfield", "city clerk", "city center", "city of just", "mission of the city",
           "district of smithfield and fairfield", "fairfield is a city", "city center of oakland",
           "wikipedia fairfield is a city", "city center of both cities", "fairfield fire", "four city", "one city",
           "fairfield fire department", "handbook from city", "city clerk july"]
    print(res)
    print("---")
    r = dv.maximal_marginal_relevance(query="city of fairfield", results=res, lamda=0.7)
    print(r)

