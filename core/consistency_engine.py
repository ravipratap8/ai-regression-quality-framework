from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class ConsistencyEngine:

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def evaluate_consistency(self, expected_text, responses):
        embeddings = self.model.encode([expected_text] + responses)

        expected_embedding = embeddings[0]
        response_embeddings = embeddings[1:]

        similarities = [
            cosine_similarity(
                [expected_embedding],
                [resp_emb]
            )[0][0]
            for resp_emb in response_embeddings
        ]

        average_similarity = float(np.mean(similarities))
        variance = float(np.var(similarities))

        return {
            "average_similarity": average_similarity,
            "variance": variance,
            "individual_scores": similarities
        }
