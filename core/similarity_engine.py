from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SimilarityEngine:

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def compute_similarity(self, expected_text, model_output):
        embeddings = self.model.encode([expected_text, model_output])
        similarity = cosine_similarity(
            [embeddings[0]],
            [embeddings[1]]
        )[0][0]
        return float(similarity)
