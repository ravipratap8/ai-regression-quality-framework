from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import numpy as np

print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")


def semantic_similarity(text1, text2):
    emb1 = model.encode(text1)
    emb2 = model.encode(text2)
    return 1 - cosine(emb1, emb2)


# Simulated LLM responses for same prompt
expected = "The capital of Australia is Canberra."

responses = [
    "Canberra is the capital city of Australia.",
    "Australia's capital is Canberra.",
    "Sydney is the capital of Australia.",   # wrong
    "The capital city of Australia is Canberra."
]

scores = []

for r in responses:
    score = semantic_similarity(expected, r)
    scores.append(score)
    print("\nResponse:", r)
    print("Similarity:", round(score, 4))

print("\nAverage similarity:", round(np.mean(scores), 4))
print("Variance:", round(np.var(scores), 4))
