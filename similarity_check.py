from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine


print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")


def semantic_similarity(text1, text2):
    emb1 = model.encode(text1)
    emb2 = model.encode(text2)

    similarity = 1 - cosine(emb1, emb2)
    return round(float(similarity), 4)


# Example test
expected = "The capital of Australia is Canberra."
model_output = "Canberra is the capital city of Australia."

score = semantic_similarity(expected, model_output)

print("\nExpected:", expected)
print("Model Output:", model_output)
print("Similarity Score:", score)
