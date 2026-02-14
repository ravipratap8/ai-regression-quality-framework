print("Starting environment verification...\n")

try:
    import openai
    print("openai installed")
except Exception as e:
    print("openai FAILED:", e)

try:
    from sentence_transformers import SentenceTransformer
    print("sentence-transformers installed")
except Exception as e:
    print("sentence-transformers FAILED:", e)

try:
    import scipy
    print("scipy installed")
except Exception as e:
    print("scipy FAILED:", e)

try:
    import textstat
    print("textstat installed")
except Exception as e:
    print("textstat FAILED:", e)

print("\nLoading sentence transformer model...")

try:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(["AI testing is important", "Quality engineering matters"])
    print("Sentence transformer working")
    print("Embedding shape:", len(embeddings), "vectors")
except Exception as e:
    print("Sentence transformer FAILED:", e)

print("\nEnvironment check complete.")
