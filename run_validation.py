import json
from core.similarity_engine import SimilarityEngine
from core.consistency_engine import ConsistencyEngine
from core.threshold_rules import ThresholdRules

print("Starting AI Quality Validation Framework...")

# Load dataset
with open("data/golden_dataset.json", "r") as f:
    dataset = json.load(f)

similarity_engine = SimilarityEngine()
consistency_engine = ConsistencyEngine()
rules = ThresholdRules()

for item in dataset:

    print("\n==============================")
    print("Prompt:", item["prompt"])
    print("==============================")

    expected = item["expected"]

    # Simulated model outputs
    responses = [
        expected,
        expected.replace("Canberra", "Sydney") if "Australia" in expected else expected,
        expected
    ]

    # Semantic similarity check
    similarity_score = similarity_engine.compute_similarity(expected, responses[0])
    semantic_result = rules.semantic_pass(similarity_score)

    print("Semantic Similarity:", round(similarity_score, 4))
    print("Semantic Pass:", semantic_result)

    # Consistency check
    consistency_results = consistency_engine.evaluate_consistency(expected, responses)
    consistency_result = rules.consistency_pass(consistency_results["variance"])

    print("Average Similarity:", round(consistency_results["average_similarity"], 4))
    print("Variance:", round(consistency_results["variance"], 6))
    print("Consistency Pass:", consistency_result)

print("\nFramework execution complete.")
