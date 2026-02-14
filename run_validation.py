import json
from core.similarity_engine import SimilarityEngine
from core.consistency_engine import ConsistencyEngine
from core.threshold_rules import ThresholdRules


# Toggle this to simulate unstable AI behaviour
CHAOS_MODE = True


def load_dataset(path):
    with open(path, "r") as f:
        return json.load(f)


def generate_responses(expected_answer):
    """
    Simulates model responses.
    Replace this later with real LLM integration.
    """

    if CHAOS_MODE:
        return [
            expected_answer,
            expected_answer.replace("Canberra", "Sydney") if "Canberra" in expected_answer else expected_answer,
            "Completely wrong answer"
        ]
    else:
        return [
            expected_answer,
            expected_answer,
            expected_answer
        ]


def main():
    print("\nStarting AI Regression Validation Framework...\n")

    similarity_engine = SimilarityEngine()
    consistency_engine = ConsistencyEngine()
    rules = ThresholdRules()

    dataset = load_dataset("data/golden_dataset.json")

    semantic_pass_count = 0
    consistency_pass_count = 0

    for test_case in dataset:
        prompt = test_case["prompt"]
        expected = test_case["expected_answer"]

        responses = generate_responses(expected)

        print("=" * 50)
        print("Prompt:", prompt)

        # --- Semantic Validation (first response only) ---
        similarity_score = similarity_engine.compute_similarity(
            expected, responses[0]
        )
        semantic_pass = rules.semantic_pass(similarity_score)

        if semantic_pass:
            semantic_pass_count += 1

        print("Semantic Score:", round(similarity_score, 4))
        print("Semantic Pass:", semantic_pass)

        # --- Consistency Validation (all responses) ---
        consistency_results = consistency_engine.evaluate_consistency(
            expected, responses
        )
        consistency_pass = rules.consistency_pass(
            consistency_results["variance"]
        )

        if consistency_pass:
            consistency_pass_count += 1

        print("Average Similarity:", round(consistency_results["average_similarity"], 4))
        print("Variance:", round(consistency_results["variance"], 6))
        print("Consistency Pass:", consistency_pass)

    # --- Summary ---
    print("\n" + "=" * 50)
    print("REGRESSION SUMMARY")
    print("=" * 50)
    print("Total Test Cases:", len(dataset))
    print("Semantic Passed:", semantic_pass_count)
    print("Consistency Passed:", consistency_pass_count)

    overall_status = (
        semantic_pass_count == len(dataset)
        and consistency_pass_count == len(dataset)
    )

    print("Overall Regression Status:", "PASSED" if overall_status else "FAILED")
    print("\nFramework execution complete.\n")


if __name__ == "__main__":
    main()
