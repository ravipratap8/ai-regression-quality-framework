import pytest
from core.similarity_engine import SimilarityEngine


def test_identical_sentences_high_similarity():
    engine = SimilarityEngine()

    sentence1 = "The capital of Australia is Canberra."
    sentence2 = "The capital of Australia is Canberra."

    score = engine.compute_similarity(sentence1, sentence2)

    assert score > 0.95


def test_semantically_similar_sentences():
    engine = SimilarityEngine()

    sentence1 = "The capital of Australia is Canberra."
    sentence2 = "Canberra is the capital city of Australia."

    score = engine.compute_similarity(sentence1, sentence2)

    assert score > 0.85


def test_different_sentences_low_similarity():
    engine = SimilarityEngine()

    sentence1 = "The capital of Australia is Canberra."
    sentence2 = "Bananas grow in tropical climates."

    score = engine.compute_similarity(sentence1, sentence2)

    assert score < 0.70
