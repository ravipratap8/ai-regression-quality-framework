class ThresholdRules:

    def __init__(self):
        self.semantic_threshold = 0.90
        self.consistency_variance_threshold = 0.01

    def semantic_pass(self, similarity_score):
        return similarity_score >= self.semantic_threshold

    def consistency_pass(self, variance):
        return variance <= self.consistency_variance_threshold
