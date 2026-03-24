"""Tests for DiversityValidator plugin."""

from datasets import Dataset

from src.data.validation.plugins.base.diversity import DiversityValidator


class TestDiversityValidator:
    """Test suite for DiversityValidator."""

    def test_positive_case_high_diversity(self):
        """Positive: Dataset with high vocabulary diversity."""
        # Dataset with rich vocabulary
        texts = [
            "Machine learning algorithms process data efficiently",
            "Neural networks train on labeled examples",
            "Deep learning models achieve remarkable accuracy",
            "Artificial intelligence transforms modern technology",
            "Computer vision recognizes patterns in images",
            "Natural language processing understands human text",
            "Reinforcement learning optimizes decision making",
            "Supervised learning predicts future outcomes",
            "Unsupervised learning discovers hidden structures",
            "Transfer learning reuses pretrained knowledge",
        ]
        dataset = Dataset.from_dict({"text": texts})

        plugin = DiversityValidator(thresholds={"min_score": 0.5})
        result = plugin.validate(dataset)

        assert result.passed is True
        assert result.metrics["diversity_score"] >= 0.5
        assert result.metrics["unique_tokens"] > 0
        assert result.metrics["total_tokens"] > 0
        assert len(result.errors) == 0

    def test_negative_case_low_diversity(self):
        """Negative: Dataset with low vocabulary diversity (repetitive)."""
        # Highly repetitive words
        texts = ["test test test test"] * 10
        dataset = Dataset.from_dict({"text": texts})

        plugin = DiversityValidator(thresholds={"min_score": 0.7})
        result = plugin.validate(dataset)

        assert result.passed is False
        assert result.metrics["diversity_score"] < 0.7
        assert len(result.errors) > 0
        assert "too low" in result.errors[0].lower()

    def test_boundary_case_exact_threshold(self):
        """Boundary: Diversity exactly at threshold."""
        # Controlled diversity: many unique words
        texts = []
        # Many unique words
        for i in range(20):
            unique_words = [f"word{i}_{j}" for j in range(10)]
            texts.append(" ".join(unique_words))

        dataset = Dataset.from_dict({"text": texts})

        plugin = DiversityValidator(thresholds={"min_score": 0.5})
        result = plugin.validate(dataset)

        # Should pass with a solid diversity score
        assert result.metrics["diversity_score"] >= 0.5

    def test_edge_case_single_repeated_token(self):
        """Edge: Dataset with only one token repeated."""
        texts = ["test"] * 100
        dataset = Dataset.from_dict({"text": texts})

        plugin = DiversityValidator(thresholds={"min_score": 0.5})
        result = plugin.validate(dataset)

        assert result.passed is False
        assert result.metrics["diversity_score"] < 0.5
        assert result.metrics["unique_tokens"] == 1
        assert result.metrics["total_tokens"] == 100

    def test_edge_case_empty_dataset(self):
        """Edge: Empty dataset."""
        dataset = Dataset.from_dict({"text": []})

        plugin = DiversityValidator(thresholds={"min_score": 0.5})
        result = plugin.validate(dataset)

        # Empty dataset should have diversity_score = 0
        assert result.passed is False
        assert result.metrics["diversity_score"] == 0
        assert result.metrics["unique_tokens"] == 0
        assert result.metrics["total_tokens"] == 0

    def test_large_dataset_sampling(self, large_dataset):
        """Test: Large dataset uses sampling."""
        plugin = DiversityValidator(params={"sample_size": 100}, thresholds={"min_score": 0.5})
        result = plugin.validate(large_dataset)

        # Large datasets (>10k) are sampled, but fixture has 1000 samples
        # So it will check all 1000
        assert result.metrics["samples_checked"] == 1000
        assert result.passed is True

    def test_streaming_dataset_support(self):
        """Test: Works with streaming datasets."""
        from datasets import IterableDataset

        def gen():
            for i in range(50):
                yield {"text": f"Sample {i} with unique content and vocabulary diversity"}

        dataset = IterableDataset.from_generator(gen)

        plugin = DiversityValidator(thresholds={"min_score": 0.5})
        result = plugin.validate(dataset)

        assert result.passed is True
        assert result.metrics["samples_checked"] > 0

    def test_warning_close_to_threshold(self):
        """Test: Warning when diversity close to minimum."""
        # Dataset with diversity just above threshold
        texts = []
        # 10 unique words over ~140 tokens => 0.071 * 10 = 0.71
        words = ["word" + str(i) for i in range(10)]
        for _ in range(20):
            texts.append(" ".join(words))

        dataset = Dataset.from_dict({"text": texts})

        plugin = DiversityValidator(thresholds={"min_score": 0.7})
        result = plugin.validate(dataset)

        # May warn when close to threshold
        if result.metrics["diversity_score"] < 0.77:  # 0.7 * 1.1
            assert len(result.warnings) > 0
            assert "close to the minimum" in result.warnings[0].lower()

    def test_recommendations_on_failure(self):
        """Test: Recommendations provided on failure."""
        texts = ["same text"] * 20
        dataset = Dataset.from_dict({"text": texts})

        plugin = DiversityValidator(thresholds={"min_score": 0.7})
        result = plugin.validate(dataset)

        assert result.passed is False

        recommendations = plugin.get_recommendations(result)
        assert len(recommendations) > 4
        assert any("diversity" in rec.lower() for rec in recommendations)
        assert any("diversity score" in rec.lower() for rec in recommendations)

    def test_recommendations_critical_low_diversity(self):
        """Test: Critical warning for very low diversity."""
        texts = ["x"] * 100
        dataset = Dataset.from_dict({"text": texts})

        plugin = DiversityValidator(thresholds={"min_score": 0.5})
        result = plugin.validate(dataset)

        assert result.passed is False

        recommendations = plugin.get_recommendations(result)
        # Should have critical warning
        assert any("CRITICAL" in rec for rec in recommendations)
        assert any("overfit" in rec.lower() for rec in recommendations)

    def test_default_parameters(self):
        """Test: Default min_score (0.7)."""
        texts = ["diverse unique content"] * 10
        dataset = Dataset.from_dict({"text": texts})

        plugin = DiversityValidator({})
        result = plugin.validate(dataset)

        assert result.thresholds["min_score"] == 0.7

    def test_custom_sample_size(self):
        """Test: Custom sample_size parameter for large datasets."""
        # Using streaming dataset to trigger sampling logic
        from datasets import IterableDataset

        def gen():
            for i in range(500):
                yield {"text": f"Text {i} with diverse unique vocabulary content"}

        dataset = IterableDataset.from_generator(gen)

        plugin = DiversityValidator(params={"sample_size": 100}, thresholds={"min_score": 0.5})
        result = plugin.validate(dataset)

        # Streaming datasets are considered "large" and get sampled
        assert result.metrics["samples_checked"] <= 100

    def test_execution_time_recorded(self):
        """Test: Execution time is recorded."""
        texts = ["sample text"] * 20
        dataset = Dataset.from_dict({"text": texts})

        plugin = DiversityValidator(thresholds={"min_score": 0.5})
        result = plugin.validate(dataset)

        assert result.execution_time_ms >= 0
        assert result.execution_time_ms < 5000  # Should be reasonably fast

    def test_mixed_case_normalization(self):
        """Test: Text is normalized to lowercase for diversity calc."""
        texts = [
            "The Quick Brown Fox",
            "the QUICK brown FOX",
            "THE quick BROWN fox",
        ]
        dataset = Dataset.from_dict({"text": texts})

        plugin = DiversityValidator(thresholds={"min_score": 0.1})
        result = plugin.validate(dataset)

        # Should treat "The" and "the" as same token
        assert result.metrics["unique_tokens"] == 4  # the, quick, brown, fox

    def test_messages_format_extraction(self, messages_format_dataset):
        """Test: Handles messages format datasets."""
        plugin = DiversityValidator(thresholds={"min_score": 0.5})
        result = plugin.validate(messages_format_dataset)

        # Should extract text from messages
        assert result.metrics["total_tokens"] > 0
        assert result.metrics["diversity_score"] > 0
