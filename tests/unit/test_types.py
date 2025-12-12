"""Unit tests for CV.inference.types module."""

import pytest

from CV.inference.types import PredictionResult


class TestPredictionResult:
    """Tests for PredictionResult dataclass."""

    def test_create_prediction_result(self):
        """Should create PredictionResult with all fields."""
        result = PredictionResult(
            gloss="hello",
            label=0,
            probability=0.95,
            topk_glosses=["hello", "hi", "hey"],
            topk_probabilities=[0.95, 0.03, 0.02],
        )
        
        assert result.gloss == "hello"
        assert result.label == 0
        assert result.probability == 0.95
        assert result.topk_glosses == ["hello", "hi", "hey"]
        assert result.topk_probabilities == [0.95, 0.03, 0.02]

    def test_prediction_result_immutable_fields(self):
        """PredictionResult fields should be accessible."""
        result = PredictionResult(
            gloss="thank_you",
            label=5,
            probability=0.88,
            topk_glosses=["thank_you"],
            topk_probabilities=[0.88],
        )
        
        # Access all fields
        assert isinstance(result.gloss, str)
        assert isinstance(result.label, int)
        assert isinstance(result.probability, float)
        assert isinstance(result.topk_glosses, list)
        assert isinstance(result.topk_probabilities, list)

    def test_prediction_result_with_single_prediction(self):
        """Should work with single prediction (topk=1)."""
        result = PredictionResult(
            gloss="yes",
            label=10,
            probability=0.99,
            topk_glosses=["yes"],
            topk_probabilities=[0.99],
        )
        
        assert len(result.topk_glosses) == 1
        assert len(result.topk_probabilities) == 1
