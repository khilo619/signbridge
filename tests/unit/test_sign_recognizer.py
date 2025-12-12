"""Unit tests for CV.inference.sign_recognizer module."""

import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock

import torch


class TestSignRecognizerPredictClip:
    """Tests for SignRecognizer.predict_clip method (mocked model)."""

    @patch("CV.inference.sign_recognizer.load_model_from_checkpoint")
    def test_predict_clip_returns_prediction_result(self, mock_loader):
        """predict_clip should return a PredictionResult with correct fields."""
        # Setup mock model
        mock_model = Mock()
        mock_model.parameters.return_value = iter([torch.tensor([1.0])])
        mock_model.return_value = torch.tensor([[0.1, 0.2, 0.5, 0.15, 0.05]])  # 5 classes
        
        mock_loader.return_value = (
            mock_model,
            {"hello": 0, "thank_you": 1, "please": 2, "yes": 3, "no": 4},
            {0: "hello", 1: "thank_you", 2: "please", 3: "yes", 4: "no"},
        )
        
        from CV.inference.sign_recognizer import SignRecognizer
        
        recognizer = SignRecognizer()
        frames = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(32)]
        
        result = recognizer.predict_clip(frames, topk=3)
        
        assert result.gloss == "please"  # index 2 has highest prob (0.5)
        assert result.label == 2
        assert abs(result.probability - 0.5) < 0.1  # softmax will change values
        assert len(result.topk_glosses) == 3
        assert len(result.topk_probabilities) == 3

    @patch("CV.inference.sign_recognizer.load_model_from_checkpoint")
    def test_predict_clip_empty_frames_raises(self, mock_loader):
        """predict_clip should raise ValueError for empty frame list."""
        mock_model = Mock()
        mock_model.parameters.return_value = iter([torch.tensor([1.0])])
        
        mock_loader.return_value = (mock_model, {}, {})
        
        from CV.inference.sign_recognizer import SignRecognizer
        
        recognizer = SignRecognizer()
        
        with pytest.raises(ValueError, match="empty frame list"):
            recognizer.predict_clip([])

    @patch("CV.inference.sign_recognizer.load_model_from_checkpoint")
    def test_predict_clip_topk_clamped(self, mock_loader):
        """topk should be clamped to number of classes."""
        mock_model = Mock()
        mock_model.parameters.return_value = iter([torch.tensor([1.0])])
        mock_model.return_value = torch.tensor([[0.3, 0.7]])  # Only 2 classes
        
        mock_loader.return_value = (
            mock_model,
            {"a": 0, "b": 1},
            {0: "a", 1: "b"},
        )
        
        from CV.inference.sign_recognizer import SignRecognizer
        
        recognizer = SignRecognizer()
        frames = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(32)]
        
        # Request topk=10 but only 2 classes exist
        result = recognizer.predict_clip(frames, topk=10)
        
        assert len(result.topk_glosses) == 2
        assert len(result.topk_probabilities) == 2


class TestSignRecognizerInit:
    """Tests for SignRecognizer initialization."""

    @patch("CV.inference.sign_recognizer.load_model_from_checkpoint")
    def test_init_loads_model(self, mock_loader):
        """SignRecognizer should call load_model_from_checkpoint on init."""
        mock_model = Mock()
        mock_model.parameters.return_value = iter([torch.tensor([1.0])])
        mock_loader.return_value = (mock_model, {}, {})
        
        from CV.inference.sign_recognizer import SignRecognizer
        
        recognizer = SignRecognizer(
            checkpoint_path="/custom/path.pth",
            label_map_path="/custom/labels.json",
            device="cpu"
        )
        
        mock_loader.assert_called_once_with(
            checkpoint_path="/custom/path.pth",
            label_map_path="/custom/labels.json",
            device="cpu",
        )

    @patch("CV.inference.sign_recognizer.load_model_from_checkpoint")
    def test_init_sets_device(self, mock_loader):
        """SignRecognizer should detect device from model parameters."""
        mock_model = Mock()
        mock_param = torch.tensor([1.0])
        mock_model.parameters.return_value = iter([mock_param])
        mock_loader.return_value = (mock_model, {}, {})
        
        from CV.inference.sign_recognizer import SignRecognizer
        
        recognizer = SignRecognizer()
        
        assert recognizer.device == mock_param.device
