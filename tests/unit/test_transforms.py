"""Unit tests for CV.data.transforms module."""

import numpy as np
import pytest
import torch

from CV.data.transforms import preprocess_frames, _pad_or_sample_frames


class TestPadOrSampleFrames:
    """Tests for _pad_or_sample_frames function."""

    def test_exact_frame_count(self):
        """When frame count matches target, return unchanged."""
        frames = [np.zeros((224, 224, 3)) for _ in range(32)]
        result = _pad_or_sample_frames(frames, 32)
        assert len(result) == 32

    def test_pad_short_sequence(self):
        """When fewer frames than target, pad with last frame."""
        frames = [np.ones((224, 224, 3)) * i for i in range(10)]
        result = _pad_or_sample_frames(frames, 32)
        
        assert len(result) == 32
        # Last 22 frames should be copies of frame 9
        for i in range(10, 32):
            np.testing.assert_array_equal(result[i], frames[9])

    def test_sample_long_sequence(self):
        """When more frames than target, uniformly sample."""
        frames = [np.ones((224, 224, 3)) * i for i in range(100)]
        result = _pad_or_sample_frames(frames, 32)
        
        assert len(result) == 32
        # First and last should be from original first and last
        np.testing.assert_array_equal(result[0], frames[0])
        np.testing.assert_array_equal(result[-1], frames[-1])

    def test_empty_frames_raises(self):
        """Empty frame list should raise ValueError."""
        with pytest.raises(ValueError, match="No frames provided"):
            _pad_or_sample_frames([], 32)

    def test_single_frame_padding(self):
        """Single frame should be repeated to fill target."""
        frames = [np.ones((224, 224, 3)) * 42]
        result = _pad_or_sample_frames(frames, 32)
        
        assert len(result) == 32
        for frame in result:
            np.testing.assert_array_equal(frame, frames[0])


class TestPreprocessFrames:
    """Tests for preprocess_frames function."""

    def test_output_shape(self, sample_numpy_frames):
        """Output tensor should have shape (1, 3, T, H, W)."""
        frames = [sample_numpy_frames[i] for i in range(32)]
        tensor = preprocess_frames(frames, num_frames=32, image_size=224)
        
        assert tensor.shape == (1, 3, 32, 224, 224)

    def test_output_dtype(self, sample_numpy_frames):
        """Output should be float32."""
        frames = [sample_numpy_frames[i] for i in range(32)]
        tensor = preprocess_frames(frames, num_frames=32, image_size=224)
        
        assert tensor.dtype == torch.float32

    def test_output_range(self, sample_numpy_frames):
        """Output values should be in [0, 1] range."""
        frames = [sample_numpy_frames[i] for i in range(32)]
        tensor = preprocess_frames(frames, num_frames=32, image_size=224)
        
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0

    def test_resize_frames(self):
        """Frames should be resized to target image_size."""
        # Create frames with different size
        frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(16)]
        tensor = preprocess_frames(frames, num_frames=16, image_size=112)
        
        assert tensor.shape == (1, 3, 16, 112, 112)

    def test_custom_num_frames(self):
        """Should handle custom num_frames parameter."""
        frames = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(50)]
        tensor = preprocess_frames(frames, num_frames=16, image_size=224)
        
        assert tensor.shape[2] == 16  # T dimension

    def test_empty_frames_raises(self):
        """Empty frame list should raise ValueError."""
        with pytest.raises(ValueError):
            preprocess_frames([], num_frames=32, image_size=224)
