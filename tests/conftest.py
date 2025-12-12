import pytest
import torch
import numpy as np

@pytest.fixture
def sample_video_tensor():
    """
    Creates a random video tensor mimicking the input to the I3D model.
    Shape: (Batch, Channels, Frames, Height, Width) -> (1, 3, 32, 224, 224)
    """
    # Create random float data between 0 and 1
    # Batch size = 1
    return torch.randn(1, 3, 32, 224, 224)

@pytest.fixture
def sample_numpy_frames():
    """
    Creates a random numpy array mimicking raw video frames before preprocessing.
    Shape: (Frames, Height, Width, Channels) -> (32, 224, 224, 3)
    """
    return np.random.randint(0, 255, (32, 224, 224, 3), dtype=np.uint8)

@pytest.fixture
def demo_config():
    """
    Configuration for the 55-class demo model.
    """
    return {
        "num_classes": 55,
        "dropout_prob": 0.5,
        "input_frames": 32
    }
