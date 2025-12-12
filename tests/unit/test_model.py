import pytest
import torch

from CV.models.i3d import InceptionI3d


def test_i3d_model_instantiation(demo_config):
    """
    Test that we can create the model with the correct number of classes.
    """
    model = InceptionI3d(num_classes=demo_config["num_classes"], in_channels=3)
    assert model.num_classes == 55
    # Verify the logits layer (the classifier) has the correct output size
    # InceptionI3d logits are a Conv3d layer
    assert model.logits.conv3d.out_channels == 55

def test_i3d_forward_pass(sample_video_tensor, demo_config):
    """
    Test that the model accepts a tensor and produces an output.
    This does NOT check for accuracy, only for 'plumbing' (shapes/types).
    """
    model = InceptionI3d(num_classes=demo_config["num_classes"], in_channels=3)
    model.eval() # Set to eval mode to disable dropout behavior for deterministic check
    
    with torch.no_grad():
        output = model(sample_video_tensor)
    
    # Expected output shape: (Batch_Size, Num_Classes)
    assert output.shape == (1, demo_config["num_classes"])
    
    # Check that output is a valid tensor (not NaN)
    assert not torch.isnan(output).any()

@pytest.mark.slow
def test_i3d_dropout_config(demo_config):
    """
    Test that dropout is correctly configured in the model.
    """
    dropout_prob = 0.5
    model = InceptionI3d(
        num_classes=demo_config["num_classes"], 
        in_channels=3,
        dropout_keep_prob=1.0 - dropout_prob
    )
    
    # Check if dropout module exists and has correct probability
    # Note: InceptionI3d implements dropout as nn.Dropout(p = 1 - keep_prob)
    assert isinstance(model.dropout, torch.nn.Dropout)
    assert abs(model.dropout.p - dropout_prob) < 1e-5
