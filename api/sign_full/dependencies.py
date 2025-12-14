"""Dependency injection for the 100-class (full) sign recognition API."""

import os
from functools import lru_cache

from huggingface_hub import hf_hub_download

from CV.inference.sign_recognizer import SignRecognizer

from .config import CHECKPOINT_PATH, LABEL_MAP_PATH


@lru_cache(maxsize=1)
def get_sign_recognizer() -> SignRecognizer:
    """Return a singleton SignRecognizer for the 100-class model.

    The first call constructs the recognizer and loads the I3D model
    and label mappings. Subsequent calls reuse the same instance.

    This function is self-healing: if the checkpoint file is not found, it
    will be downloaded from the Hugging Face Hub.
    """
    model_repo_id = os.getenv("MODEL_REPO_ID", "KhaLood619/signbridge-models")
    model_revision = os.getenv("MODEL_REVISION", "v1.0.0")
    checkpoint_filename = os.getenv(
        "MODEL_CHECKPOINT_FILENAME",
        "checkpoints/best_model_citizen100_87pct.pth",
    )
    label_map_filename = os.getenv(
        "MODEL_LABEL_MAP_FILENAME",
        "assets/label_mapping.json",
    )

    checkpoint_path = CHECKPOINT_PATH
    if not os.path.exists(checkpoint_path):
        print(
            f"Checkpoint not found at {checkpoint_path}. "
            f"Downloading {checkpoint_filename} from {model_repo_id}@{model_revision}..."
        )
        checkpoint_path = hf_hub_download(
            repo_id=model_repo_id,
            revision=model_revision,
            filename=checkpoint_filename,
        )
        print("Checkpoint download complete.")

    label_map_path = LABEL_MAP_PATH
    if not os.path.exists(label_map_path):
        print(
            f"Label map not found at {label_map_path}. "
            f"Downloading {label_map_filename} from {model_repo_id}@{model_revision}..."
        )
        label_map_path = hf_hub_download(
            repo_id=model_repo_id,
            revision=model_revision,
            filename=label_map_filename,
        )
        print("Label map download complete.")

    if os.path.getsize(checkpoint_path) < 10 * 1024 * 1024:
        raise RuntimeError(
            f"Checkpoint file looks too small/corrupt: {checkpoint_path} "
            f"({os.path.getsize(checkpoint_path)} bytes)"
        )
    if os.path.getsize(label_map_path) < 100:
        raise RuntimeError(
            f"Label map file looks too small/corrupt: {label_map_path} "
            f"({os.path.getsize(label_map_path)} bytes)"
        )

    return SignRecognizer(
        checkpoint_path=checkpoint_path,
        label_map_path=label_map_path,
    )
