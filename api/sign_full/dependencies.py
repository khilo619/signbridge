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
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Checkpoint not found at {CHECKPOINT_PATH}, downloading from Hub...")
        hf_hub_download(
            repo_id="KhaLood619/signbridge-api",
            filename=os.path.basename(CHECKPOINT_PATH),
            local_dir=os.path.dirname(CHECKPOINT_PATH),
            repo_type="space",
        )
        print("Download complete.")

    return SignRecognizer(
        checkpoint_path=CHECKPOINT_PATH,
        label_map_path=LABEL_MAP_PATH,
    )
