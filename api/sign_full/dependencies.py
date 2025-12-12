"""Dependency injection for the 100-class (full) sign recognition API."""

from functools import lru_cache

from CV.inference.sign_recognizer import SignRecognizer
from .config import CHECKPOINT_PATH, LABEL_MAP_PATH


@lru_cache(maxsize=1)
def get_sign_recognizer() -> SignRecognizer:
    """Return a singleton SignRecognizer for the 100-class model.

    The first call constructs the recognizer and loads the I3D model
    and label mappings. Subsequent calls reuse the same instance.
    """
    return SignRecognizer(
        checkpoint_path=CHECKPOINT_PATH,
        label_map_path=LABEL_MAP_PATH,
    )
