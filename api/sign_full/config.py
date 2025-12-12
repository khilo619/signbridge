"""Configuration for the 100-class (full) sign recognition API."""

import os
from pathlib import Path

# CV directory (relative to repo root)
CV_DIR = Path(__file__).resolve().parents[2] / "CV"
ASSETS_DIR = CV_DIR / "assets"
CHECKPOINTS_DIR = CV_DIR / "checkpoints"

# Model paths for the 100-class model
LABEL_MAP_PATH = os.environ.get(
    "SIGNBRIDGE_FULL_LABEL_MAP",
    str(ASSETS_DIR / "label_mapping.json"),
)

CHECKPOINT_PATH = os.environ.get(
    "SIGNBRIDGE_FULL_CHECKPOINT",
    str(CHECKPOINTS_DIR / "best_model_citizen100_87pct.pth"),
)
