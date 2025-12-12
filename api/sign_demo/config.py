"""Configuration for the 55-class (demo) sign recognition API."""

import os
from pathlib import Path

# CV directory (relative to repo root)
CV_DIR = Path(__file__).resolve().parents[2] / "CV"
ASSETS_DIR = CV_DIR / "assets"
CHECKPOINTS_DIR = CV_DIR / "checkpoints"

# Model paths for the 55-class demo model
LABEL_MAP_PATH = os.environ.get(
    "SIGNBRIDGE_DEMO_LABEL_MAP",
    str(ASSETS_DIR / "label_mapping_demo.json"),
)

CHECKPOINT_PATH = os.environ.get(
    "SIGNBRIDGE_DEMO_CHECKPOINT",
    str(CHECKPOINTS_DIR / "demo_model_55class.pth"),
)
