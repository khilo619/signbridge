---
title: SignBridge API (100-Class)
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
pinned: true
---
# SignBridge: Multimodal Sign & Speech to Text

A production-ready research project for **word-level sign language recognition**, with experimental **speech-to-text** components for future multimodal fusion.  
The current main deliverable focuses on **offline translation for word-level sign videos** using a high-accuracy **I3D video model** (deployed on Hugging Face) and an optional **FastAPI backend**. Speech-to-text pipelines live in notebooks as exploratory work and are **not yet integrated** into the main application.

This repository contains:

- **Computer Vision (CV) module** for I3D-based sign recognition.
- **FastAPI backend** exposing a clean HTTP API around the model.
- **Jupyter notebooks** for sign-to-text workflows and experimental speech-to-text pipelines.
- **Archived experiments** (real-time prototypes, ISLR landmark models, W&B logs) under `experiments/`.

---

## 1. Features

- **Offline word-level sign recognition**
  - Uses a pre-trained **Inception I3D** model.
  - Trained on a curated 100-class dataset (Citizen + WLASL subset).
  - Top-1 accuracy: **~87.6%** on the final validation set.
  - Packaged for deployment on the **Hugging Face Hub**.

- **Sign-to-text workflows (notebooks)**
  - `notebooks/01_sign_to_text.ipynb` – offline translation from sign video to text.

- **Speech-to-text workflows (experimental notebooks)**
  - `notebooks/01_speech_to_text.ipynb` – basic speech recognition pipeline (prototype, **not used by the main app yet**).
  - `notebooks/02_streaming_speech_to_text.ipynb` – streaming speech recognition prototype (for future integration).

- **FastAPI inference service**
  - FastAPI-based HTTP API for sign recognition using the I3D model.
  - Easy to integrate with frontends or additional services.

- **Well-structured experiments**
  - `experiments/notebooks/` – real-time sign prototypes, ISLR LSTM experiments, Colab helper notebooks.
  - `experiments/wandb/` – historical Weights & Biases logs (not required for running the app).

---

## 2. Repository Structure

A high-level view of the most relevant files and directories:

```text
NHA-057/
├── api/                      # FastAPI backend (SignBridge API)
│   ├── common/               # Shared utilities (health, schemas, video_io)
│   ├── sign_full/            # 100-class model API (main.py, routers.py, config.py)
│   └── sign_demo/            # 55-class demo model API (main.py, routers.py, config.py)
│
├── CV/                       # Computer Vision module (I3D sign model)
│   ├── config.py             # Central config (paths, device, num_classes, ...)
│   ├── assets/               # Label mapping, config assets (e.g., label_mapping.json)
│   ├── checkpoints/          # Model checkpoints (.pth)
│   ├── data/                 # Video reader & transforms
│   ├── models/               # I3D and model loading utilities
│   ├── inference/            # High-level SignRecognizer wrapper
│   ├── training/             # Datasets and training scripts for I3D
│   └── scripts/              # Utility scripts (e.g., webcam test)
│
├── notebooks/                # Main project notebooks (final workflows)
│   ├── 01_sign_to_text.ipynb
│   ├── 01_speech_to_text.ipynb
│   └── 02_streaming_speech_to_text.ipynb
│
├── experiments/              # Archived / research experiments (not required for core app)
│   ├── notebooks/            # Real-time sign + ISLR training notebooks, Colab notebooks
│   └── wandb/                # Weights & Biases logs (ignored in typical deployments)
│
├── configs/                  # JSON configs for data/training (if needed)
│   ├── data_config.json
│   └── train_config.json
│
├── requirements.txt          # Full development environment
├── requirements-api.txt      # Minimal dependencies for the FastAPI inference service
├── Dockerfile                # Containerization of the inference stack
├── setup.sh / setup.bat      # Helper setup scripts
└── README.md                 # This file
```

---

## 3. Installation

### 3.1. Prerequisites

- **Python** ≥ 3.9
- Recommended OS: Linux or Windows with a recent GPU driver (CPU also works, but slower).
- (Optional) **CUDA-capable GPU** for faster video inference.

### 3.2. Clone the repository

```bash
git clone <YOUR_REPO_URL> NHA-057
cd NHA-057
```

### 3.3. Install dependencies

You can choose between the **full development environment** or the **minimal API environment**.

#### Option A – Full environment (notebooks + training utilities + API)

```bash
pip install -r requirements.txt
```

This installs:

- Core scientific stack (NumPy, Pandas, SciPy, etc.)
- PyTorch, TensorFlow, Transformers
- OpenCV, MediaPipe, and other CV utilities
- Whisper, Vosk, and audio dependencies for speech processing
- FastAPI + Uvicorn
- Testing and misc utilities

#### Option B – Minimal API environment

If you only want to run the **FastAPI I3D inference service**:

```bash
pip install -r requirements-api.txt
```

This installs only what is needed for:

- PyTorch I3D inference
- Basic image/video handling
- FastAPI + Uvicorn

---

## 4. Models & Checkpoints

### 4.1. I3D model and label mapping

The CV module expects a **pre-trained I3D** checkpoint and a **label mapping JSON**.

Default locations (see `CV/config.py`):

- **Label map**: `CV/assets/label_mapping.json`
- **Checkpoint**: `CV/checkpoints/best_model_citizen100_87pct.pth`

You can override these paths using environment variables:

```bash
export SIGNBRIDGE_LABEL_MAP=/path/to/label_mapping.json
export SIGNBRIDGE_CHECKPOINT=/path/to/checkpoint.pth
```

The mapping file must contain:

```json
{
  "gloss_to_label": {"HELLO": 0, "THANK_YOU": 1, "...": 99},
  "label_to_gloss": {"0": "HELLO", "1": "THANK_YOU", "99": "..."},
  "num_classes": 100
}
```

> **Note:** The repository focuses on **inference** using an already-trained model (e.g., exported and uploaded to the Hugging Face Hub). Training code and historical experiments are kept separately in notebooks and the `experiments/` folder.

### 4.2. Device configuration

`CV/config.py` automatically selects `cuda` if a GPU is available:

```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

No additional configuration is required for basic use; this can be overridden via code if needed.

---

## 5. Using the Notebooks

### 5.1. Offline sign-to-text (`01_sign_to_text.ipynb`)

This notebook demonstrates **offline word-level sign recognition** from video using the I3D model.

Typical workflow:

1. Launch Jupyter:
   ```bash
   jupyter notebook
   ```
2. Open `notebooks/01_sign_to_text.ipynb`.
3. Ensure the I3D checkpoint and label mapping are placed as described in [Models & Checkpoints](#4-models--checkpoints).
4. Follow the notebook cells to:
   - Load or record a sign video clip.
   - Preprocess frames and feed them to the I3D model via the `SignRecognizer` wrapper.
   - Display top-1 or top-k predicted glosses and probabilities.

Under the hood, the notebook uses:

- `CV.data.video_reader` – to load frames from video.
- `CV.data.transforms` – to resize/crop/normalize frames to the expected I3D input.
- `CV.inference.sign_recognizer.SignRecognizer` – to run the model and decode predictions.

### 5.2. Speech-to-text (`01_speech_to_text.ipynb`)

This notebook provides an **experimental** demonstration of basic speech recognition using one or more of the installed speech libraries (e.g., Whisper, Vosk). It is intended for future multimodal extensions and is **not part of the current core deliverable**.

Typical workflow:

1. Open `notebooks/01_speech_to_text.ipynb`.
2. Select an audio file or microphone input (depending on the cells).
3. Run cells to:
   - Capture / load audio.
   - Run the chosen STT model.
   - Display the transcribed text.

### 5.3. Streaming speech-to-text (`02_streaming_speech_to_text.ipynb`)

This notebook explores **streaming or near-real-time speech recognition**, using an audio stream and incremental decoding.

It serves as a **research prototype** and reference implementation for building more advanced speech-based interfaces in the future and is **not currently integrated into the main application**.

> **Note:** The exact behavior (Whisper vs. Vosk, streaming strategy, etc.) depends on how you configure/install the relevant backends in the notebook.

---

## 6. FastAPI Inference Service

The `api/` package exposes a **FastAPI app** named **SignBridge API** that wraps the I3D sign recognizer.

### 6.1. Start the API server

From the project root, you can run either the **100-class full model** or the **55-class demo model**:

```bash
# 100-class full model (default)
uvicorn api.sign_full.main:app --reload --host 0.0.0.0 --port 8000

# 55-class demo model
uvicorn api.sign_demo.main:app --reload --host 0.0.0.0 --port 8001
```

- The root endpoint returns a basic health message.
- Interactive API docs are available at: `http://localhost:8000/docs`.

### 6.2. API overview

The API is organized into two subpackages under `api/`:

- **`api.sign_full`** – 100-class model (87.6% accuracy)
- **`api.sign_demo`** – 55-class conversational demo model (91.7% accuracy)

Each subpackage has its own `main.py` entrypoint:

```python
# api/sign_full/main.py
from fastapi import FastAPI
from api.common import health
from .routers import router as sign_router

app = FastAPI(title="SignBridge API (100-class Full Model)", version="1.0.0")

@app.get("/")
def read_root():
    return {"message": "SignBridge API is running", "docs_url": "/docs"}

app.include_router(health.router)
app.include_router(sign.router)
```

Typical routes:

- `GET /` – Basic status.
- `GET /health` – Health check endpoint.
- `POST /sign/...` – Endpoints (defined in `api/routers/sign.py`) for sending video clips or frame sequences and receiving predicted gloss/labels.

> **Implementation note:** The sign endpoints typically:
> - Accept video or pre-extracted frames.
> - Use `CV.inference.sign_recognizer.SignRecognizer` to run inference.
> - Return top-1 and top-k predictions with probabilities.

### 6.3. Running behind Docker (optional)

A `Dockerfile` is provided to containerize the FastAPI service.

Basic usage (example, adjust to your environment):

```bash
# Build image
docker build -t signbridge-api .

# Run container (CPU example)
docker run --rm -p 8000:8000 \
  -e SIGNBRIDGE_LABEL_MAP=/app/CV/assets/label_mapping.json \
  -e SIGNBRIDGE_CHECKPOINT=/app/CV/checkpoints/best_model_citizen100_87pct.pth \
  signbridge-api
```

Then access the service at `http://localhost:8000`.

---

## 7. CV Module (I3D) Internals

The CV module is designed to be **modular and reusable**:

- `CV/models/i3d.py` – Inception I3D architecture definition.
- `CV/models/loader.py` – High-level functions to:
  - Load label mappings from JSON.
  - Create an I3D model with the right number of classes.
  - Load checkpoints and handle `DataParallel` prefixes.
- `CV/data/transforms.py` – Frame preprocessing for I3D (resize, normalize, etc.).
- `CV/data/video_reader.py` – Utilities to decode video files into frame sequences.
- `CV/inference/sign_recognizer.py` – A convenient wrapper:

  ```python
  from CV.inference import SignRecognizer

  recognizer = SignRecognizer()
  result = recognizer.predict_clip(frames, topk=5)

  print(result.gloss, result.probability)
  print(result.topk_glosses, result.topk_probabilities)
  ```

- `CV/training/datasets.py` – Generic CSV/manifest-based video dataset class for training.
- `CV/training/train_i3d.py` – CLI training script for (re)training or fine-tuning the I3D model.
- `CV/scripts/test_webcam.py` – Example script to test live webcam sign capture (if configured).

To train or fine-tune the I3D model on your own manifests, you can run, for example:

```bash
python -m CV.training.train_i3d \
  --train-manifest /path/to/train_manifest.csv \
  --val-manifest /path/to/val_manifest.csv \
  --base-dir /path/to/videos_root \
  --label-map CV/assets/label_mapping.json \
  --epochs 50 \
  --batch-size 8
```

The manifest CSVs must contain at least two columns:

- `video_path` – path to each video file (relative to `base-dir` or absolute).
- `label` – integer class id in `[0, num_classes-1]` consistent with the label mapping.

This encapsulation makes it easy to:

- Swap out checkpoints.
- Change label mappings.
- Integrate into new APIs or UIs.

---

## 8. Experiments & Research

All experimental and research-oriented materials are collected under `experiments/` to keep the main application clean.

### 8.1. Notebooks

`experiments/notebooks/` includes, for example:

- Real-time sign-to-text prototypes using webcam and streaming (experimental, **not part of the current offline deliverable**).
- ISLR (isolated sign language recognition) training notebooks using landmarks and BiLSTMs.
- Colab-specific notebooks used during development.

These are **not required** for running the current offline word-level I3D app, but are valuable for understanding the project’s evolution and for future research.

### 8.2. Weights & Biases logs

`experiments/wandb/` contains historical **Weights & Biases** runs and logs.

- This folder is versioned in this repository to expose full training curves, hyperparameters, and run metadata. Be aware it can be large; for very lightweight deployments you may choose to exclude it.
- A key W&B run for the final 100-class I3D model (~87.6% top-1) is available at:  
  https://wandb.ai/Sign_Bridge/Sign_Bridge/runs/bj3s5cle

---

## 9. Flutter client applications

A dedicated Flutter team has developed full mobile/desktop client applications that consume the SignBridge backend:

- The Flutter apps provide the main user-facing UI.
- They communicate with the FastAPI service and/or the Hugging Face-deployed I3D model.
- This repository focuses on the backend, CV pipeline, and notebooks; the Flutter code lives in a separate repository.

---

## 10. Extending the Project

Some ideas for future work and extensions:

- **Multimodal fusion**
  - Combine video-based sign recognition with audio-based speech recognition for robust multi-user interaction, for example via late fusion of I3D logits and speech model outputs.

- **Real-time sign recognition**
  - Explore low-latency pipelines using pose extraction models such as **MediaPipe Holistic** or **OpenPose** to obtain 2D/3D keypoints, then feed them to lightweight sequence models (e.g., BiLSTMs, TGCN/ST-GCN) or fuse pose features with I3D features for real-time feedback.

- **Structured training pipeline**
  - Further extend the existing `CV/training/` package with richer configs, advanced augmentation, and experiment management so others can easily reproduce, fine-tune, or compare models.

- **More languages and domains**
  - Extend datasets and label mappings to additional sign languages or domain-specific vocabularies.

---

## 11. Datasets

This project builds on publicly available datasets hosted on Kaggle:

- **WLASL2000** – word-level American Sign Language videos  
  Kaggle: https://www.kaggle.com/datasets/ngphmng/wlasl2000-dataset

- **ASL Citizen** – crowd-sourced ASL signing videos  
  Kaggle: https://www.kaggle.com/datasets/abd0kamel/asl-citizen

- **Google - Isolated Sign Language Recognition (ASL Signs)** – Kaggle competition  
  Competition page: https://www.kaggle.com/competitions/asl-signs

> Depending on the experiment, we use subsets or combinations of these datasets (e.g., a 100-class subset of Citizen + WLASL for the final I3D model, and the ASL Signs competition landmarks for separate ISLR experiments).

---

## 12. Acknowledgements

- The I3D architecture and many design choices are inspired by existing **sign language recognition research** and open-source implementations.
- This project builds on widely-used open-source libraries: **PyTorch**, **FastAPI**, **OpenCV**, **MediaPipe**, and others listed in `requirements.txt`.

If you use or extend this project in academic work, please consider citing the relevant underlying datasets and models (e.g., WLASL and other sign language resources) according to their licenses.
