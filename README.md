---
title: SignBridge API (100-Class)
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
pinned: true
---
# SignBridge: Multimodal Sign & Speech to Text

[![Hugging Face Deployment](https://github.com/khilo619/signbridge/actions/workflows/deploy.yml/badge.svg)](https://github.com/khilo619/signbridge/actions/workflows/deploy.yml)

A production-ready research project for **word-level sign language recognition**, with experimental **speech-to-text** components for future multimodal fusion.  
The current main deliverable focuses on **offline translation for word-level sign videos** using a high-accuracy **I3D video model** (deployed on Hugging Face) and an optional **FastAPI backend**. Speech-to-text pipelines live in notebooks as exploratory work and are **not yet integrated** into the main application.

This repository contains:

- **Computer Vision (CV) module** for I3D-based sign recognition.
- **FastAPI backend** exposing a clean HTTP API around the model.
- **Jupyter notebooks** for sign-to-text workflows and experimental speech-to-text pipelines.
- **Archived experiments** (real-time prototypes, ISLR landmark models, W&B logs) under `experiments/`.

---

## 1. Deployment

This project is configured for continuous deployment to **Hugging Face Spaces**.

- **Live Application**: [https://huggingface.co/spaces/KhaLood619/signbridge-api](https://huggingface.co/spaces/KhaLood619/signbridge-api)

### CI/CD Pipeline

This repository uses **GitHub Actions** to automate testing and deployment:

- **Continuous Integration (CI)** (`test.yml`): On every push to `main`, a workflow runs `pytest` to ensure that no code changes have introduced regressions. 
- **Continuous Deployment (CD)** (`deploy.yml`): After the CI checks pass, a second workflow automatically syncs the `main` branch to the Hugging Face Space, deploying the latest version of the application.

This means any commit pushed to the `main` branch of the `origin` remote will be live within minutes.

---

## 2. Development Workflow

To support a stable production deployment, this project uses two primary Git remotes:

- **`origin`**: The main GitHub repository (`khilo619/signbridge`). Pushing to this remote's `main` branch **triggers the live deployment**.
- **`nhahub`**: A secondary repository (`nhahub/NHA-057`) used for development, collaboration, and backup. Pushing to this remote **does not** trigger any automated actions.

**Recommended workflow:**
1. Work on features in local branches.
2. Push feature branches to the `nhahub` remote for collaboration and review.
3. Once a feature is complete and tested, merge it into your local `main` branch.
4. Push the `main` branch to `origin` to deploy it.

---

## 3. Features

- **Offline word-level sign recognition**
  - Uses a pre-trained **Inception I3D** model.
  - Trained on a curated 100-class dataset (Citizen + WLASL subset).
  - Top-1 accuracy: **~87.6%** on the final validation set.
  - Deployed on the **Hugging Face Hub** via a self-contained Docker environment.

- **FastAPI Inference Service**
  - The primary deliverable is the `api.sign_full` application, which serves the 87.6% accuracy model.

- **Experimental Notebooks**
  - The repository contains notebooks for sign-to-text and speech-to-text as exploratory work. These are not part of the deployed application.

---

## 4. Repository Structure

A high-level view of the most relevant files and directories:

```text
NHA-057/
├── .github/workflows/        # CI/CD pipeline definitions
│   ├── test.yml              # Runs pytest on every push
│   └── deploy.yml            # Syncs code to Hugging Face Spaces
│
├── api/                      # FastAPI backend (SignBridge API)
│   ├── common/               # Shared utilities
│   │   ├── health.py         # Health check endpoint
│   │   ├── schemas.py        # Pydantic models
│   │   └── video_io.py       # Video I/O utilities
│   └── sign_full/            # 100-class model API
│       ├── main.py           # FastAPI app entry point
│       ├── routers.py        # API routes
│       ├── config.py         # Configuration
│       └── dependencies.py  # Dependency injection
│
├── CV/                       # Computer Vision module (I3D sign model)
│   ├── config.py             # Central config (paths, device, num_classes, ...)
│   ├── assets/               # Label mapping, config assets
│   ├── checkpoints/          # Model checkpoints (.pth) - managed by Git LFS
│   ├── data/                 # Video reader & transforms
│   ├── models/               # I3D and model loading utilities
│   ├── inference/            # High-level SignRecognizer wrapper
│   ├── training/             # Datasets and training scripts for I3D
│   └── scripts/              # Utility scripts (webcam test, etc.)
│
├── notebooks/                # Main project notebooks
│   ├── 01_sign_to_text.ipynb
│   ├── 01_speech_to_text.ipynb
│   ├── 02_conversational_demo_seed.ipynb
│   ├── 02_streaming_speech_to_text.ipynb
│   └── 05_msasl_downloader.ipynb
│
├── experiments/              # Archived experiments
│   ├── notebooks/            # Research notebooks
│   │   ├── 02_real-time_sign_to_text.ipynb
│   │   ├── 02_real-time_sign_to_text_clean.ipynb
│   │   ├── 02_real-time_sign_to_text_islr_combined.ipynb
│   │   ├── 03_islr200_training_final.ipynb
│   │   ├── 04_islr200_ultimate_training.ipynb
│   │   └── colab_inference.ipynb
│   └── wandb/                # Weights & Biases logs
│
├── tests/                    # Unit and integration tests
│   ├── conftest.py           # Pytest fixtures
│   ├── integration/           # Integration tests
│   │   └── test_api_demo.py
│   └── unit/                 # Unit tests
│       └── test_model.py
│
├── configs/                  # JSON configs for data/training
│   ├── data_config.json
│   └── train_config.json
│
├── .gitattributes            # Git LFS tracking rules
├── requirements.txt          # Full development environment
├── requirements-api.txt      # Minimal API dependencies
├── Dockerfile                # Containerization
├── setup.sh                  # Setup script (Linux)
└── setup.bat                 # Setup script (Windows)
```

---

## 5. Installation

### 5.1. Prerequisites

- **Python** >= 3.9
- Recommended OS: Linux or Windows with a recent GPU driver (CPU also works, but slower).
- (Optional) **CUDA-capable GPU** for faster video inference.
- **Git LFS** for downloading the model checkpoint.

### 5.2. Clone the repository

```bash
git clone https://github.com/khilo619/signbridge.git NHA-057
cd NHA-057

# Pull the model file from LFS
git lfs pull
```

### 5.3. Install dependencies

You can choose between the **full development environment** or the **minimal API environment**.

#### Option A - Full environment (notebooks + training utilities + API)

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

#### Option B - Minimal API environment

If you only want to run the **FastAPI I3D inference service**:

```bash
pip install -r requirements-api.txt
```

This installs only what is needed for:

- PyTorch I3D inference
- Basic image/video handling
- FastAPI + Uvicorn

---

## 6. Models & Checkpoints

### 6.1. I3D Model Management with Git LFS

The primary I3D model (`best_model_citizen100_87pct.pth`) is a large file (~149MB) and is managed using **Git LFS (Large File Storage)**.

- **Tracking:** The `.gitattributes` file tells Git to handle all `*.pth` files in `CV/checkpoints/` with LFS.
- **Automatic Download:** The FastAPI application is **self-healing**. If the model checkpoint is not found when the application starts, the `api.sign_full.dependencies.get_sign_recognizer` function will automatically download it from the Hugging Face Space repository. This ensures the application can always run, even in a fresh environment.

### 6.2. Local Setup

To work with the model locally, you must have Git LFS installed.

```bash
# Install Git LFS (once per machine)
winget install --id Git.GitLFS

# Set up LFS in the repository (once per clone)
git lfs install

# Download the model file
git lfs pull
```

---

## 7. FastAPI Inference Service

The `api/` package exposes the **SignBridge API**, which is containerized using the provided `Dockerfile`.

### 7.1. Running Locally

The deployed application is the `sign_full` API. To run it locally:

```bash
# Install minimal dependencies
pip install -r requirements-api.txt

# Run the server
uvicorn api.sign_full.main:app --reload --host 0.0.0.0 --port 8000
```

Interactive API documentation will be available at `http://localhost:8000/docs`.

### 7.2. Docker

The `Dockerfile` packages the `sign_full` API for deployment. It is configured to use the `api.sign_full.main:app` entrypoint.

```bash
# Build the image
docker build -t signbridge-api .

# Run the container
docker run --rm -p 7860:7860 signbridge-api
```

---

## 8. Software Development & Automation

This project follows modern software development practices to ensure code quality, reliability, and ease of deployment.

- **Automated Testing (CI):** The repository is configured with a Continuous Integration pipeline that runs a full suite of unit and integration tests (`pytest`) on every commit. This acts as a safety net, preventing regressions and ensuring that new features do not break existing functionality.

- **Automated Deployment (CD):** Upon successful completion of the test suite, a Continuous Deployment pipeline automatically syncs the application to Hugging Face Spaces. This GitOps-based approach means the deployed application is always a direct reflection of the tested and verified `main` branch.

- **Infrastructure as Code:** The entire application environment is defined in the `Dockerfile`, ensuring that the development, testing, and production environments are consistent and reproducible.

This level of automation minimizes manual errors, accelerates the development cycle, and provides confidence in the stability of the live application.

---

## 9. Datasets

This project builds on publicly available datasets hosted on Kaggle:

- **MS-ASL** - A large-scale American Sign Language dataset.
  Kaggle: https://www.kaggle.com/datasets/nadayoussefamrawy/ms-asl
