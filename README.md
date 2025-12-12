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
  - The `api.sign_demo` application is an experimental alternative and not the focus of the main deployment.

- **Experimental Notebooks**
  - The repository contains notebooks for sign-to-text and speech-to-text as exploratory work. These are not part of the deployed application.

---

## 4. Models & Checkpoints

### 4.1. I3D Model Management with Git LFS

The primary I3D model (`best_model_citizen100_87pct.pth`) is a large file (~149MB) and is managed using **Git LFS (Large File Storage)**.

- **Tracking:** The `.gitattributes` file tells Git to handle all `*.pth` files in `CV/checkpoints/` with LFS.
- **Automatic Download:** The FastAPI application is **self-healing**. If the model checkpoint is not found when the application starts, the `api.sign_full.dependencies.get_sign_recognizer` function will automatically download it from the Hugging Face Space repository. This ensures the application can always run, even in a fresh environment.

### 4.2. Local Setup

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

## 5. FastAPI Inference Service

The `api/` package exposes the **SignBridge API**, which is containerized using the provided `Dockerfile`.

### 5.1. Running Locally

The deployed application is the `sign_full` API. To run it locally:

```bash
# Install minimal dependencies
pip install -r requirements-api.txt

# Run the server
uvicorn api.sign_full.main:app --reload --host 0.0.0.0 --port 8000
```

Interactive API documentation will be available at `http://localhost:8000/docs`.

### 5.2. Docker

The `Dockerfile` packages the `sign_full` API for deployment. It is configured to use the `api.sign_full.main:app` entrypoint.

```bash
# Build the image
docker build -t signbridge-api .

# Run the container
docker run --rm -p 8000:8000 signbridge-api
```

