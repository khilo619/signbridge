# SignBridge

A **real-time sign language video call platform** that bridges communication between deaf/hard-of-hearing users and hearing users.

---

## Features

| Feature | Description |
|---------|-------------|
| **Video Call (WebRTC)** | Peer-to-peer video/audio with TURN relay fallback |
| **Sign Language Recognition** | I3D model (100 classes, ~87% accuracy) via HuggingFace |
| **Gloss → Sentence Refinement** | LLM (OpenAI/Gemini) converts glosses to proper English |
| **Speech-to-Text** | Browser Web Speech API for hearing users → text for deaf users |
| **Real-time Sync** | Pusher Channels for signaling + translation events |

---

## Overview & Goals

- **G1 – Build a strong word-level sign recognizer** (ASL, 100 classes) → **~87.6% top-1** with Inception I3D on Citizen + WLASL100.
- **G2 – Deliver a deployable app**: FastAPI backend + Docker + HF Space for the model; Next.js frontend for calls.
- **G3 – Clean API surface**: HTTP endpoints for other clients (e.g., Flutter) and web API routes in the Next.js app.
- **G4 – Explore extensions**: Speech-to-text (browser), gloss-to-sentence LLM refinement, and real-time prototypes (archived).
- **G5 – Reproducible pipeline**: Training code in `CV/training`, checkpoints in LFS, experiments and W&B logs retained.

Status: G1–G5 achieved for the 100-class offline pipeline; real-time and multimodal remain experimental.

---

## Repository Structure (Monorepo)

```text
signbridge/ (root)
├── .github/workflows/              # CI/CD
│   ├── deploy.yml                  # Sync to Hugging Face
│   └── test.yml                    # Pytest on push
│
├── apps/
│   └── Web/                        # Next.js web app
│       ├── pages/                  # Next.js pages + API routes
│       │   ├── index.js
│       │   ├── room/[roomId].js
│       │   └── api/
│       │       ├── pusher/         # auth.js, trigger.js
│       │       └── sign/           # predict.js, refine.js
│       ├── styles/                 # globals.css
│       ├── package.json
│       ├── package-lock.json
│       ├── next.config.js
│       ├── tailwind.config.js
│       ├── netlify.toml
│       ├── jsconfig.json
│       └── .env.example
│
├── api/                            # FastAPI backend
│   ├── common/                     # health, schemas, video_io
│   ├── sign_full/                  # 100-class API (main, routers, config, deps)
│   └── sign_demo/                  # 55-class demo API
│
├── CV/                             # Computer Vision module
│   ├── assets/                     # label mappings
│   ├── checkpoints/                # model weights (.pth) via LFS
│   ├── data/                       # video reader, transforms
│   ├── models/                     # I3D architecture
│   ├── inference/                  # SignRecognizer wrapper
│   ├── training/                   # training scripts
│   └── scripts/                    # webcam test, utilities
│
├── notebooks/                      # Primary notebooks
│   ├── 01_sign_to_text.ipynb
│   ├── 01_speech_to_text.ipynb
│   ├── 02_conversational_demo_seed.ipynb
│   ├── 02_streaming_speech_to_text.ipynb
│   └── 05_msasl_downloader.ipynb
│
├── experiments/                    # Archived research
│   ├── notebooks/                  # real-time / ISLR experiments
│   └── wandb/                      # W&B logs
│
├── tests/                          # Python tests
│   ├── integration/                # test_api_demo.py
│   └── unit/                       # test_model.py, test_sign_recognizer.py, test_transforms.py, test_types.py
│
├── configs/                        # JSON configs
│   ├── data_config.json
│   └── train_config.json
│
├── docs/                           # Project docs (MD/PDF/Tex)
├── manifests/                      # (empty placeholder)
├── requirements.txt                # Full Python env
├── requirements-api.txt            # Minimal API deps
├── pyproject.toml                  # Python project config
├── Dockerfile                      # API container
├── setup.sh / setup.bat            # Environment setup
└── README.md                       # This file
```

---

## Installation (Python)

Choose a full environment (notebooks + training + API) or the minimal API stack.

```bash
# Clone
git clone https://github.com/khilo619/signbridge.git
cd signbridge

# Python env
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Option A: full stack (notebooks + training + API)
pip install -r requirements.txt

# Option B: minimal API only
pip install -r requirements-api.txt
```

Notes:
- Python 3.10+ recommended.
- Git LFS is required to pull the checkpoint: `git lfs install && git lfs pull`.

---

## 1) Web App (Next.js)

### 1.1 Requirements
- **Node.js** 18+
- **Pusher** account (free tier works)
- **Metered TURN** credentials (recommended for cross-network calls)

### 1.2 Setup

```bash
cd signbridge/apps/Web
npm install
```

### 1.3 Environment Variables

Copy `.env.example` to `.env.local`:

```env
# Pusher (required)
PUSHER_APP_ID=
PUSHER_KEY=
PUSHER_SECRET=
PUSHER_CLUSTER=
NEXT_PUBLIC_PUSHER_KEY=
NEXT_PUBLIC_PUSHER_CLUSTER=

# LLM for sentence refinement (optional)
OPENAI_API_KEY=
# or
GEMINI_API_KEY=
```

### 1.4 Run

```bash
npm run dev
```

Open: `http://localhost:3000`

### 1.5 How It Works

```
┌─────────────┐                              ┌─────────────┐
│  Deaf User  │◄────── WebRTC Video ────────►│Hearing User │
│    🤟       │                              │     👂      │
└──────┬──────┘                              └──────┬──────┘
       │                                            │
       │ Signs → 32-frame clip                      │ Speech
       ▼                                            ▼
┌─────────────┐                              ┌─────────────┐
│ HuggingFace │                              │ Web Speech  │
│  I3D Model  │                              │    API      │
└──────┬──────┘                              └──────┬──────┘
       │ Gloss                                      │ Text
       ▼                                            │
┌─────────────┐                                     │
│  LLM Refine │ (OpenAI/Gemini)                     │
└──────┬──────┘                                     │
       │ Sentence                                   │
       └──────────────► Pusher ◄────────────────────┘
                           │
                    Both users see text
```

---

## 2) Python API (FastAPI)

### 2.1 Requirements
- **Python** 3.10+
- **PyTorch** (GPU recommended)

### 2.2 Setup (API-only)

```bash
cd signbridge
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements-api.txt
```

### 2.3 Run

```bash
# 100-class model
uvicorn api.sign_full.main:app --reload --host 0.0.0.0 --port 8000

# or 55-class demo
uvicorn api.sign_demo.main:app --reload --host 0.0.0.0 --port 8001
```

Open: `http://localhost:8000/docs`

### 2.4 Docker

```bash
docker build -t signbridge-api .
docker run --rm -p 8000:8000 signbridge-api
```

---

## 3) CV Module (I3D Sign Recognition)

### 3.1 Model Info
| Property | Value |
|----------|-------|
| Architecture | Inception I3D |
| Classes | 100 (Citizen + WLASL subset) |
| Top-1 Accuracy | ~87.6% |
| Input | 32 frames @ 25fps |

### 3.2 Assets & Checkpoints
- **Label map**: `CV/assets/label_mapping.json`
- **Checkpoint**: `CV/checkpoints/best_model_citizen100_87pct.pth` (Git LFS)
- **Config**: `CV/config.py` (paths, num_classes=100, frames=32, image_size=224)

### 3.3 Training (reproducible)
- Data: Citizen + WLASL100 cleaned/filtered manifests (see datasets below).
- Preprocessing: 32 frames, 224×224 RGB, uniform sampling; augmentation mirrors notebook (flip, temporal crop/resample, brightness/contrast, small rotations, noise).
- Scripts: `CV/training/train_i3d.py` + `CV/training/datasets.py` for CLI training with JSON manifests.
- Tracking: W&B run for final model (`bj3s5cle`) with ~87.6% top-1.

### 3.2 Usage

```python
from CV.inference import SignRecognizer

recognizer = SignRecognizer()
result = recognizer.predict_clip(frames, topk=5)

print(result.gloss, result.probability)
```

### 3.3 Assets
- **Label mapping**: `CV/assets/label_mapping.json`
- **Checkpoint**: `CV/checkpoints/best_model_citizen100_87pct.pth`

---

## 4) Notebooks

| Notebook | Purpose |
|----------|---------|
| `01_sign_to_text.ipynb` | Offline sign video → text |
| `01_speech_to_text.ipynb` | Speech recognition (experimental) |
| `02_streaming_speech_to_text.ipynb` | Streaming STT prototype |
| `02_conversational_demo_seed.ipynb` | Conversational demo seed |
| `05_msasl_downloader.ipynb` | MS-ASL dataset helper |

---

## 5) Deployment

### 5.1 Web App (Vercel/Netlify)

For **Vercel**:
- Set **Root Directory** = `apps/Web`
- Add environment variables in project settings

For **Netlify**:
- `netlify.toml` is in `apps/Web`
- Set **Base directory** = `apps/Web`

### 5.2 Python API (HuggingFace Spaces / Docker)

The I3D model is deployed on HuggingFace:
`https://khalood619-signbridge-api.hf.space`

---

## 6) Datasets

| Dataset | Modality / size | Role |
|---------|-----------------|------|
| [WLASL2000 → WLASL100](https://www.kaggle.com/datasets/ngphmng/wlasl2000-dataset) | RGB video, 100-gloss subset (~2038 videos, 1013 available) | Core training data for the 100-class I3D |
| [ASL Citizen](https://www.kaggle.com/datasets/abd0kamel/asl-citizen) | RGB video, crowdsourced 100-gloss overlap | Augments WLASL100 for the final 100-class dataset |
| [Google ASL Signs](https://www.kaggle.com/competitions/asl-signs) | Landmark sequences (pose/hands/face) | ISLR/landmark experiments (archived) |

---

## 7) Project Scope & Limitations
- Vocabulary: 100 word-level glosses (ASL); not full sentence translation.
- Modality: RGB I3D pipeline is production; speech/real-time are experimental.
- Language: ASL-focused; other sign languages not covered yet.
- Real-time: Prototypes exist in `experiments/`, not productionized.
- Dataset bias: Trained on Citizen + WLASL; may not cover all dialects/demographics.

---

## 8) Experiments & Research

- `experiments/notebooks/`: real-time prototypes, ISLR (landmark) runs, Colab helpers.
- `experiments/wandb/`: tracked training runs (final 100-class I3D ~87.6%: https://wandb.ai/Sign_Bridge/Sign_Bridge/runs/bj3s5cle).
- Landmark-based models (ASL Signs) and Hyso/TGCN explorations are archived; not production.

---

## 9) Future Work (from original plan)
- Multimodal fusion: combine I3D outputs with speech (late fusion or cross-modal models).
- Landmark model based sign recognition: lightweight pose/landmark pipelines (MediaPipe/OpenPose + BiLSTM/ST-GCN/TGCN).
- Larger vocabularies and more languages: extend beyond 100 ASL glosses.
- Training framework: richer config-driven experiments, more augmentation/ablation support in `CV/training`.

---

## 10) Acknowledgements
- Built on PyTorch, FastAPI, OpenCV, MediaPipe, and related open-source libraries.
- Datasets: WLASL, ASL Citizen, Google ASL Signs (Kaggle).
- Thanks to the broader sign language research community for architectures and baselines that informed the I3D setup.

---

## 11) Troubleshooting

| Issue | Solution |
|-------|----------|
| **STT not working** | Use Chrome, allow mic permission |
| **WebRTC fails** | Check TURN credentials in `[roomId].js` |
| **LLM refine fails** | Set `OPENAI_API_KEY` or `GEMINI_API_KEY` |
| **Sign recognition slow** | Ensure HuggingFace Space is awake |

---

## 12) Tech Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | Next.js 14, React, Tailwind CSS |
| **Real-time** | WebRTC, Pusher Channels |
| **Sign Model** | PyTorch I3D, HuggingFace Spaces |
| **LLM** | OpenAI GPT-3.5 / Google Gemini |
| **STT** | Web Speech API (browser) |
| **Backend** | FastAPI, Uvicorn |
| **Deploy** | Vercel, Netlify, Docker |

---

## License

See individual dataset licenses for training data. Code is provided as-is for educational purposes.
