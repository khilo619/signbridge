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

## Repository Structure (Monorepo)

```text
signbridge/                         # Root (khilo619/signbridge)
├── apps/
│   └── Web/                        # Next.js video call app
│       ├── pages/
│       │   ├── index.js            # Home: create/join room + role selection
│       │   ├── room/[roomId].js    # Video call room + sign recognition + STT
│       │   └── api/
│       │       ├── pusher/         # auth.js, trigger.js
│       │       └── sign/           # predict.js, refine.js
│       ├── styles/
│       ├── package.json
│       ├── next.config.js
│       ├── tailwind.config.js
│       ├── netlify.toml
│       └── .env.example
│
├── api/                            # FastAPI backend (Python)
│   ├── sign_full/                  # 100-class model API
│   └── sign_demo/                  # 55-class demo API
│
├── CV/                             # Computer Vision module
│   ├── assets/                     # label_mapping.json
│   ├── checkpoints/                # Model weights (.pth)
│   ├── data/                       # Video reader, transforms
│   ├── models/                     # I3D architecture
│   ├── inference/                  # SignRecognizer wrapper
│   ├── training/                   # Training scripts
│   └── scripts/                    # Webcam test, utilities
│
├── notebooks/                      # Jupyter notebooks
│   ├── 01_sign_to_text.ipynb       # Offline sign recognition
│   ├── 01_speech_to_text.ipynb     # Speech recognition (experimental)
│   └── 02_streaming_speech_to_text.ipynb
│
├── experiments/                    # Archived research experiments
├── configs/                        # JSON configs for training
├── tests/                          # Python tests
│
├── requirements.txt                # Full Python environment
├── requirements-api.txt            # Minimal API dependencies
├── pyproject.toml                  # Python project config
├── Dockerfile                      # Container for API
├── setup.sh / setup.bat            # Environment setup scripts
└── README.md                       # This file
```

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

| Dataset | Description |
|---------|-------------|
| [WLASL2000](https://www.kaggle.com/datasets/ngphmng/wlasl2000-dataset) | Word-level ASL videos |
| [ASL Citizen](https://www.kaggle.com/datasets/abd0kamel/asl-citizen) | Crowd-sourced ASL |
| [Google ASL Signs](https://www.kaggle.com/competitions/asl-signs) | Kaggle competition |

---

## 7) Troubleshooting

| Issue | Solution |
|-------|----------|
| **STT not working** | Use Chrome, allow mic permission |
| **WebRTC fails** | Check TURN credentials in `[roomId].js` |
| **LLM refine fails** | Set `OPENAI_API_KEY` or `GEMINI_API_KEY` |
| **Sign recognition slow** | Ensure HuggingFace Space is awake |

---

## 8) Tech Stack

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
