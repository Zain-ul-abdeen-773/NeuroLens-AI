# 🧠 NeuroLens: AI Behavioral Intelligence Engine

A production-grade system that analyzes human text conversations to detect **deception signals**, **emotional intent**, **manipulation patterns**, and **psychological states** in real-time with full explainability.

![Architecture: DeBERTa + BiLSTM + Dense](https://img.shields.io/badge/Architecture-Hybrid_Transformer-blueviolet)
![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue)
![React 18](https://img.shields.io/badge/React-18-61dafb)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Features

- **Deception Detection** — Probability-based analysis using linguistic cues, hedging patterns, and semantic analysis
- **Multi-Label Emotion Analysis** — 28 emotion categories from the GoEmotions taxonomy
- **Manipulation Detection** — Identifies guilt-tripping, gaslighting, love-bombing, fear-mongering, flattery, and coercion
- **Explainability Layer** — Token importance heatmaps, natural-language reasoning, and feature analysis
- **Real-Time Streaming** — WebSocket-based progressive analysis with live updates
- **Behavioral Profiling** — Session-based personality inference and anomaly detection
- **3D Neural Network Visualization** — Interactive Three.js visualization
- **Cyberpunk UI** — Dark glassmorphism design with neon accents and particle effects

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────┐
│  Input Text                                       │
│     ↓                                            │
│  Preprocessing (clean → normalize → tokenize)     │
│     ↓                                            │
│  ┌──────────┐  ┌────────────────────────────┐    │
│  │ TF-IDF   │  │ DeBERTa Transformer        │    │
│  │ LIWC     │  │     ↓                      │    │
│  │ Sentiment│  │ BiLSTM (sequential)         │    │
│  │ Syntactic│  │     ↓                      │    │
│  └────┬─────┘  │ Attention Pooling           │    │
│       │        └──────────┬─────────────────┘    │
│       └──────────┬────────┘                      │
│                  ↓                               │
│           Fused Representation                    │
│          ↙       ↓         ↘                     │
│   Deception   Emotion   Manipulation             │
│   Head        Head      Head                     │
└──────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+**
- **Node.js 18+**
- **CUDA-capable GPU** (optional, for training)

### 1. Backend Setup

```bash
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000/docs` (Swagger UI).

### 2. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

The UI will be available at `http://localhost:5173`.

### 3. Docker Deployment

```bash
docker-compose up --build
```

- Frontend: `http://localhost`
- Backend API: `http://localhost:8000`

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/analyze` | Analyze single text |
| `POST` | `/api/v1/batch-analyze` | Analyze multiple texts |
| `POST` | `/api/v1/train` | Trigger training run |
| `GET` | `/api/v1/metrics` | Model performance metrics |
| `POST` | `/api/v1/timeline` | Get behavioral timeline |
| `WS` | `/ws/analyze` | Real-time streaming analysis |

### Example Request

```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "I honestly had nothing to do with it, trust me.", "explain": true}'
```

### Example Response

```json
{
  "text": "I honestly had nothing to do with it, trust me.",
  "deception": {
    "probability": 0.73,
    "verdict": "deceptive",
    "confidence": 0.73,
    "reasons": [
      "Presence of classic deception markers ('honestly', 'trust me')",
      "Low sensory detail ratio"
    ]
  },
  "emotions": [
    {"emotion": "nervousness", "probability": 0.61},
    {"emotion": "fear", "probability": 0.42}
  ],
  "manipulation": {
    "detected": null,
    "risk_level": "low"
  },
  "confidence_score": 0.73,
  "processing_time_ms": 89.4
}
```

---

## 🧪 Training

### Using Custom Data

Create a JSON dataset:

```json
[
  {
    "text": "Sample text...",
    "deception": 0,
    "emotions": [0,0,0,1,0,...],
    "manipulation": 0
  }
]
```

### Trigger Training

```bash
curl -X POST http://localhost:8000/api/v1/train \
  -H "Content-Type: application/json" \
  -d '{"dataset_path": "./data/custom.json", "dataset_type": "custom", "epochs": 10}'
```

### Training Features

- Mixed precision (FP16)
- Gradient accumulation & clipping
- Cosine decay with warmup scheduling
- Early stopping with patience
- K-fold cross validation
- Optuna hyperparameter search
- TensorBoard logging

```bash
tensorboard --logdir backend/runs
```

---

## 📁 Project Structure

```
NeuroLens AI/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI entry point
│   │   ├── config.py            # Configuration
│   │   ├── api/
│   │   │   ├── routes.py        # REST endpoints
│   │   │   └── websocket.py     # WebSocket handler
│   │   ├── ml/
│   │   │   ├── model.py         # Hybrid model
│   │   │   ├── trainer.py       # Training pipeline
│   │   │   ├── inference.py     # Inference engine
│   │   │   ├── features.py      # Feature engineering
│   │   │   ├── preprocessing.py # Data preprocessing
│   │   │   └── evaluator.py     # Metrics & explainability
│   │   ├── services/
│   │   │   ├── analyzer.py      # Analysis orchestrator
│   │   │   ├── session.py       # Session manager
│   │   │   └── cache.py         # LRU cache
│   │   └── utils/
│   │       ├── logger.py        # Logging
│   │       └── helpers.py       # Utilities
│   ├── tests/
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── ChatInterface.jsx
│   │   │   ├── AnalysisDashboard.jsx
│   │   │   ├── ExplainabilityPanel.jsx
│   │   │   ├── TimelineView.jsx
│   │   │   ├── NeuralNetworkViz.jsx
│   │   │   ├── ParticleBackground.jsx
│   │   │   ├── RadialProgress.jsx
│   │   │   ├── ConfidenceMeter.jsx
│   │   │   ├── TextHeatmap.jsx
│   │   │   └── WaveformEffect.jsx
│   │   ├── hooks/
│   │   ├── services/
│   │   └── utils/
│   ├── package.json
│   ├── tailwind.config.js
│   └── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 🛡️ Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Framework | PyTorch + HuggingFace Transformers |
| Model | DeBERTa-v3 + BiLSTM + Dense |
| API | FastAPI + Uvicorn |
| Frontend | React 18 + Tailwind CSS + Framer Motion |
| 3D Viz | Three.js / React Three Fiber |
| Training | Mixed Precision, Optuna, TensorBoard |
| Deployment | Docker + Docker Compose |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
