# 🏦 FedFraud v2.0 — Full-Stack Federated Fraud Detection

> **Production-style federated learning system with FastAPI backend, React dashboard, WebSocket streaming, and MongoDB persistence.**

---

## 📐 Architecture

```
┌─────────────────────────────────────────────────────────┐
│              React Dashboard (Vite + Recharts)           │
│   Control Panel · KPI Cards · Live Charts · WS Client   │
└──────────────────────┬──────────────────────────────────┘
                       │  REST + WebSocket
┌──────────────────────▼──────────────────────────────────┐
│                  FastAPI Backend                         │
│  POST /start-training  GET /status  GET /history  /ws   │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│               FL Service Layer (fl_service.py)           │
│  TrainingConfig · run_training() · ExperimentState       │
└────────────┬─────────────────────────┬──────────────────┘
             │                         │
┌────────────▼──────────┐  ┌───────────▼──────────────────┐
│   FederatedServer     │  │  BankClient × 5              │
│  FedAvg/FedProx/Adam  │  │  LocalFraudModel (NumPy MLP) │
│  DP noise injection   │  │  FedProx · EarlyStopping     │
└───────────────────────┘  └──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                  MongoDB (optional)                      │
│        experiments · rounds · client_metrics             │
└─────────────────────────────────────────────────────────┘
```

---

## 🗂️ Project Structure

```
fedfraud/
├── backend/
│   ├── main.py                  # FastAPI app + WebSocket manager
│   ├── requirements.txt
│   ├── .env.example
│   ├── api/
│   │   └── routes.py            # REST endpoints
│   ├── services/
│   │   ├── fl_service.py        # ← Orchestration layer (NEW)
│   │   ├── fl_server.py         # FedAvg / FedProx / FedAdam aggregation
│   │   ├── fl_client.py         # Bank client — local training
│   │   └── generator.py         # Non-IID data generator
│   ├── models/
│   │   └── local_model.py       # NumPy MLP (BatchNorm + Dropout + Adam)
│   └── database/
│       └── mongo.py             # Async MongoDB helpers (motor)
│
└── frontend/
    ├── index.html
    ├── package.json
    ├── vite.config.js
    ├── .env.example
    └── src/
        ├── main.jsx             # React entry point
        └── App.jsx              # Full dashboard (all components)
```

---

## ⚡ Quick Start

### 1 — Clone / set up files

Place the project in a folder, then:

```bash
cd fedfraud
```

---

### 2 — Backend Setup

```bash
cd backend

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env if needed (MongoDB URL, port)

# Start the API server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The backend will be live at:
- **API docs**: http://localhost:8000/docs
- **Health check**: http://localhost:8000/health
- **WebSocket**: ws://localhost:8000/ws

> **MongoDB is optional.** If not running, the server logs a warning and operates fully in-memory.

---

### 3 — Frontend Setup

```bash
cd frontend

# Install Node dependencies
npm install

# Configure environment
cp .env.example .env.local
# VITE_API_URL=http://localhost:8000
# VITE_WS_URL=ws://localhost:8000/ws

# Start dev server
npm run dev
```

Dashboard available at: **http://localhost:3000**

---

### 4 — MongoDB Setup (optional)

**Using Docker (easiest):**
```bash
docker run -d --name fedfraud-mongo -p 27017:27017 mongo:7
```

**Or locally:**
```bash
# macOS
brew install mongodb-community && brew services start mongodb-community

# Ubuntu
sudo apt install mongodb && sudo systemctl start mongodb
```

Set in `backend/.env`:
```
MONGO_URL=mongodb://localhost:27017
MONGO_DB=fedfraud
```

---

## 🔧 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/start-training` | Start FL training (background task) |
| `GET`  | `/api/status` | Training status + progress |
| `GET`  | `/api/metrics` | Latest round metrics |
| `GET`  | `/api/history` | Full round-by-round history |
| `GET`  | `/api/bank-profiles` | Participating bank profiles |
| `POST` | `/api/reset` | Clear experiment state |
| `WS`   | `/ws` | Real-time training stream |
| `GET`  | `/health` | Health check |

### POST /api/start-training

```json
{
  "num_rounds": 15,
  "local_epochs": 5,
  "learning_rate": 0.001,
  "batch_size": 64,
  "mu": 0.01,
  "strategy": "fedavg",
  "dp_noise_multiplier": 0.0,
  "dp_max_grad_norm": 1.0,
  "dropout_rate": 0.3,
  "seed": 42
}
```

### WebSocket Message Types

```jsonc
// Server → Client on connect
{ "type": "init", "state": { ... } }

// Server → Client each round
{ "type": "round_update", "data": { "round": 3, "f1": 0.82, "auc": 0.93, ... }, "state": { ... } }

// Server → Client on completion
{ "type": "training_complete", "data": { "final": { ... }, "baseline": { ... } } }

// Server → Client on error
{ "type": "error", "message": "..." }

// Client → Server (keep-alive)
"ping"
```

---

## 🏦 Bank Profiles

| Bank | Transactions | Fraud Rate | Geography | Channels |
|------|-------------|-----------|-----------|---------|
| NationalBank | 10,000 | 2.5% | Urban | POS, ATM |
| DigitalFirst | 8,000 | 3.5% | International | Online |
| RuralCoopBank | 5,000 | 1.0% | Rural | POS, ATM |
| PremiumWealth | 3,000 | 1.5% | Urban | Wire, Online |
| FinTechNeo | 7,000 | 4.5% | International | Online |

Non-IID by design — each bank has distinct transaction distributions, fraud rates, and channel patterns.

---

## 📊 FL Strategies

| Strategy | Description | Best For |
|----------|-------------|---------|
| `fedavg` | Weighted average of client weights | Homogeneous data |
| `fedprox` | FedAvg + proximal regularization `(μ/2)‖w−w₀‖²` | Non-IID / heterogeneous |
| `fedadam` | Server-side Adam on pseudo-gradients | Fastest convergence |

---

## 🔐 Privacy — Differential Privacy

When `dp_noise_multiplier > 0`:
1. Each client's weight tensor is **clipped** to `dp_max_grad_norm`
2. **Calibrated Gaussian noise** `N(0, σ²)` is added where `σ = dp_noise_multiplier × dp_max_grad_norm`
3. Higher noise = stronger privacy guarantee (lower ε) but reduced accuracy

**Tradeoff table:**

| DP Noise | Privacy (ε) | Expected F1 drop |
|----------|------------|-----------------|
| 0.0 | None | — |
| 0.1 | Moderate | ~3–5% |
| 0.5 | Strong | ~8–12% |
| 1.0 | Very strong | ~15–20% |

---

## 🚀 Production Deployment

### Docker Compose

```yaml
version: "3.9"
services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - MONGO_URL=mongodb://mongo:27017
    depends_on:
      - mongo
    command: uvicorn main:app --host 0.0.0.0 --port 8000

  frontend:
    build: ./frontend
    ports:
      - "3000:80"
    environment:
      - VITE_API_URL=http://backend:8000
      - VITE_WS_URL=ws://backend:8000/ws

  mongo:
    image: mongo:7
    volumes:
      - mongo_data:/data/db

volumes:
  mongo_data:
```

```bash
docker-compose up --build
```

---

## 📈 Expected Results

| Strategy | Rounds | F1 | AUC |
|----------|--------|----|-----|
| FedAvg | 15 | ~0.83 | ~0.92 |
| FedProx | 15 | ~0.85 | ~0.93 |
| FedAdam | 15 | ~0.87 | ~0.94 |
| FedAvg + DP (0.1) | 15 | ~0.78 | ~0.89 |

---

## 🧩 Extending the System

- **Add a new aggregation strategy**: Subclass logic in `services/fl_server.py` and add the enum variant
- **Add a new bank**: Append a `BankProfile` to `PREDEFINED_BANKS` in `services/generator.py`
- **Real data**: Replace `TransactionDataGenerator` in `fl_service.py` with a CSV/DB loader
- **Secure Aggregation**: Add SMPC / homomorphic encryption layer before `_fedavg()` in `fl_server.py`
- **Async Celery**: Replace `BackgroundTasks` in `routes.py` with a Celery worker + Redis broker

---

**Developed by Prachi Shende** · [GitHub](https://github.com/prachishende007)
