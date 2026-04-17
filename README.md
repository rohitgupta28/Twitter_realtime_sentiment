# 📡 Realtime Sentiment Dashboard

A real-time Twitter/X sentiment analysis system that processes tweets using **RoBERTa** (HuggingFace) and **VADER**, stores results in **MongoDB**, streams data via **Kafka**, and displays everything on a live **Streamlit** dashboard.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![MongoDB](https://img.shields.io/badge/MongoDB-7.0-brightgreen)
![Kafka](https://img.shields.io/badge/Kafka-7.5-black)

---

## 🚀 Features

- **Real-time sentiment analysis** using RoBERTa (transformer model) with VADER fallback
- **Live dashboard** with auto-refreshing charts and tweet feed
- **Kafka streaming** pipeline for processing live Twitter data
- **MongoDB storage** with full-text search and indexed queries
- **Redis caching** for fast repeated lookups
- **Alert engine** — Slack and email alerts when negative sentiment spikes
- **Topic modeling** using LDA (Gensim) to find trending topics
- **FastAPI backend** with REST + WebSocket endpoints
- **Sentiment140 CSV support** — run without Twitter API using 1.6M tweet dataset

---

## 🏗️ Architecture

```
Twitter API → Kafka (tweets-raw) → Kafka Consumer → NLP (RoBERTa/VADER)
                                                          ↓
CSV File → ingest.py ────────────────────────────→ MongoDB
                                                          ↓
                                              FastAPI (REST + WebSocket)
                                                          ↓
                                              Streamlit Dashboard (localhost:8501)
```

---

## 📁 Project Structure

```
realtime_sentiment/
├── app.py                  # Streamlit dashboard (main UI)
├── api.py                  # FastAPI REST + WebSocket server
├── ingest.py               # Sentiment140 CSV → MongoDB pipeline
├── processing.py           # RoBERTa + VADER sentiment analysis
├── mongoconnection.py      # All MongoDB operations
├── kafka_consumer.py       # Faust stream processor
├── twitter_producer.py     # Twitter API → Kafka producer
├── alert_engine.py         # Slack/email alert system
├── topic_modeling.py       # LDA topic modeling (Gensim)
├── config.py               # Pydantic settings manager
├── docker-compose.yml      # Kafka + Zookeeper + Redis + Kafka UI
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variables template
└── README.md               # This file
```

---

## ⚙️ Prerequisites

- Python 3.10+
- MongoDB (local install or MongoDB Atlas free tier)
- Docker Desktop (for Kafka + Redis)
- Git

---

## 🛠️ Installation & Setup

### Step 1 — Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/realtime-sentiment.git
cd realtime-sentiment
```

### Step 2 — Create environment file
```bash
# Windows
copy .env.example .env

# Mac/Linux
cp .env.example .env
```
Edit `.env` and fill in your values (MongoDB URI, Twitter token if using live stream).

### Step 3 — Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### Step 4 — Install dependencies
```bash
pip install -r requirements.txt
pip install pydantic-settings
```

### Step 5 — Start infrastructure (Kafka + Redis)
```bash
docker compose up -d
```
Verify Kafka UI at: http://localhost:8080

---

## 🚦 Running the Project

You need **3 terminals** running simultaneously:

**Terminal 1 — Start MongoDB:**
```bash
mongod
```

**Terminal 2 — Start FastAPI backend:**
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 3 — Start Streamlit dashboard:**
```bash
streamlit run app.py
```

Open your browser at: **http://localhost:8501**

---

## 📊 Loading Data (CSV Mode — No Twitter API needed)

Download the Sentiment140 dataset from [Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140) and place it in the project folder.

```bash
# Quick test — 2000 tweets with NLP (~10 min on CPU)
python ingest.py --csv training.1600000.processed.noemoticon.csv --limit 2000

# Medium dataset — 10000 tweets (good for demo)
python ingest.py --csv training.1600000.processed.noemoticon.csv --limit 10000

# Fast raw insert — no NLP (very fast)
python ingest.py --csv training.1600000.processed.noemoticon.csv --limit 50000 --no-nlp
```

---

## 🐦 Live Twitter Streaming (Optional)

Requires a Twitter Developer Account and Bearer Token.

```bash
# Terminal 4 — Stream live tweets to Kafka
python twitter_producer.py

# Terminal 5 — Process Kafka stream → MongoDB
python kafka_consumer.py
```

---

## 🔧 Environment Variables

Copy `.env.example` to `.env` and configure:

| Variable | Description | Default |
|----------|-------------|---------|
| `TWITTER_BEARER_TOKEN` | Twitter API v2 Bearer Token | — |
| `TWITTER_TRACK_KEYWORDS` | Keywords to track | `python,AI,machinelearning` |
| `MONGO_URI` | MongoDB connection string | `mongodb://localhost:27017/` |
| `DB_NAME` | Database name | `twitter_db` |
| `KAFKA_BOOTSTRAP_SERVERS` | Kafka broker address | `localhost:9092` |
| `REDIS_HOST` | Redis host | `localhost` |
| `SLACK_WEBHOOK_URL` | Slack webhook for alerts | — |
| `ALERT_NEGATIVE_THRESHOLD` | Negative sentiment alert threshold | `0.65` |

---

## 📈 Dashboard Features

| Tab | What it shows |
|-----|--------------|
| Overview | Live sentiment gauge, pie chart, trend line |
| Live Feed | Real-time tweet stream with sentiment labels |
| Historical | Time-series analysis, hashtag trends |
| Topics | LDA topic modeling results |

---

## 🧠 NLP Models

| Model | Type | Speed | Accuracy |
|-------|------|-------|----------|
| `cardiffnlp/twitter-roberta-base-sentiment-latest` | Transformer (Primary) | ~4 tweets/sec CPU | High |
| VADER | Rule-based (Fallback) | ~200 tweets/sec | Medium |

The system automatically falls back to VADER if the transformer model fails or is unavailable.

---

## 🐳 Docker Services

| Service | Port | Purpose |
|---------|------|---------|
| Zookeeper | 2181 | Kafka coordinator |
| Kafka | 9092 | Message broker |
| Kafka UI | 8080 | Web UI to monitor topics |
| Redis | 6379 | Cache + live counters |

---

## 📦 Tech Stack

| Layer | Technology |
|-------|-----------|
| Dashboard | Streamlit, Plotly, WordCloud |
| API | FastAPI, Uvicorn, WebSockets |
| NLP | HuggingFace Transformers, VADER, Gensim |
| Streaming | Apache Kafka, Faust |
| Database | MongoDB, Redis |
| Twitter | Tweepy v4 |
| Config | Pydantic Settings |

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

---

## 📄 License

MIT License — feel free to use this project for learning and personal use.

---

## 🙏 Acknowledgements

- [Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140) by Stanford
- [Cardiff NLP](https://github.com/cardiffnlp) for the Twitter RoBERTa model
- [HuggingFace](https://huggingface.co) for the Transformers library
