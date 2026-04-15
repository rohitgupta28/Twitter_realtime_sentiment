# 📡 Real-Time Twitter Sentiment Analysis
### Twitter API v2 + Kafka + RoBERTa + MongoDB + FastAPI + Streamlit

A **production-grade real-time sentiment analysis pipeline** that streams
tweets from Twitter/X, analyzes sentiment using transformer-based NLP,
stores everything in MongoDB, and displays live results on an interactive
Streamlit dashboard.

---

## 🏗️ Architecture

```
Twitter API v2 (Filtered Stream)
        ↓
twitter_producer.py  ──→  Kafka (tweets-raw topic)
                                   ↓
                     kafka_consumer.py (Faust worker × 4)
                           ↓          ↓          ↓
                     RoBERTa NLP   MongoDB    Redis counters
                                   ↓          ↓
                              FastAPI (REST + WebSocket)
                                         ↓
                                Streamlit Dashboard
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone <your-repo>
cd realtime_sentiment

# Create virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your Twitter Bearer Token
nano .env
```

Get your Twitter Bearer Token at: https://developer.twitter.com/en/portal/dashboard

### 3. Start Infrastructure (Kafka + Redis)

```bash
# Start all Docker services in background
docker-compose up -d

# Verify everything is running
docker-compose ps

# Open Kafka UI (optional but very useful)
# → http://localhost:8080
```

### 4. Start the Pipeline

Open **4 separate terminal windows**:

**Terminal 1 — Kafka Consumer (NLP Worker)**
```bash
faust -A kafka_consumer worker -l info
```

**Terminal 2 — Twitter Producer (Stream Ingest)**
```bash
python twitter_producer.py
```

**Terminal 3 — FastAPI Backend**
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 4 — Streamlit Dashboard**
```bash
streamlit run app.py
```

Open your browser at: **http://localhost:8501**

---

## 🧪 Testing Without Twitter API

If you don't have a Twitter API key, test with the Sentiment140 dataset:

```bash
# Download from Kaggle (requires free account):
# https://www.kaggle.com/datasets/kazanova/sentiment140

# Ingest 5000 sample tweets with NLP
python ingest.py --csv training.1600000.processed.noemoticon.csv --limit 5000
```

---

## 📁 Project Structure

```
realtime_sentiment/
├── docker-compose.yml      # Kafka + Zookeeper + Redis + Kafka UI
├── .env.example            # Environment variable template
├── requirements.txt        # All Python dependencies
│
├── config.py               # Central config (loads .env)
├── mongoconnection.py      # All MongoDB operations
├── processing.py           # RoBERTa + VADER sentiment analysis
├── topic_modeling.py       # LDA topic modeling
├── alert_engine.py         # Slack + Email alerts
│
├── twitter_producer.py     # Twitter API → Kafka producer
├── kafka_consumer.py       # Kafka → NLP → MongoDB (Faust worker)
├── api.py                  # FastAPI REST + WebSocket server
├── app.py                  # Streamlit dashboard
└── ingest.py               # CSV import for offline testing
```

---

## 🔧 Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `TWITTER_BEARER_TOKEN` | — | **Required.** Your Twitter API Bearer Token |
| `TWITTER_TRACK_KEYWORDS` | `python,AI` | Comma-separated keywords to track |
| `KAFKA_BOOTSTRAP_SERVERS` | `localhost:9092` | Kafka broker address |
| `MONGO_URI` | `mongodb://localhost:27017/` | MongoDB connection string |
| `HF_MODEL` | `cardiffnlp/twitter-roberta-base-sentiment-latest` | HuggingFace model |
| `SLACK_WEBHOOK_URL` | — | Slack webhook for alerts |
| `ALERT_NEGATIVE_THRESHOLD` | `0.65` | Alert when negative% exceeds this |

---

## 📊 Dashboard Features

| Feature | How it works |
|---|---|
| **Live Sentiment Gauge** | Redis counters, updates every 10s |
| **Sentiment Breakdown Pie** | Real-time Plotly chart from Redis |
| **Trending Hashtags** | MongoDB aggregation, last 6 hours |
| **Sentiment Over Time** | Line chart from MongoDB time-buckets |
| **Word Clouds × 3** | Per-sentiment word clouds (WordCloud lib) |
| **Topic Modeling** | LDA via Gensim, cached in Redis |
| **Live Tweet Feed** | Polling MongoDB for latest processed tweets |
| **Download CSV** | Export query results as CSV |

---

## ⚡ Scaling

To process more tweets faster, run multiple Faust workers:

```bash
# Scale to 4 parallel NLP workers
for i in {1..4}; do
  faust -A kafka_consumer worker -l info &
done
```

For cloud deployment:
- MongoDB Atlas (managed MongoDB)
- Confluent Cloud (managed Kafka)  
- Deploy FastAPI + workers to a VPS or Kubernetes

---

## 🎓 Project Info

**Original project:** Twitter Sentiment Analysis (5-7 years ago)  
**Upgraded to:** Real-time streaming pipeline  
**Tech stack:** Python 3.10+, Kafka, Faust, RoBERTa, MongoDB, Redis, FastAPI, Streamlit

Built with ❤️ — Upgraded from batch CSV processing to live real-time streaming.
