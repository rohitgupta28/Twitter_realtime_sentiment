"""
api.py
─────────────────────────────────────────────────────────────
WHY THIS EXISTS:
  The Streamlit dashboard needs TWO kinds of data:
  1. Historical data  → REST endpoints  (GET /api/tweets, etc.)
  2. Real-time data   → WebSocket       (ws://localhost:8000/ws/live)

  This FastAPI server provides both.

  WHY NOT USE STREAMLIT DIRECTLY FOR KAFKA?
  Streamlit is single-threaded by design. You can't run a Kafka
  consumer inside Streamlit — it would block the UI thread.
  FastAPI + WebSocket is the clean solution:
    Kafka consumer → FastAPI WebSocket server → browser client

  HOW TO RUN:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

  ENDPOINTS:
    GET  /api/tweets              → recent tweets with filters
    GET  /api/sentiment/counts    → live sentiment totals
    GET  /api/trending/hashtags   → top hashtags last 6h
    GET  /api/sentiment/timeseries → time-bucketed counts for chart
    GET  /api/stats               → overall system stats
    WS   /ws/live                 → real-time tweet stream
─────────────────────────────────────────────────────────────
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Optional, List, Set

import redis as redis_lib
from bson import ObjectId
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger

from config import settings
from mongoconnection import (
    ensure_indexes,
    find_by_query_text,
    get_sentiment_counts,
    get_trending_hashtags,
    get_sentiment_over_time,
    get_recent_tweets,
    collection
)
from processing import get_live_counts

# ── App Setup ──────────────────────────────────────────────
app = FastAPI(
    title="Twitter Sentiment API",
    description="Real-time sentiment analysis API with WebSocket support",
    version="2.0.0"
)

# Allow Streamlit (running on port 8501) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Redis client ───────────────────────────────────────────
_redis = redis_lib.Redis(
    host=settings.redis_host,
    port=settings.redis_port,
    db=settings.redis_db,
    decode_responses=True
)


# ── JSON Serialization Helper ──────────────────────────────
def serialize_doc(doc: dict) -> dict:
    """Convert MongoDB document to JSON-serializable dict."""
    if doc is None:
        return {}
    result = {}
    for k, v in doc.items():
        if isinstance(v, ObjectId):
            result[k] = str(v)
        elif isinstance(v, datetime):
            result[k] = v.isoformat()
        elif isinstance(v, dict):
            result[k] = serialize_doc(v)
        else:
            result[k] = v
    return result


# ── Pydantic Response Models ───────────────────────────────
class SentimentCounts(BaseModel):
    Positive: int = 0
    Negative: int = 0
    Neutral: int = 0
    total: int = 0


class TweetResponse(BaseModel):
    tweet_id: Optional[str] = None
    text: str
    sentiment: Optional[str] = None
    score: Optional[float] = None
    created_at: Optional[str] = None
    hashtags: Optional[List[str]] = []


# ── WebSocket Connection Manager ───────────────────────────
class ConnectionManager:
    """
    Manages all active WebSocket connections.

    WHY WE NEED THIS:
    Multiple browser tabs can connect simultaneously.
    When a new tweet arrives, we broadcast to ALL of them.
    We also handle disconnections gracefully.
    """

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: str):
        """Send a message to all connected clients."""
        dead = set()
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                dead.add(connection)
        # Clean up dead connections
        for conn in dead:
            self.active_connections.discard(conn)


manager = ConnectionManager()


# ── Background Kafka Listener ──────────────────────────────
async def kafka_to_websocket():
    """
    Background task: reads from tweets-processed Kafka topic
    and broadcasts each processed tweet to all WebSocket clients.

    This runs as an async task in the FastAPI event loop,
    so it never blocks API requests.
    """
    from kafka import KafkaConsumer

    logger.info("Starting Kafka → WebSocket bridge...")
    consumer = KafkaConsumer(
        settings.kafka_topic_processed,
        bootstrap_servers=settings.kafka_bootstrap_servers,
        group_id="websocket-bridge",
        auto_offset_reset="latest",     # only new messages (don't replay old ones)
        enable_auto_commit=True,
        value_deserializer=lambda m: json.loads(m.decode("utf-8"))
    )

    # Run blocking Kafka poll in a thread so it doesn't block asyncio
    loop = asyncio.get_event_loop()

    while True:
        try:
            # Poll Kafka in thread pool (non-blocking)
            messages = await loop.run_in_executor(
                None,
                lambda: consumer.poll(timeout_ms=100, max_records=20)
            )

            for topic_partition, records in messages.items():
                for record in records:
                    tweet_data = record.value
                    if manager.active_connections:
                        await manager.broadcast(json.dumps(tweet_data))

            await asyncio.sleep(0.05)  # small yield to other tasks

        except Exception as e:
            logger.error(f"Kafka bridge error: {e}")
            await asyncio.sleep(5)  # back off on error


# ── Startup / Shutdown ─────────────────────────────────────
@app.on_event("startup")
async def startup():
    ensure_indexes()
    # Start the Kafka→WebSocket bridge as a background task
    asyncio.create_task(kafka_to_websocket())
    logger.info("✅ FastAPI started. Kafka bridge running.")


# ── REST Endpoints ─────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "Twitter Sentiment API",
        "version": "2.0",
        "endpoints": [
            "/api/tweets", "/api/sentiment/counts",
            "/api/trending/hashtags", "/api/sentiment/timeseries",
            "/api/stats", "/ws/live"
        ]
    }


@app.get("/api/tweets")
def get_tweets(
    keyword: str = Query(default="", description="Search keyword"),
    limit: int = Query(default=50, le=500),
    hours: int = Query(default=24, description="Look back N hours")
):
    """
    Fetch recent tweets matching a keyword.
    Used by the dashboard for historical analysis.
    """
    start = datetime.utcnow() - timedelta(hours=hours)
    docs = find_by_query_text(keyword, limit=limit, start_date=start)
    return [serialize_doc(d) for d in docs]


@app.get("/api/sentiment/counts", response_model=SentimentCounts)
def get_counts(
    keyword: Optional[str] = Query(default=None),
    hours: int = Query(default=24),
    use_redis: bool = Query(default=True,
                            description="Use fast Redis counts vs MongoDB aggregation")
):
    """
    Get sentiment counts.

    use_redis=True  → instant Redis counters (live, approximate)
    use_redis=False → exact MongoDB aggregation (slower, precise)
    """
    if use_redis:
        counts = get_live_counts(keyword)
    else:
        counts = get_sentiment_counts(keyword, hours=hours)

    counts["total"] = sum(counts.values())
    return counts


@app.get("/api/trending/hashtags")
def get_hashtags(
    limit: int = Query(default=20, le=50),
    hours: int = Query(default=6)
):
    """Top hashtags in the last N hours. Sorted by frequency."""
    return get_trending_hashtags(limit=limit, hours=hours)


@app.get("/api/sentiment/timeseries")
def get_timeseries(
    keyword: Optional[str] = Query(default=None),
    hours: int = Query(default=24),
    interval_minutes: int = Query(default=30)
):
    """
    Time-bucketed sentiment counts for line charts.
    Returns data like:
    [{"bucket_ts": 1714000000000, "sentiment": "Positive", "count": 42}, ...]
    """
    raw = get_sentiment_over_time(keyword, hours=hours,
                                   interval_minutes=interval_minutes)
    result = []
    for row in raw:
        result.append({
            "bucket_ts": row["_id"]["bucket"],
            "sentiment": row["_id"]["sentiment"],
            "count": row["count"]
        })
    return result


@app.get("/api/stats")
def get_stats():
    """
    Overall system stats.
    Shown in the dashboard header.
    """
    total = collection.count_documents({})
    processed = collection.count_documents({"processed": True})
    last_24h = collection.count_documents({
        "created_at": {"$gte": datetime.utcnow() - timedelta(hours=24)}
    })

    # Redis pipeline for live counts (atomic multi-read)
    pipe = _redis.pipeline()
    pipe.get("sentiment:total:Positive")
    pipe.get("sentiment:total:Negative")
    pipe.get("sentiment:total:Neutral")
    redis_vals = pipe.execute()

    return {
        "total_tweets": total,
        "processed_tweets": processed,
        "tweets_last_24h": last_24h,
        "live_positive": int(redis_vals[0] or 0),
        "live_negative": int(redis_vals[1] or 0),
        "live_neutral": int(redis_vals[2] or 0),
        "active_websocket_connections": len(manager.active_connections),
        "timestamp": datetime.utcnow().isoformat()
    }


# ── WebSocket Endpoint ─────────────────────────────────────
@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    """
    Real-time tweet stream via WebSocket.

    HOW THE STREAMLIT DASHBOARD USES THIS:
    It connects on load and receives JSON messages like:
    {
        "tweet_id": "...",
        "text": "Just tried the new #GPT4 and it's amazing!",
        "sentiment": "Positive",
        "score": 0.954,
        "hashtags": ["gpt4"],
        "created_at": "2024-05-01T12:34:56"
    }

    The dashboard appends each message to a live feed table
    and updates the sentiment gauge.
    """
    await manager.connect(websocket)

    # Send a welcome message with current stats
    stats = get_stats()
    await websocket.send_text(json.dumps({
        "type": "welcome",
        "data": stats
    }))

    try:
        # Keep connection alive — wait for client messages
        # (client can send {"type": "filter", "keyword": "AI"} to filter)
        while True:
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0   # ping/keep-alive every 30s
                )
                # Echo back any client commands (for future use)
                await websocket.send_text(json.dumps({
                    "type": "ack", "received": data
                }))
            except asyncio.TimeoutError:
                # Send a heartbeat so the browser doesn't close the connection
                await websocket.send_text(json.dumps({"type": "ping"}))

    except WebSocketDisconnect:
        manager.disconnect(websocket)
