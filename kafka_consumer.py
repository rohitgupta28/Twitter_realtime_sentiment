"""
kafka_consumer.py
─────────────────────────────────────────────────────────────
WHY THIS EXISTS:
  This is the PROCESSING HEART of the pipeline.
  It reads tweets from Kafka, runs sentiment analysis,
  extracts hashtags, and saves everything to MongoDB.

  WHY FAUST?
  Faust is Python-native stream processing (like Kafka Streams
  for Java, but in Python). It handles:
  - Consuming from Kafka topics
  - Running async processing (multiple tweets in parallel)
  - Writing results back to Kafka or directly to databases
  - Checkpointing (so it doesn't reprocess tweets on restart)

  HOW TO RUN:
    faust -A kafka_consumer worker -l info

  PARALLEL SCALING:
  Run 4 workers for 4x throughput:
    faust -A kafka_consumer worker -l info &
    faust -A kafka_consumer worker -l info &
    faust -A kafka_consumer worker -l info &
    faust -A kafka_consumer worker -l info &

  FLOW:
    Kafka (tweets-raw)
        ↓  [Faust agent reads messages]
    preprocess tweet
        ↓
    run RoBERTa / VADER
        ↓
    extract hashtags, detect geo
        ↓
    save to MongoDB
        ↓
    update Redis live counters
        ↓
    publish to Kafka (tweets-processed)
        ↓
    FastAPI WebSocket pushes to dashboard
─────────────────────────────────────────────────────────────
"""

import asyncio
import json
from datetime import datetime
from typing import Optional

import faust
from loguru import logger

from config import settings
from mongoconnection import (
    insert_raw_tweet,
    save_processed_tweet,
    ensure_indexes
)
from processing import analyze_text, extract_hashtags, increment_sentiment_counter
from alert_engine import check_and_fire_alert

# ── Faust App Setup ────────────────────────────────────────
# A Faust "app" is like a Flask app — it's the central object.
# It connects to Kafka and manages agents (workers).
app = faust.App(
    id="twitter-sentiment-app",          # unique name for this app
    broker=f"kafka://{settings.kafka_bootstrap_servers}",
    value_serializer="json",             # auto-parse JSON from Kafka
    store="memory://",                   # use in-memory state (fine for our use case)
    # For production, use RocksDB: store="rocksdb://"
)


# ── Faust Record (Schema) ──────────────────────────────────
# Faust Records are like typed dataclasses for Kafka messages.
# They validate that incoming messages have the right structure.
class TweetRecord(faust.Record, serializer="json"):
    """
    Schema for tweets coming from Kafka (tweets-raw topic).
    All fields map to what twitter_producer.py sends.
    """
    tweet_id: Optional[str] = None
    text: str = ""
    author_id: Optional[str] = None
    created_at: Optional[str] = None
    lang: Optional[str] = "en"
    geo: Optional[dict] = None
    source: Optional[str] = None
    matched_keywords: Optional[list] = None
    public_metrics: Optional[dict] = None
    ingested_at: Optional[str] = None
    processed: bool = False


# ── Kafka Topics ───────────────────────────────────────────
# Declare topics so Faust knows about them
raw_topic = app.topic(settings.kafka_topic_raw, value_type=TweetRecord)
processed_topic = app.topic(settings.kafka_topic_processed, value_type=bytes)


# ── Main Processing Agent ──────────────────────────────────
@app.agent(raw_topic, concurrency=4)
async def process_tweet(stream):
    """
    Faust agent = a function that runs forever processing a stream.

    The `async for` loop reads messages from Kafka one by one.
    `concurrency=4` means Faust runs 4 instances of this agent
    in parallel (using asyncio, not threads), giving us 4x throughput.

    WHAT HAPPENS FOR EACH TWEET:
    1. Insert raw tweet to MongoDB (captures it even if NLP fails)
    2. Run sentiment analysis (RoBERTa → VADER fallback)
    3. Extract hashtags
    4. Update MongoDB with results
    5. Increment Redis live counters
    6. Publish processed tweet to tweets-processed topic
    7. Check if negative spike warrants an alert
    """
    async for tweet in stream:
        try:
            await process_single_tweet(tweet)
        except Exception as e:
            logger.error(f"Failed to process tweet {getattr(tweet, 'tweet_id', '?')}: {e}")
            # Don't raise — Faust will move to the next message
            # The failed tweet stays in Kafka for re-processing
            continue


async def process_single_tweet(tweet: TweetRecord):
    """Process one tweet through the full pipeline."""

    # ── Step 1: Insert raw tweet to MongoDB ───────────────
    # We insert BEFORE analysis so even if NLP crashes,
    # the raw tweet is preserved.
    tweet_doc = {
        "tweet_id": tweet.tweet_id,
        "text": tweet.text,
        "author_id": tweet.author_id,
        "created_at": parse_twitter_date(tweet.created_at),
        "lang": tweet.lang,
        "geo": tweet.geo,
        "source": tweet.source,
        "matched_keywords": tweet.matched_keywords or [],
        "public_metrics": tweet.public_metrics or {},
        "processed": False,
    }

    # Run blocking MongoDB insert in executor (keeps async loop responsive)
    doc_id = await asyncio.get_event_loop().run_in_executor(
        None, insert_raw_tweet, tweet_doc
    )

    if not doc_id:
        logger.warning(f"Failed to insert tweet {tweet.tweet_id}")
        return

    # ── Step 2: Run NLP sentiment analysis ────────────────
    # NLP inference is CPU-bound (slow). We run it in a thread pool
    # so it doesn't block the asyncio event loop.
    # The event loop can process other tweets while waiting.
    meta = await asyncio.get_event_loop().run_in_executor(
        None, analyze_text, tweet.text
    )

    sentiment = meta.get("label", "Neutral")
    score = meta.get("score", 0.5)
    model_used = meta.get("model", "unknown")

    # ── Step 3: Extract hashtags ──────────────────────────
    hashtags = extract_hashtags(tweet.text)

    # ── Step 4: Update MongoDB with results ───────────────
    await asyncio.get_event_loop().run_in_executor(
        None,
        save_processed_tweet,
        doc_id, sentiment, score, model_used, hashtags, tweet.geo
    )

    # ── Step 5: Update Redis live counters ────────────────
    # This is instant — just incrementing integers in Redis
    for keyword in (tweet.matched_keywords or []):
        await asyncio.get_event_loop().run_in_executor(
            None, increment_sentiment_counter, sentiment, keyword
        )
    await asyncio.get_event_loop().run_in_executor(
        None, increment_sentiment_counter, sentiment, None
    )

    # ── Step 6: Publish to processed topic ────────────────
    # The FastAPI WebSocket layer listens to this topic
    # and pushes results to connected dashboard clients
    processed_event = {
        "tweet_id": tweet.tweet_id,
        "text": tweet.text[:280],   # cap length for WS payload
        "sentiment": sentiment,
        "score": round(score, 3),
        "hashtags": hashtags,
        "created_at": str(tweet.created_at),
        "author_id": tweet.author_id,
    }
    await processed_topic.send(
        key=tweet.tweet_id,
        value=json.dumps(processed_event).encode("utf-8")
    )

    # ── Step 7: Check for alert conditions ────────────────
    # Fire a Slack/email alert if negative sentiment spikes
    await asyncio.get_event_loop().run_in_executor(
        None, check_and_fire_alert, sentiment, tweet.matched_keywords
    )

    logger.debug(
        f"✅ Processed tweet | {sentiment:8s} ({score:.2f}) | "
        f"#{' #'.join(hashtags[:3])} | {tweet.text[:60]}..."
    )


# ── Faust Table: Sliding Window Counts ────────────────────
# Faust Tables are distributed in-memory key-value stores
# backed by Kafka for fault tolerance.
# We use one to count sentiment per keyword over the last 60 min.

sentiment_window = app.Table(
    "sentiment-counts",
    default=int,
    partitions=4,
).tumbling(
    size=3600.0,    # 1-hour window (seconds)
    expires=7200.0  # keep data for 2 hours
)


@app.agent(processed_topic, sink=[])
async def update_window_counts(stream):
    """
    Maintain real-time sliding window counts in Faust Table.
    These are ultra-fast (in-memory) and survive worker restarts.
    """
    async for event in stream.events():
        try:
            data = json.loads(event.value)
            sentiment = data.get("sentiment", "Neutral")
            key = f"total:{sentiment}"
            sentiment_window[key] += 1
        except Exception:
            continue


# ── Helper ────────────────────────────────────────────────
def parse_twitter_date(date_str: Optional[str]) -> datetime:
    """Parse Twitter's ISO 8601 date string to Python datetime."""
    if not date_str:
        return datetime.utcnow()
    try:
        from dateutil import parser
        return parser.parse(date_str)
    except Exception:
        return datetime.utcnow()


# ── Startup Hook ──────────────────────────────────────────
@app.on_start.connect
async def on_startup(app, **kwargs):
    """Called once when the Faust worker starts up."""
    logger.info("🚀 Faust worker starting...")
    ensure_indexes()
    logger.info(f"📥 Listening on topic: {settings.kafka_topic_raw}")
    logger.info(f"📤 Publishing to topic: {settings.kafka_topic_processed}")


if __name__ == "__main__":
    app.main()
