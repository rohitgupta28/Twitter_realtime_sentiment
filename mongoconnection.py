"""
mongoconnection.py
─────────────────────────────────────────────────────────────
WHY THIS EXISTS:
  All MongoDB operations live here in one place.
  The rest of the app never imports pymongo directly —
  they just call these functions.

  KEY UPGRADE from your old version:
  • Added save_processed_tweet() for the Faust consumer
  • Added get_recent_tweets() for the FastAPI/WebSocket layer
  • Added get_sentiment_counts() for real-time dashboard stats
  • Added get_trending_hashtags() for the trend tracker
  • All indexes created once at startup via ensure_indexes()
─────────────────────────────────────────────────────────────
"""

from pymongo import MongoClient, TEXT, ASCENDING, DESCENDING
from pymongo.errors import BulkWriteError
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from loguru import logger
from config import settings


# ── Connection (module-level singleton) ────────────────────
# MongoClient is thread-safe. Creating it once at module import
# is the right pattern — don't create a new client per request.
client = MongoClient(settings.mongo_uri)
db = client[settings.db_name]
collection = db[settings.col_name]


# ── Index Setup ────────────────────────────────────────────
def ensure_indexes():
    """
    Create all MongoDB indexes once at startup.
    Indexes make queries fast. Without them, MongoDB
    does a full collection scan (very slow on millions of tweets).

    WHAT EACH INDEX DOES:
    • text index  → powers keyword search ($text queries)
    • created_at  → fast time-range filtering
    • sentiment   → fast sentiment aggregation queries
    • geo         → powers geo-radius queries (2dsphere)
    • hashtags    → fast hashtag filtering
    """
    try:
        # Full-text search on tweet content
        collection.create_index(
            [("text", TEXT)],
            name="text_idx",
            default_language="english"
        )
        # Time-based queries (most common: "tweets from last 24h")
        collection.create_index(
            [("created_at", DESCENDING)],
            name="created_at_idx"
        )
        # Sentiment filtering
        collection.create_index(
            [("sentiment", ASCENDING)],
            name="sentiment_idx"
        )
        # Compound index for dashboard: sentiment over time per keyword
        collection.create_index(
            [("created_at", DESCENDING), ("sentiment", ASCENDING)],
            name="time_sentiment_idx"
        )
        # Geospatial (for the geo heatmap)
        collection.create_index(
            [("geo.coordinates", "2dsphere")],
            name="geo_idx",
            sparse=True   # sparse = only index docs that HAVE the geo field
        )
        # Hashtag lookup
        collection.create_index(
            [("hashtags", ASCENDING)],
            name="hashtags_idx",
            sparse=True
        )
        logger.info("✅ MongoDB indexes verified")
    except Exception as e:
        logger.warning(f"Index creation warning (may already exist): {e}")


# ── Write Operations ───────────────────────────────────────
def insert_raw_tweet(tweet: Dict[str, Any]) -> Optional[str]:
    """
    Insert a single raw tweet from the Twitter stream.
    Returns the MongoDB _id as a string.

    Called by: twitter_producer.py (before Kafka)
               OR kafka_consumer.py (after receiving from Kafka)
    """
    tweet.setdefault("ingested_at", datetime.utcnow())
    tweet.setdefault("sentiment", None)
    tweet.setdefault("analysis_meta", None)
    result = collection.insert_one(tweet)
    return str(result.inserted_id)


def insert_many(records: List[Dict[str, Any]]) -> int:
    """Bulk insert — used by ingest.py for CSV import."""
    for r in records:
        r.setdefault("ingested_at", datetime.utcnow())
    try:
        result = collection.insert_many(records, ordered=False)
        return len(result.inserted_ids)
    except BulkWriteError as e:
        logger.warning(f"Bulk write partial error: {e.details}")
        return e.details.get("nInserted", 0)


def save_processed_tweet(tweet_id: str, sentiment: str, score: float,
                          model: str, hashtags: List[str],
                          geo: Optional[Dict] = None) -> bool:
    """
    Called by the Faust consumer AFTER NLP analysis.
    Updates the existing MongoDB document with:
      - sentiment label  ("Positive" / "Negative" / "Neutral")
      - confidence score
      - model name used
      - extracted hashtags
      - geo data (if available)

    Uses $set so only these fields are updated — the original
    tweet text and metadata are preserved.
    """
    from bson import ObjectId
    update = {
        "sentiment": sentiment,
        "analysis_meta": {
            "label": sentiment,
            "score": round(score, 4),
            "model": model,
            "model_version": settings.model_version,
            "analyzed_at": datetime.utcnow()
        },
        "hashtags": hashtags,
        "processed": True,
        "processed_at": datetime.utcnow()
    }
    if geo:
        update["geo"] = geo

    result = collection.update_one(
        {"_id": ObjectId(tweet_id)},
        {"$set": update}
    )
    return result.modified_count > 0


# ── Read Operations ────────────────────────────────────────
def find_by_query_text(
    query: str,
    limit: int = 500,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> List[Dict]:
    """
    Full-text search across tweet content.
    Used by the Streamlit dashboard for keyword analysis.
    """
    q: Dict[str, Any] = {}
    if query:
        q["$text"] = {"$search": query}
    if start_date or end_date:
        q["created_at"] = {}
        if start_date:
            q["created_at"]["$gte"] = start_date
        if end_date:
            q["created_at"]["$lte"] = end_date

    projection = {
        "text": 1, "created_at": 1, "user": 1,
        "sentiment": 1, "hashtags": 1, "geo": 1,
        "analysis_meta": 1
    }
    return list(collection.find(q, projection).limit(limit))


def get_recent_tweets(limit: int = 50, since_minutes: int = 5) -> List[Dict]:
    """
    Fetch the most recent N processed tweets.
    Used by FastAPI WebSocket to push live updates to the dashboard.
    """
    cutoff = datetime.utcnow() - timedelta(minutes=since_minutes)
    return list(
        collection.find(
            {"processed": True, "created_at": {"$gte": cutoff}},
            {"text": 1, "sentiment": 1, "user": 1, "created_at": 1, "hashtags": 1}
        )
        .sort("created_at", DESCENDING)
        .limit(limit)
    )


def get_sentiment_counts(
    keyword: Optional[str] = None,
    hours: int = 24
) -> Dict[str, int]:
    """
    Aggregate sentiment counts for the live gauge on the dashboard.
    Returns: {"Positive": 342, "Negative": 178, "Neutral": 91}
    """
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    match: Dict[str, Any] = {
        "processed": True,
        "created_at": {"$gte": cutoff}
    }
    if keyword:
        match["$text"] = {"$search": keyword}

    pipeline = [
        {"$match": match},
        {"$group": {"_id": "$sentiment", "count": {"$sum": 1}}}
    ]
    result = collection.aggregate(pipeline)
    counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
    for doc in result:
        if doc["_id"] in counts:
            counts[doc["_id"]] = doc["count"]
    return counts


def get_trending_hashtags(limit: int = 20, hours: int = 6) -> List[Dict]:
    """
    Find the most-used hashtags in the last N hours.
    Used by the dashboard trend tracker.
    Returns: [{"tag": "AI", "count": 512}, ...]
    """
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    pipeline = [
        {"$match": {"processed": True, "created_at": {"$gte": cutoff}}},
        {"$unwind": "$hashtags"},
        {"$group": {"_id": "$hashtags", "count": {"$sum": 1}}},
        {"$sort": {"count": DESCENDING}},
        {"$limit": limit},
        {"$project": {"tag": "$_id", "count": 1, "_id": 0}}
    ]
    return list(collection.aggregate(pipeline))


def get_sentiment_over_time(
    keyword: Optional[str] = None,
    hours: int = 24,
    interval_minutes: int = 30
) -> List[Dict]:
    """
    Time-bucketed sentiment counts.
    Used for the line chart on the dashboard.
    Returns hourly/30-min buckets of Positive/Negative/Neutral counts.
    """
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    match: Dict[str, Any] = {"processed": True, "created_at": {"$gte": cutoff}}
    if keyword:
        match["$text"] = {"$search": keyword}

    interval_ms = interval_minutes * 60 * 1000
    pipeline = [
        {"$match": match},
        {"$group": {
            "_id": {
                "bucket": {
                    "$subtract": [
                        {"$toLong": "$created_at"},
                        {"$mod": [{"$toLong": "$created_at"}, interval_ms]}
                    ]
                },
                "sentiment": "$sentiment"
            },
            "count": {"$sum": 1}
        }},
        {"$sort": {"_id.bucket": ASCENDING}}
    ]
    return list(collection.aggregate(pipeline))


def aggregate_sentiment_for_query(
    query: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> List[Dict]:
    """Legacy function — kept for backwards compatibility with old app.py."""
    match: Dict[str, Any] = {}
    if query:
        match["$text"] = {"$search": query}
    if start_date or end_date:
        match["created_at"] = {}
        if start_date:
            match["created_at"]["$gte"] = start_date
        if end_date:
            match["created_at"]["$lte"] = end_date

    pipeline = [
        {"$match": match},
        {"$group": {"_id": "$sentiment", "count": {"$sum": 1}}}
    ]
    return list(collection.aggregate(pipeline))
