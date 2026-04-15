"""
processing.py
─────────────────────────────────────────────────────────────
WHY THIS EXISTS:
  This is the "brain" of the project — it takes raw tweet text
  and returns a sentiment label + confidence score.

  HOW IT WORKS (two-model strategy):
  1. PRIMARY: cardiffnlp/twitter-roberta-base-sentiment-latest
     — Trained specifically on tweets (not general text)
     — Understands hashtags, @mentions, slang, emojis
     — Runs on CPU (slower) or GPU (fast)

  2. FALLBACK: VADER (rule-based, no model download needed)
     — Instant, zero GPU, works offline
     — Less accurate but always available
     — Perfect fallback if transformers fail or are too slow

  CACHING:
     Results are cached in Redis for 1 hour by tweet text hash.
     So if the same tweet text appears again (retweets are common!),
     we return the cached result instantly without running inference.

  PREPROCESSING:
     Tweets contain URLs, @mentions, RT prefixes that confuse models.
     We clean them out before inference.
─────────────────────────────────────────────────────────────
"""

import re
import hashlib
import json
from datetime import datetime
from typing import Dict, Any, Optional
from loguru import logger

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from config import settings

# ── Redis cache client ─────────────────────────────────────
import redis as redis_lib
_redis = redis_lib.Redis(
    host=settings.redis_host,
    port=settings.redis_port,
    db=settings.redis_db,
    decode_responses=True
)

# ── VADER (always available, no download needed) ───────────
vader = SentimentIntensityAnalyzer()

# ── Transformer model (lazy loaded on first use) ───────────
# We lazy-load so the app starts up fast and only downloads
# the model when the first tweet arrives
_transformer_pipeline = None

def _get_transformer():
    """Load transformer pipeline on first call (lazy init)."""
    global _transformer_pipeline
    if _transformer_pipeline is None:
        try:
            from transformers import pipeline as hf_pipeline
            logger.info(f"Loading HuggingFace model: {settings.hf_model}")
            _transformer_pipeline = hf_pipeline(
                "sentiment-analysis",
                model=settings.hf_model,
                top_k=None,          # return scores for ALL labels
                truncation=True,
                max_length=128       # tweets are short, 128 tokens is enough
            )
            logger.info("✅ Transformer model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load transformer model: {e}")
            logger.warning("Will use VADER fallback for all sentiment analysis")
    return _transformer_pipeline


# ── Text Preprocessing ─────────────────────────────────────
def preprocess_tweet(text: str) -> str:
    """
    Clean a tweet before sentiment analysis.
    What we remove:
      - RT prefix  (retweet marker, not the author's words)
      - URLs        (links don't carry sentiment)
      - @mentions   (usernames don't carry sentiment)
      - Repeated spaces

    What we KEEP:
      - Hashtags (the word matters, not the # symbol)
      - Emojis   (VADER handles these! They're important sentiment signals)
      - Punctuation (! and ? matter for sentiment)
    """
    text = re.sub(r'^RT @\w+:\s*', '', text)         # Remove RT prefix
    text = re.sub(r'http\S+|www\.\S+', '', text)      # Remove URLs
    text = re.sub(r'@\w+', '', text)                   # Remove @mentions
    text = re.sub(r'\s+', ' ', text).strip()           # Collapse whitespace
    return text


# ── Core Analysis Function ─────────────────────────────────
def analyze_text(text: str, doc_id: Optional[str] = None,
                 force: bool = False) -> Dict[str, Any]:
    """
    Main function: takes tweet text → returns sentiment dict.

    Args:
        text     : Raw tweet text
        doc_id   : MongoDB doc ID (used for cache key)
        force    : If True, skip cache and re-run inference

    Returns dict like:
        {
            "label": "Positive",
            "score": 0.923,
            "model": "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "model_version": "2.0",
            "analyzed_at": datetime(...)
        }

    FLOW:
      1. Check Redis cache  →  return if hit
      2. Preprocess text
      3. Try transformer   →  map labels to Positive/Negative/Neutral
      4. Fall back to VADER if transformer unavailable
      5. Store result in Redis cache (1 hour TTL)
      6. Return result
    """
    if not text or not text.strip():
        return {"label": "Neutral", "score": 0.5, "model": "none",
                "model_version": settings.model_version,
                "analyzed_at": datetime.utcnow()}

    # ── Step 1: Check Redis cache ──────────────────────────
    cache_key = f"sentiment:{hashlib.md5(text.encode()).hexdigest()}"
    if not force:
        try:
            cached = _redis.get(cache_key)
            if cached:
                result = json.loads(cached)
                result["from_cache"] = True
                return result
        except Exception:
            pass  # Redis down → continue without cache

    # ── Step 2: Preprocess ────────────────────────────────
    clean_text = preprocess_tweet(text)

    # ── Step 3: Try transformer model ─────────────────────
    pipeline = _get_transformer()
    meta: Dict[str, Any] = {}

    if pipeline:
        try:
            raw_results = pipeline(clean_text)[0]
            # raw_results looks like:
            # [{"label": "LABEL_0", "score": 0.9}, {"label": "LABEL_1", ...}, ...]
            # or [{"label": "positive", "score": 0.8}, ...]
            # We find the label with the highest score
            best = max(raw_results, key=lambda x: x["score"])
            label = best["label"].upper()
            score = float(best["score"])

            # Normalize label names to our standard 3-class output
            if "0" in label or "NEG" in label:
                canonical = "Negative"
            elif "2" in label or "POS" in label:
                canonical = "Positive"
            else:
                canonical = "Neutral"

            meta = {
                "label": canonical,
                "score": round(score, 4),
                "raw_label": best["label"],
                "model": settings.hf_model,
                "model_version": settings.model_version,
                "analyzed_at": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            logger.warning(f"Transformer inference failed: {e}, falling back to VADER")
            pipeline = None  # force fallback below

    # ── Step 4: VADER fallback ────────────────────────────
    if not pipeline or not meta:
        scores = vader.polarity_scores(clean_text)
        compound = scores["compound"]
        if compound >= 0.05:
            canonical = "Positive"
        elif compound <= -0.05:
            canonical = "Negative"
        else:
            canonical = "Neutral"

        meta = {
            "label": canonical,
            "score": round(abs(compound), 4),
            "raw_scores": scores,
            "model": "vader",
            "model_version": "vader-1.0",
            "analyzed_at": datetime.utcnow().isoformat(),
        }

    # ── Step 5: Cache result in Redis (1 hour TTL) ────────
    try:
        _redis.setex(cache_key, 3600, json.dumps(meta))
    except Exception:
        pass  # Redis down → continue, just won't cache

    return meta


# ── Redis Live Counter Helpers ────────────────────────────
def increment_sentiment_counter(sentiment: str, keyword: Optional[str] = None):
    """
    Increment real-time counters in Redis.
    These power the live gauge on the dashboard without hitting MongoDB.

    Keys stored:
      sentiment:total:Positive  → global count
      sentiment:kw:{keyword}:Positive → per-keyword count
    """
    try:
        _redis.incr(f"sentiment:total:{sentiment}")
        _redis.expire(f"sentiment:total:{sentiment}", 86400)  # 24h TTL
        if keyword:
            _redis.incr(f"sentiment:kw:{keyword}:{sentiment}")
            _redis.expire(f"sentiment:kw:{keyword}:{sentiment}", 86400)
    except Exception:
        pass


def get_live_counts(keyword: Optional[str] = None) -> Dict[str, int]:
    """
    Read live counts from Redis.
    Fast O(1) lookup — no MongoDB scan needed.
    """
    try:
        prefix = f"sentiment:kw:{keyword}" if keyword else "sentiment:total"
        pos = int(_redis.get(f"{prefix}:Positive") or 0)
        neg = int(_redis.get(f"{prefix}:Negative") or 0)
        neu = int(_redis.get(f"{prefix}:Neutral") or 0)
        return {"Positive": pos, "Negative": neg, "Neutral": neu}
    except Exception:
        return {"Positive": 0, "Negative": 0, "Neutral": 0}


# ── Hashtag Extractor ─────────────────────────────────────
def extract_hashtags(text: str) -> list:
    """Extract all hashtags from tweet text as a clean list."""
    return [tag.lower() for tag in re.findall(r'#(\w+)', text)]
