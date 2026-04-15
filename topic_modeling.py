"""
topic_modeling.py
─────────────────────────────────────────────────────────────
WHY THIS EXISTS:
  Sentiment tells you HOW people feel.
  Topics tell you WHAT they're talking about.

  Latent Dirichlet Allocation (LDA) discovers hidden "topics"
  in a collection of documents without being told in advance
  what the topics are. It's unsupervised learning.

  HOW LDA WORKS:
  Imagine every tweet is a mixture of topics, and every topic
  is a mixture of words. LDA reverse-engineers:
    "Given these tweets, what were the latent topics?"

  EXAMPLE OUTPUT:
  Topic 1: 0.08 * "model" + 0.06 * "training" + 0.05 * "data" → "ML Training"
  Topic 2: 0.09 * "price" + 0.07 * "stock" + 0.06 * "market" → "Finance"
  Topic 3: 0.07 * "gpu" + 0.06 * "nvidia" + 0.05 * "cuda"   → "Hardware"

  NEW IN THIS VERSION:
  • coherence_score() — measures how good the topics are (higher = better)
  • label_topic() — tries to give each topic a human-readable name
  • Added caching so we don't re-run LDA on every dashboard refresh
─────────────────────────────────────────────────────────────
"""

import hashlib
import json
from typing import List, Tuple, Optional, Dict, Any

import gensim
from gensim import corpora
from gensim.models import CoherenceModel
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from loguru import logger

import redis as redis_lib
from config import settings

# ── NLTK Downloads (first run only) ───────────────────────
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

STOP = set(stopwords.words("english"))
# Add Twitter-specific stopwords that don't add meaning
STOP.update([
    "rt", "via", "amp", "like", "just", "get", "got", "know",
    "think", "way", "one", "us", "will", "would", "could", "good",
    "people", "time", "today", "day", "new", "make", "see", "go"
])

# ── Redis cache ────────────────────────────────────────────
_redis = redis_lib.Redis(
    host=settings.redis_host,
    port=settings.redis_port,
    db=settings.redis_db,
    decode_responses=True
)
LDA_CACHE_TTL = 600  # 10 minutes — topic model re-run every 10 min


# ── Text Preprocessing ─────────────────────────────────────
def preprocess_texts(texts: List[str]) -> List[List[str]]:
    """
    Tokenize and clean tweets for LDA.

    Steps:
    1. Lowercase
    2. Keep only alphabetic words (remove numbers, symbols)
    3. Remove stopwords and very short words (≤2 chars)
    4. Keep words ≥ 3 chars (avoids noise from initials)
    """
    docs = []
    for text in texts:
        tokens = [
            w.lower() for w in word_tokenize(text)
            if w.isalpha() and len(w) > 2
        ]
        tokens = [w for w in tokens if w not in STOP]
        if tokens:
            docs.append(tokens)
    return docs


# ── Main LDA Function ──────────────────────────────────────
def compute_lda_topics(
    texts: List[str],
    num_topics: int = 5,
    use_cache: bool = True
) -> Tuple[List[Tuple], Any, Any, Any]:
    """
    Run LDA topic modeling on a list of tweet texts.

    Args:
        texts      : List of raw tweet strings
        num_topics : How many topics to find (5 is usually good for tweets)
        use_cache  : Check Redis cache before recomputing

    Returns:
        topics     : List of (topic_id, word_weights) tuples
        lda        : Trained Gensim LDA model
        dictionary : Gensim Dictionary (word ↔ id mapping)
        corpus     : Bag-of-words corpus

    CHOOSING num_topics:
    Too few: topics are vague (everything lumped together)
    Too many: topics repeat or are too narrow
    For 200-500 tweets, 4-6 topics is usually right.
    """
    if len(texts) < 10:
        logger.warning("Not enough texts for LDA (need ≥10)")
        return [], None, None, None

    # ── Check cache ────────────────────────────────────────
    if use_cache:
        cache_key = f"lda:{hashlib.md5(''.join(texts[:50]).encode()).hexdigest()}:{num_topics}"
        cached = _redis.get(cache_key)
        if cached:
            try:
                data = json.loads(cached)
                logger.debug("LDA result from cache")
                return data["topics"], None, None, None
            except Exception:
                pass

    # ── Preprocess ─────────────────────────────────────────
    docs = preprocess_texts(texts)
    if not docs:
        return [], None, None, None

    # ── Build dictionary and corpus ────────────────────────
    # Dictionary maps each unique word to an integer ID
    dictionary = corpora.Dictionary(docs)

    # Filter extremes:
    # - Remove words in fewer than 2 documents (too rare, probably noise)
    # - Remove words in more than 80% of documents (too common, no discrimination)
    dictionary.filter_extremes(no_below=2, no_above=0.8)

    # Corpus: list of (word_id, count) tuples per document
    corpus = [dictionary.doc2bow(doc) for doc in docs]

    if not corpus or len(dictionary) < 5:
        logger.warning("Vocabulary too small for LDA after filtering")
        return [], None, None, None

    # ── Train LDA model ────────────────────────────────────
    # passes=15: go through the corpus 15 times (more = more accurate, slower)
    # alpha='auto': let gensim find the best sparsity parameter
    lda = gensim.models.LdaModel(
        corpus,
        num_topics=num_topics,
        id2word=dictionary,
        passes=15,
        random_state=42,
        alpha="auto",
        per_word_topics=True
    )

    # Get human-readable topic strings
    topics = lda.print_topics(num_words=7)

    # ── Cache result ───────────────────────────────────────
    if use_cache:
        try:
            _redis.setex(
                cache_key, LDA_CACHE_TTL,
                json.dumps({"topics": [[t[0], t[1]] for t in topics]})
            )
        except Exception:
            pass

    return topics, lda, dictionary, corpus


# ── Coherence Score ────────────────────────────────────────
def compute_coherence(lda_model, corpus, dictionary, texts: List[str]) -> float:
    """
    Compute CV coherence score for the LDA model.
    Coherence measures how semantically similar the top words
    in each topic are to each other.

    Score range: typically 0.3 to 0.7
    > 0.5 = good topics
    > 0.6 = excellent topics
    """
    try:
        docs = preprocess_texts(texts)
        coherence = CoherenceModel(
            model=lda_model,
            texts=docs,
            dictionary=dictionary,
            coherence="c_v"
        )
        return round(coherence.get_coherence(), 4)
    except Exception as e:
        logger.warning(f"Could not compute coherence: {e}")
        return 0.0


# ── Topic Labeler ──────────────────────────────────────────
def label_topic(topic_str: str) -> str:
    """
    Try to give a topic a short human-readable label
    based on its top words.

    This is a simple heuristic — for production, you could
    use an LLM to generate topic labels automatically.
    """
    topic_keywords = {
        "ai": "Artificial Intelligence",
        "model": "AI Models",
        "data": "Data Science",
        "train": "Model Training",
        "learn": "Machine Learning",
        "python": "Python Development",
        "code": "Software Development",
        "stock": "Stock Market",
        "price": "Finance/Markets",
        "crypto": "Cryptocurrency",
        "news": "Current Events",
        "game": "Gaming",
        "health": "Health",
        "covid": "COVID-19",
        "climate": "Climate",
        "election": "Politics",
        "music": "Music",
        "sport": "Sports",
    }

    topic_lower = topic_str.lower()
    for keyword, label in topic_keywords.items():
        if keyword in topic_lower:
            return label

    return "General Discussion"


# ── Format Topics for Display ──────────────────────────────
def format_topics_for_display(
    topics: List[Tuple],
    lda_model=None,
    corpus=None,
    dictionary=None,
    texts: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Convert raw LDA output into a clean list for the dashboard.

    Returns:
    [
        {
            "topic_id": 0,
            "label": "Artificial Intelligence",
            "top_words": ["model", "gpt", "training", "llm"],
            "raw": "0.08*\"model\" + 0.06*\"gpt\" + ..."
        },
        ...
    ]
    """
    import re

    result = []
    for topic_id, topic_str in topics:
        # Extract just the words (not the weights)
        words = re.findall(r'"(\w+)"', topic_str)

        result.append({
            "topic_id": topic_id,
            "label": label_topic(topic_str),
            "top_words": words,
            "raw": topic_str
        })

    return result
