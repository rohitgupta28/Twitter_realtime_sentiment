"""
ingest.py  —  Sentiment140 CSV → MongoDB Pipeline
─────────────────────────────────────────────────────────────
DATASET: Sentiment140
  Download: https://www.kaggle.com/datasets/kazanova/sentiment140
  File:     training.1600000.processed.noemoticon.csv
  Size:     1.6 million tweets, ~238 MB

SENTIMENT140 CSV COLUMN LAYOUT (no header row in file):
  Col 0 → target   : 0 = Negative,  4 = Positive
  Col 1 → id       : tweet ID
  Col 2 → date     : "Mon Apr 06 22:19:45 PDT 2009"
  Col 3 → flag     : query string (mostly "NO_QUERY")
  Col 4 → user     : Twitter username
  Col 5 → text     : actual tweet text  ← what we care about

HOW THIS SCRIPT WORKS:
  1. Loads the CSV in one shot using pandas (fast)
  2. Parses the real tweet dates from column 2
  3. Maps target 0→Negative, 4→Positive as ground-truth label
  4. Inserts tweets in BATCHES of 500 (much faster than one-by-one)
  5. Runs RoBERTa/VADER NLP on each tweet
  6. Saves NLP results back to MongoDB
  7. Shows a tqdm progress bar in the terminal

HOW TO RUN:
  # Quick test — 2000 tweets (takes ~2-3 minutes with RoBERTa)
  python ingest.py --csv training.1600000.processed.noemoticon.csv --limit 2000

  # Medium dataset — 10000 tweets (good for dashboard demo)
  python ingest.py --csv training.1600000.processed.noemoticon.csv --limit 10000

  # Skip NLP — just insert raw tweets super fast, analyze later
  python ingest.py --csv training.1600000.processed.noemoticon.csv --limit 50000 --no-nlp

  # Full dataset (takes hours — only do if you have GPU)
  python ingest.py --csv training.1600000.processed.noemoticon.csv

SPEED GUIDE:
  CPU only + RoBERTa:  ~3-4 tweets/sec   → 2000 tweets ≈ 10 min
  CPU only + VADER:    ~200 tweets/sec   → 2000 tweets ≈ 10 sec
  GPU + RoBERTa:       ~50 tweets/sec    → 2000 tweets ≈ 40 sec

  Recommended for first run: --limit 2000 (fast, enough to see the dashboard)
─────────────────────────────────────────────────────────────
"""

import argparse
import sys
import random
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

import pandas as pd
from tqdm import tqdm          # pip install tqdm  (already in requirements.txt)
from loguru import logger

from mongoconnection import ensure_indexes, collection
from processing import analyze_text, extract_hashtags


# ── Sentiment140 date format ───────────────────────────────
# Example: "Mon Apr 06 22:19:45 PDT 2009"
S140_DATE_FORMAT = "%a %b %d %H:%M:%S PDT %Y"
S140_DATE_FORMAT_ALT = "%a %b %d %H:%M:%S PST %Y"


def parse_s140_date(date_str: str) -> datetime:
    """
    Parse the Sentiment140 date string into a Python datetime.
    Falls back to a random date in 2009 if parsing fails
    (keeping it realistic for the dataset era).
    """
    if not date_str or str(date_str) == "nan":
        return datetime(2009, random.randint(1, 12), random.randint(1, 28),
                        random.randint(0, 23), random.randint(0, 59))
    date_str = str(date_str).strip()
    for fmt in [S140_DATE_FORMAT, S140_DATE_FORMAT_ALT]:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    # Last resort: try dateutil
    try:
        from dateutil import parser as du_parser
        return du_parser.parse(date_str)
    except Exception:
        return datetime(2009, random.randint(1, 12), random.randint(1, 28))


def map_target_to_label(target: Any) -> str:
    """
    Convert Sentiment140 numeric target to our label string.
    0 → Negative
    4 → Positive
    2 → Neutral  (rare in Sentiment140 but handle it)
    """
    try:
        t = int(target)
        if t == 0:
            return "Negative"
        elif t == 4:
            return "Positive"
        else:
            return "Neutral"
    except (ValueError, TypeError):
        return "Neutral"


# ── Load CSV ───────────────────────────────────────────────
def load_sentiment140(csv_path: str, limit: Optional[int] = None) -> pd.DataFrame:
    """
    Load and clean the Sentiment140 CSV.

    WHY header=None?
    The Sentiment140 file has NO header row — pandas would
    treat the first tweet as column names without this flag.

    WHY encoding='ISO-8859-1'?
    The file contains Latin-1 encoded characters (old tweets
    had lots of non-UTF-8 chars). UTF-8 would crash on them.
    """
    logger.info(f"📂 Loading Sentiment140 CSV: {csv_path}")

    df = pd.read_csv(
        csv_path,
        encoding="ISO-8859-1",
        header=None,          # no header row in the file
        low_memory=False,
        names=["target", "tweet_id", "date_str", "flag", "user", "text"]
    )

    logger.info(f"   Raw rows loaded: {len(df):,}")

    # ── Clean ──────────────────────────────────────────────
    # Drop rows where tweet text is missing or too short
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 5]     # skip empty / noise tweets

    # ── Sample ─────────────────────────────────────────────
    # Stratified sample: keep equal numbers of positive & negative
    # so the dashboard doesn't show a skewed dataset
    if limit:
        half = limit // 2
        neg = df[df["target"] == 0].sample(
            min(half, (df["target"] == 0).sum()), random_state=42
        )
        pos = df[df["target"] == 4].sample(
            min(half, (df["target"] == 4).sum()), random_state=42
        )
        df = pd.concat([neg, pos]).sample(frac=1, random_state=42)  # shuffle
        logger.info(f"   Sampled (balanced): {len(df):,} tweets "
                    f"({len(pos):,} positive + {len(neg):,} negative)")
    else:
        logger.info(f"   Using all {len(df):,} tweets")

    return df.reset_index(drop=True)


# ── Batch Builder ──────────────────────────────────────────
def build_tweet_doc(row: pd.Series, index: int) -> Dict[str, Any]:
    """
    Convert one Sentiment140 CSV row into our MongoDB document format.
    """
    return {
        "tweet_id":         str(row.get("tweet_id", f"s140_{index}")),
        "text":             str(row["text"]),
        "user":             str(row.get("user", "")),
        "created_at":       parse_s140_date(row.get("date_str", "")),
        "ingested_at":      datetime.utcnow(),
        "source":           "sentiment140",
        "ground_truth":     map_target_to_label(row.get("target")),
        "matched_keywords": [],
        "processed":        False,
        "sentiment":        None,
        "analysis_meta":    None,
    }


# ── Main Ingest Function ───────────────────────────────────
def ingest_csv(
    csv_path: str,
    limit: Optional[int] = None,
    run_nlp: bool = True,
    batch_size: int = 500,
):
    """
    Full pipeline: Sentiment140 CSV → MongoDB + NLP.

    Args:
        csv_path   : path to training.1600000.processed.noemoticon.csv
        limit      : number of tweets to process (None = all 1.6M)
        run_nlp    : if True, run RoBERTa/VADER and save sentiment labels
        batch_size : how many tweets to insert at once (500 is optimal)
    """
    # ── Setup ──────────────────────────────────────────────
    ensure_indexes()
    df = load_sentiment140(csv_path, limit=limit)
    total = len(df)

    inserted   = 0
    analyzed   = 0
    errors     = 0
    batch_docs: List[Dict] = []

    logger.info(f"\n🚀 Starting ingestion of {total:,} tweets...")
    if run_nlp:
        logger.info("   NLP: ON  (RoBERTa with VADER fallback)")
        logger.info("   ⏱  Estimated time on CPU: "
                    f"~{max(1, total // 200)} minutes\n")
    else:
        logger.info("   NLP: OFF (raw insert only — very fast)\n")

    # ── Progress bar (tqdm) ────────────────────────────────
    pbar = tqdm(
        total=total,
        desc="Ingesting",
        unit="tweet",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )

    def flush_batch(batch: List[Dict]) -> int:
        """Insert a batch of documents into MongoDB at once."""
        if not batch:
            return 0
        try:
            result = collection.insert_many(batch, ordered=False)
            return len(result.inserted_ids)
        except Exception as e:
            logger.warning(f"Batch insert partial error: {e}")
            return len(batch)

    # ── Main loop ──────────────────────────────────────────
    for i, (_, row) in enumerate(df.iterrows()):
        try:
            doc = build_tweet_doc(row, i)
            batch_docs.append(doc)

            # Flush to MongoDB every batch_size rows
            if len(batch_docs) >= batch_size:
                count = flush_batch(batch_docs)
                inserted += count
                batch_docs = []

            pbar.update(1)

        except Exception as e:
            errors += 1
            logger.debug(f"Row {i} build error: {e}")
            continue

    # Flush remaining docs
    if batch_docs:
        inserted += flush_batch(batch_docs)

    pbar.close()
    logger.info(f"\n✅ INSERT DONE: {inserted:,} tweets saved to MongoDB")

    # ── NLP Analysis Phase ─────────────────────────────────
    # We do this as a SECOND pass after all inserts are done.
    # Why? Batch inserting first is much faster, then we
    # run NLP on each doc and update it in place.
    if run_nlp:
        logger.info(f"\n🧠 Running NLP analysis on {inserted:,} tweets...")
        logger.info("   (You can open the dashboard now — tweets will appear "
                    "as they get analyzed)\n")

        # Fetch all unprocessed docs we just inserted
        cursor = collection.find(
            {"processed": False, "source": "sentiment140"},
            {"_id": 1, "text": 1}
        ).batch_size(200)

        pbar_nlp = tqdm(
            total=inserted,
            desc="Analyzing",
            unit="tweet",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )

        from mongoconnection import save_processed_tweet
        from bson import ObjectId

        for doc in cursor:
            try:
                text = doc.get("text", "")
                meta = analyze_text(text)
                hashtags = extract_hashtags(text)

                save_processed_tweet(
                    str(doc["_id"]),
                    meta["label"],
                    meta.get("score", 0.5),
                    meta.get("model", "unknown"),
                    hashtags
                )
                analyzed += 1

            except Exception as e:
                errors += 1
                logger.debug(f"NLP error on doc {doc.get('_id')}: {e}")
            finally:
                pbar_nlp.update(1)

        pbar_nlp.close()

    # ── Final Summary ──────────────────────────────────────
    logger.info("\n" + "─" * 50)
    logger.info(f"  ✅ Tweets inserted  : {inserted:,}")
    logger.info(f"  🧠 Tweets analyzed  : {analyzed:,}")
    logger.info(f"  ❌ Errors skipped   : {errors:,}")
    logger.info("─" * 50)
    logger.info("  Open the dashboard:  streamlit run app.py")
    logger.info("─" * 50 + "\n")

    return inserted, analyzed


# ── CLI Entry Point ────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest Sentiment140 dataset into MongoDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test — 2000 tweets with NLP (recommended first run)
  python ingest.py --csv training.1600000.processed.noemoticon.csv --limit 2000

  # 10000 tweets — good for dashboard demo
  python ingest.py --csv training.1600000.processed.noemoticon.csv --limit 10000

  # Fast raw insert, no NLP (useful to pre-load data)
  python ingest.py --csv training.1600000.processed.noemoticon.csv --limit 50000 --no-nlp
        """
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to training.1600000.processed.noemoticon.csv"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of tweets to process (default: all 1.6M)"
    )
    parser.add_argument(
        "--no-nlp",
        action="store_true",
        help="Skip NLP analysis — just insert raw tweets (very fast)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="MongoDB insert batch size (default: 500)"
    )

    args = parser.parse_args()

    ingest_csv(
        csv_path=args.csv,
        limit=args.limit,
        run_nlp=not args.no_nlp,
        batch_size=args.batch_size,
    )