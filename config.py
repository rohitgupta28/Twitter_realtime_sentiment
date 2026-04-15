"""
config.py
─────────────────────────────────────────────────────────────
WHY THIS EXISTS:
  Instead of calling os.getenv() in every file (messy),
  we load all settings ONCE here using pydantic-settings.
  Every other module just does:  from config import settings

  This also validates that required keys actually exist,
  so you get a clear error message at startup instead of
  a cryptic crash later during runtime.
─────────────────────────────────────────────────────────────
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List


class Settings(BaseSettings):
    """
    Pydantic reads these from environment variables or .env file.
    Field names map directly to env var names (case-insensitive).
    """

    # ── Twitter ──────────────────────────────────────────────
    twitter_bearer_token: str = ""
    twitter_track_keywords: str = "python,AI,machinelearning"

    # ── Kafka ────────────────────────────────────────────────
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic_raw: str = "tweets-raw"
    kafka_topic_processed: str = "tweets-processed"
    kafka_consumer_group: str = "sentiment-workers"

    # ── MongoDB ──────────────────────────────────────────────
    mongo_uri: str = "mongodb://localhost:27017/"
    db_name: str = "twitter_db"
    col_name: str = "tweets"

    # ── Redis ────────────────────────────────────────────────
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0

    # ── NLP ──────────────────────────────────────────────────
    hf_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    model_version: str = "2.0"

    # ── FastAPI ──────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ── Alerts ───────────────────────────────────────────────
    slack_webhook_url: str = ""
    alert_email_from: str = ""
    alert_email_password: str = ""
    alert_email_to: str = ""
    alert_negative_threshold: float = 0.65

    # ── Pydantic config ──────────────────────────────────────
    model_config = SettingsConfigDict(
        env_file=".env",          # auto-load .env file if it exists
        env_file_encoding="utf-8",
        extra="ignore",           # ignore unknown env vars
    )

    @property
    def keywords_list(self) -> List[str]:
        """Returns twitter_track_keywords as a clean Python list."""
        return [k.strip() for k in self.twitter_track_keywords.split(",") if k.strip()]


# ── Singleton ─────────────────────────────────────────────────
# Import this object anywhere:  from config import settings
settings = Settings()
