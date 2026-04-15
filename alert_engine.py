"""
alert_engine.py
─────────────────────────────────────────────────────────────
WHY THIS EXISTS:
  If negative sentiment for a tracked keyword suddenly spikes
  (e.g., a PR crisis, product outage, controversial tweet goes viral),
  this module fires alerts to Slack and/or email automatically.

  HOW IT WORKS:
  1. After each tweet is processed, we check the recent
     negative sentiment RATIO for that keyword
  2. If negative% > threshold (default: 65%) for the last
     50 tweets → fire an alert
  3. We throttle alerts: max 1 alert per keyword per 30 minutes
     (prevents spam floods during a sustained crisis)

  ALERT CHANNELS:
  • Slack  → posts to your webhook URL (instant, team-visible)
  • Email  → sends via Gmail SMTP (good for non-Slack users)
─────────────────────────────────────────────────────────────
"""

import smtplib
import json
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, List

import httpx
import redis as redis_lib
from loguru import logger

from config import settings

# ── Redis client (for alert throttling) ────────────────────
_redis = redis_lib.Redis(
    host=settings.redis_host,
    port=settings.redis_port,
    db=settings.redis_db,
    decode_responses=True
)


# ── Throttle Check ──────────────────────────────────────────
def _is_throttled(keyword: str) -> bool:
    """
    Return True if we already fired an alert for this keyword
    within the last 30 minutes.

    Uses Redis key with 30-min TTL as a simple mutex.
    """
    key = f"alert:throttle:{keyword}"
    return bool(_redis.get(key))


def _set_throttle(keyword: str):
    """Mark this keyword as 'alert sent' for 30 minutes."""
    key = f"alert:throttle:{keyword}"
    _redis.setex(key, 1800, "1")  # 1800 seconds = 30 minutes


# ── Negative Ratio Calculator ───────────────────────────────
def _get_negative_ratio(keyword: Optional[str], window: int = 50) -> float:
    """
    Calculate what % of the last `window` tweets for this keyword
    were Negative sentiment.

    Uses MongoDB aggregation to count recent tweets.
    Returns a float between 0.0 and 1.0
    """
    from mongoconnection import collection
    from datetime import datetime, timedelta

    cutoff = datetime.utcnow() - timedelta(hours=1)
    match = {
        "processed": True,
        "created_at": {"$gte": cutoff}
    }
    if keyword:
        match["matched_keywords"] = keyword

    pipeline = [
        {"$match": match},
        {"$sort": {"created_at": -1}},
        {"$limit": window},
        {"$group": {
            "_id": "$sentiment",
            "count": {"$sum": 1}
        }}
    ]

    results = list(collection.aggregate(pipeline))
    counts = {r["_id"]: r["count"] for r in results if r["_id"]}
    total = sum(counts.values())
    if total < 10:  # not enough data to judge
        return 0.0
    return counts.get("Negative", 0) / total


# ── Slack Alert ─────────────────────────────────────────────
def send_slack_alert(keyword: str, negative_ratio: float, sample_tweets: List[str]):
    """
    Post a formatted alert to Slack via Incoming Webhook.

    HOW TO SET UP SLACK WEBHOOK:
    1. Go to https://api.slack.com/apps
    2. Create an app → Enable Incoming Webhooks
    3. Add to your workspace → copy the webhook URL
    4. Paste URL into SLACK_WEBHOOK_URL in .env
    """
    if not settings.slack_webhook_url:
        return

    percent = round(negative_ratio * 100, 1)
    samples_text = "\n".join(f"• _{t[:120]}_" for t in sample_tweets[:3])

    payload = {
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"🚨 Negative Sentiment Spike: #{keyword}"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"*{percent}%* of recent tweets about *#{keyword}* are negative "
                        f"(threshold: {settings.alert_negative_threshold * 100:.0f}%)\n\n"
                        f"*Recent negative tweets:*\n{samples_text}"
                    )
                }
            },
            {
                "type": "context",
                "elements": [{"type": "mrkdwn", "text": f"🕒 Alert fired at {__import__('datetime').datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"}]
            }
        ]
    }

    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.post(settings.slack_webhook_url, json=payload)
            if resp.status_code == 200:
                logger.info(f"✅ Slack alert sent for #{keyword} ({percent}% negative)")
            else:
                logger.warning(f"Slack alert failed: {resp.status_code} {resp.text}")
    except Exception as e:
        logger.error(f"Slack alert error: {e}")


# ── Email Alert ─────────────────────────────────────────────
def send_email_alert(keyword: str, negative_ratio: float, sample_tweets: List[str]):
    """
    Send an email alert via Gmail SMTP.

    HOW TO SET UP GMAIL SMTP:
    1. Enable 2FA on your Google account
    2. Go to myaccount.google.com → Security → App Passwords
    3. Create an App Password for "Mail"
    4. Use that 16-char password in ALERT_EMAIL_PASSWORD in .env
    """
    if not all([settings.alert_email_from, settings.alert_email_password,
                settings.alert_email_to]):
        return

    percent = round(negative_ratio * 100, 1)
    samples_html = "".join(f"<li>{t[:120]}</li>" for t in sample_tweets[:5])

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"🚨 Sentiment Alert: #{keyword} ({percent}% negative)"
    msg["From"] = settings.alert_email_from
    msg["To"] = settings.alert_email_to

    html_body = f"""
    <html><body style="font-family: Arial, sans-serif; padding: 20px;">
      <h2 style="color: #e74c3c;">Negative Sentiment Spike Detected</h2>
      <p>
        <strong>{percent}%</strong> of recent tweets about
        <strong>#{keyword}</strong> are negative.<br>
        Threshold: {settings.alert_negative_threshold * 100:.0f}%
      </p>
      <h3>Sample Negative Tweets:</h3>
      <ul style="color: #555;">{samples_html}</ul>
      <hr>
      <small>This alert was generated by your Real-Time Sentiment Analysis system.</small>
    </body></html>
    """

    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(settings.alert_email_from, settings.alert_email_password)
            server.sendmail(
                settings.alert_email_from,
                settings.alert_email_to,
                msg.as_string()
            )
        logger.info(f"✅ Email alert sent for #{keyword} to {settings.alert_email_to}")
    except Exception as e:
        logger.error(f"Email alert error: {e}")


# ── Sample Negative Tweets ──────────────────────────────────
def _get_sample_negative_tweets(keyword: str, limit: int = 5) -> List[str]:
    """Fetch a few recent negative tweets for inclusion in alerts."""
    from mongoconnection import collection
    from datetime import datetime, timedelta

    cutoff = datetime.utcnow() - timedelta(hours=1)
    docs = list(
        collection.find(
            {"sentiment": "Negative", "created_at": {"$gte": cutoff},
             "matched_keywords": keyword},
            {"text": 1}
        )
        .sort("created_at", -1)
        .limit(limit)
    )
    return [d.get("text", "") for d in docs]


# ── Main Entry: check and fire ──────────────────────────────
def check_and_fire_alert(sentiment: str, keywords: Optional[List[str]]):
    """
    Called by kafka_consumer.py after each tweet is processed.

    Only checks if:
    1. The current tweet is Negative (skip checking for positives)
    2. We have keywords to check against
    3. We haven't recently fired an alert for this keyword
    """
    if sentiment != "Negative" or not keywords:
        return

    for keyword in keywords:
        if _is_throttled(keyword):
            continue  # already alerted recently

        ratio = _get_negative_ratio(keyword)
        if ratio >= settings.alert_negative_threshold:
            logger.warning(
                f"⚠️  Negative spike for #{keyword}: "
                f"{ratio*100:.1f}% >= threshold {settings.alert_negative_threshold*100:.0f}%"
            )

            samples = _get_sample_negative_tweets(keyword)

            # Fire both channels
            send_slack_alert(keyword, ratio, samples)
            send_email_alert(keyword, ratio, samples)

            # Throttle: don't fire again for 30 minutes
            _set_throttle(keyword)
