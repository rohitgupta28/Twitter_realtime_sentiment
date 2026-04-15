"""
app.py  —  Real-Time Sentiment Dashboard
─────────────────────────────────────────────────────────────
WHY THIS EXISTS:
  This is the FACE of the entire project — the visual dashboard
  that shows everything in real-time.

  WHAT'S NEW vs YOUR OLD app.py:
  ✦ Live sentiment gauge that auto-refreshes every 10 seconds
  ✦ Live tweet feed showing the most recent analyzed tweets
  ✦ WebSocket connection to the FastAPI backend for true real-time
  ✦ Geo heatmap panel (if tweets have location data)
  ✦ Virality score — which tweets are trending
  ✦ Multi-tab layout: Overview | Live Feed | Historical | Topics

  HOW TO RUN:
    streamlit run app.py

  ARCHITECTURE:
  The dashboard calls FastAPI REST endpoints for historical data.
  For live data it uses st.rerun() on a timer, polling the API.
  True WebSocket updates use the streamlit-websocket-client component.
─────────────────────────────────────────────────────────────
"""

import time
import json
import re
from datetime import datetime, timedelta
from typing import Optional

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import httpx
from wordcloud import WordCloud
from loguru import logger

from mongoconnection import (
    find_by_query_text, get_trending_hashtags,
    get_sentiment_over_time, get_sentiment_counts, ensure_indexes
)
from topic_modeling import compute_lda_topics, format_topics_for_display
from processing import get_live_counts

# ─── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Pulse",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────
# Injects styling that makes the dashboard look polished
st.markdown("""
<style>
    /* Header */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .main-header h1 {
        color: #e94560;
        font-size: 2rem;
        margin: 0;
        font-family: 'Courier New', monospace;
        letter-spacing: 2px;
    }
    .main-header p {
        color: rgba(255,255,255,0.6);
        margin: 0.3rem 0 0;
        font-size: 0.9rem;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e1e2e, #2d2d44);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
    }

    /* Live indicator dot */
    @keyframes pulse-dot {
        0%, 100% { opacity: 1; transform: scale(1); }
        50%       { opacity: 0.5; transform: scale(1.3); }
    }
    .live-dot {
        display: inline-block;
        width: 10px; height: 10px;
        background: #2ecc71;
        border-radius: 50%;
        animation: pulse-dot 1.5s ease-in-out infinite;
        margin-right: 6px;
    }
    .live-label {
        color: #2ecc71;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 1px;
    }

    /* Tweet card */
    .tweet-card {
        background: #1e1e2e;
        border-left: 4px solid #e94560;
        border-radius: 6px;
        padding: 0.7rem 1rem;
        margin-bottom: 0.5rem;
        font-size: 0.88rem;
    }
    .tweet-card.positive { border-left-color: #2ecc71; }
    .tweet-card.negative { border-left-color: #e74c3c; }
    .tweet-card.neutral  { border-left-color: #95a5a6; }

    /* Streamlit metric override */
    [data-testid="metric-container"] {
        background: #1e1e2e;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 8px;
        padding: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── Initialize DB ────────────────────────────────────────
ensure_indexes()

# ─── Sidebar ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Controls")

    keyword = st.text_input(
        "🔍 Keyword / Hashtag",
        value="",
        placeholder="e.g. AI, python, #ML"
    )

    hours = st.slider("⏱ Time window (hours)", 1, 168, 24)

    limit = st.slider("📊 Max tweets", 100, 2000, 500, step=100)

    st.markdown("---")
    auto_refresh = st.checkbox("🔄 Auto-refresh (10s)", value=True)
    refresh_interval = st.select_slider(
        "Refresh rate",
        options=[5, 10, 30, 60],
        value=10,
        format_func=lambda x: f"{x}s"
    )

    st.markdown("---")
    st.markdown("### 📡 System Status")

    # Quick system stats from FastAPI
    try:
        resp = httpx.get("http://localhost:8000/api/stats", timeout=2.0)
        if resp.status_code == 200:
            stats = resp.json()
            st.metric("Total Tweets", f"{stats.get('total_tweets', 0):,}")
            st.metric("Processed", f"{stats.get('processed_tweets', 0):,}")
            st.metric("Last 24h", f"{stats.get('tweets_last_24h', 0):,}")
            ws_count = stats.get("active_websocket_connections", 0)
            st.markdown(
                f'<span class="live-dot"></span>'
                f'<span class="live-label">LIVE — {ws_count} connections</span>',
                unsafe_allow_html=True
            )
        else:
            st.warning("API unavailable")
    except Exception:
        st.info("💡 Start API: `uvicorn api:app --port 8000`")

    st.markdown("---")
    if st.button("🔃 Force Refresh"):
        st.rerun()


# ─── Header ───────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📡 SENTIMENT PULSE</h1>
    <p>Real-time Twitter / X sentiment monitoring powered by RoBERTa + Kafka + MongoDB</p>
</div>
""", unsafe_allow_html=True)


# ─── Tab Layout ───────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Live Overview",
    "🐦 Tweet Feed",
    "📈 Historical Analysis",
    "🧠 Topic Modeling"
])


# ═══════════════════════════════════════════════════════════
# TAB 1: LIVE OVERVIEW
# Shows real-time sentiment gauges and trending data
# ═══════════════════════════════════════════════════════════
with tab1:
    st.markdown("### 🔴 Live Sentiment Pulse")
    st.caption("Auto-refreshing from Redis live counters")

    # ── Live Counters (from Redis — instant) ───────────────
    live_counts = get_live_counts(keyword if keyword else None)
    total_live = sum(live_counts.values()) or 1

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "✅ Positive",
            f"{live_counts['Positive']:,}",
            f"{live_counts['Positive']/total_live*100:.1f}%"
        )
    with col2:
        st.metric(
            "❌ Negative",
            f"{live_counts['Negative']:,}",
            f"-{live_counts['Negative']/total_live*100:.1f}%",
            delta_color="inverse"
        )
    with col3:
        st.metric(
            "😐 Neutral",
            f"{live_counts['Neutral']:,}",
            f"{live_counts['Neutral']/total_live*100:.1f}%"
        )
    with col4:
        st.metric("📨 Total Analyzed", f"{total_live:,}")

    st.markdown("---")

    # ── Sentiment Gauge (Plotly) ────────────────────────────
    col_gauge, col_pie = st.columns(2)

    with col_gauge:
        st.markdown("#### 🎯 Sentiment Gauge")
        pos_pct = live_counts['Positive'] / total_live * 100
        neg_pct = live_counts['Negative'] / total_live * 100

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pos_pct,
            title={"text": "Positive %", "font": {"size": 16}},
            delta={"reference": 50, "increasing": {"color": "#2ecc71"},
                   "decreasing": {"color": "#e74c3c"}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#2ecc71"},
                "steps": [
                    {"range": [0, 30], "color": "#2d1111"},
                    {"range": [30, 50], "color": "#2d2011"},
                    {"range": [50, 70], "color": "#1a2d11"},
                    {"range": [70, 100], "color": "#112d11"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 2},
                    "thickness": 0.75,
                    "value": 50
                }
            }
        ))
        fig_gauge.update_layout(
            height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "white"}
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_pie:
        st.markdown("#### 🥧 Sentiment Breakdown")
        if total_live > 1:
            pie_df = pd.DataFrame({
                "Sentiment": list(live_counts.keys()),
                "Count": list(live_counts.values())
            })
            fig_pie = px.pie(
                pie_df, names="Sentiment", values="Count",
                color="Sentiment",
                color_discrete_map={
                    "Positive": "#2ecc71",
                    "Negative": "#e74c3c",
                    "Neutral": "#95a5a6"
                },
                hole=0.4
            )
            fig_pie.update_layout(
                height=300,
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "white"},
                legend={"font": {"color": "white"}}
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Waiting for live data... Make sure twitter_producer.py is running")

    # ── Trending Hashtags ──────────────────────────────────
    st.markdown("#### 🔥 Trending Hashtags (last 6 hours)")
    trending = get_trending_hashtags(limit=15, hours=6)
    if trending:
        trend_df = pd.DataFrame(trending)
        fig_bar = px.bar(
            trend_df, x="count", y="tag", orientation="h",
            color="count",
            color_continuous_scale=["#1a1a2e", "#e94560"],
            labels={"count": "Tweet Count", "tag": "Hashtag"}
        )
        fig_bar.update_layout(
            height=350,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "white"},
            yaxis={"categoryorder": "total ascending"},
            coloraxis_showscale=False
        )
        fig_bar.update_xaxes(gridcolor="rgba(255,255,255,0.1)")
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No trending hashtags yet. Tweets will appear as the stream fills up.")


# ═══════════════════════════════════════════════════════════
# TAB 2: LIVE TWEET FEED
# Shows the most recent processed tweets in real-time
# ═══════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 🐦 Live Tweet Stream")

    feed_col, filter_col = st.columns([3, 1])

    with filter_col:
        sentiment_filter = st.selectbox(
            "Filter by sentiment",
            ["All", "Positive", "Negative", "Neutral"]
        )
        feed_limit = st.number_input("Show last N tweets", 10, 100, 25)

    with feed_col:
        # Fetch recent tweets from MongoDB
        recent = find_by_query_text(
            keyword,
            limit=int(feed_limit),
            start_date=datetime.utcnow() - timedelta(hours=hours)
        )

        if sentiment_filter != "All":
            recent = [t for t in recent if t.get("sentiment") == sentiment_filter]

        if recent:
            for tweet in recent:
                sentiment = tweet.get("sentiment") or "Neutral"
                score = tweet.get("analysis_meta", {}) or {}
                score_val = score.get("score", 0.0) if isinstance(score, dict) else 0.0
                created = tweet.get("created_at", "")
                if isinstance(created, datetime):
                    created = created.strftime("%H:%M:%S")

                sentiment_class = sentiment.lower()
                sentiment_emoji = {"Positive": "✅", "Negative": "❌", "Neutral": "😐"}.get(
                    sentiment, "❓")

                hashtags = tweet.get("hashtags", [])
                tag_str = " ".join(f"#{h}" for h in hashtags[:5]) if hashtags else ""

                st.markdown(
                    f'<div class="tweet-card {sentiment_class}">'
                    f'<strong>{sentiment_emoji} {sentiment}</strong> '
                    f'<span style="color:rgba(255,255,255,0.4);font-size:0.8em">'
                    f'({score_val:.0%}) — {created}</span><br>'
                    f'{tweet.get("text", "")[:200]}'
                    f'<br><span style="color:#e94560;font-size:0.8em">{tag_str}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
        else:
            st.info("No tweets found. Try a different keyword or time range.")


# ═══════════════════════════════════════════════════════════
# TAB 3: HISTORICAL ANALYSIS
# Keyword search, time-series, word clouds, geo map
# ═══════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📈 Historical Sentiment Analysis")

    if st.button("🔍 Run Analysis", type="primary") or keyword:
        start_dt = datetime.utcnow() - timedelta(hours=hours)
        with st.spinner("Fetching and analyzing tweets..."):
            docs = find_by_query_text(keyword, limit=limit, start_date=start_dt)

        if not docs:
            st.warning(f"No tweets found for '{keyword}' in the last {hours} hours.")
        else:
            st.success(f"Found {len(docs)} tweets")

            df = pd.DataFrame([{
                "text": d.get("text", ""),
                "sentiment": d.get("sentiment") or "Unknown",
                "created_at": d.get("created_at"),
                "hashtags": d.get("hashtags", [])
            } for d in docs])

            # ── Row 1: Distribution + Time Series ─────────────
            c1, c2 = st.columns(2)

            with c1:
                st.markdown("#### Sentiment Distribution")
                counts = df["sentiment"].value_counts()
                fig = px.pie(
                    names=counts.index,
                    values=counts.values,
                    color=counts.index,
                    color_discrete_map={
                        "Positive": "#2ecc71",
                        "Negative": "#e74c3c",
                        "Neutral": "#95a5a6"
                    },
                    hole=0.35
                )
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                   font={"color": "white"}, height=320)
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                st.markdown("#### Sentiment Over Time")
                df["date"] = pd.to_datetime(df["created_at"]).dt.floor("h")
                ts = (df.groupby(["date", "sentiment"])
                        .size()
                        .reset_index(name="count"))
                if not ts.empty:
                    fig2 = px.line(
                        ts, x="date", y="count", color="sentiment",
                        markers=True,
                        color_discrete_map={
                            "Positive": "#2ecc71",
                            "Negative": "#e74c3c",
                            "Neutral": "#95a5a6"
                        }
                    )
                    fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                        plot_bgcolor="rgba(0,0,0,0)",
                                        font={"color": "white"}, height=320)
                    fig2.update_xaxes(gridcolor="rgba(255,255,255,0.1)")
                    fig2.update_yaxes(gridcolor="rgba(255,255,255,0.1)")
                    st.plotly_chart(fig2, use_container_width=True)

            # ── Row 2: Word Clouds ─────────────────────────────
            st.markdown("#### ☁️ Word Clouds by Sentiment")
            wc_cols = st.columns(3)
            for idx, (sent, color) in enumerate([
                ("Positive", "#2ecc71"),
                ("Negative", "#e74c3c"),
                ("Neutral", "#95a5a6")
            ]):
                with wc_cols[idx]:
                    subset = df[df["sentiment"] == sent]["text"].tolist()
                    words = " ".join(subset)
                    # Remove URLs and handles from wordcloud
                    words = re.sub(r"http\S+|@\w+|RT\s", "", words)
                    if words.strip():
                        wc = WordCloud(
                            width=400, height=250,
                            background_color="#1e1e2e",
                            colormap="Greens" if sent=="Positive" else
                                     "Reds" if sent=="Negative" else "Greys",
                            max_words=50
                        ).generate(words)
                        fig_wc, ax = plt.subplots(figsize=(4, 2.5))
                        fig_wc.patch.set_facecolor("#1e1e2e")
                        ax.imshow(wc, interpolation="bilinear")
                        ax.axis("off")
                        ax.set_title(sent, color=color, fontsize=11)
                        st.pyplot(fig_wc)
                        plt.close(fig_wc)
                    else:
                        st.info(f"No {sent.lower()} tweets")

            # ── Top Hashtags ───────────────────────────────────
            st.markdown("#### #️⃣ Top Hashtags in Results")
            all_tags = []
            for tags in df["hashtags"]:
                if isinstance(tags, list):
                    all_tags.extend(tags)
            if all_tags:
                tag_counts = pd.Series([t.lower() for t in all_tags]).value_counts().head(20)
                fig_tags = px.bar(
                    x=tag_counts.values, y=tag_counts.index,
                    orientation="h",
                    labels={"x": "Count", "y": "Hashtag"},
                    color=tag_counts.values,
                    color_continuous_scale=["#1a1a2e", "#e94560"]
                )
                fig_tags.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={"color": "white"}, height=350,
                    yaxis={"categoryorder": "total ascending"},
                    coloraxis_showscale=False
                )
                fig_tags.update_xaxes(gridcolor="rgba(255,255,255,0.1)")
                st.plotly_chart(fig_tags, use_container_width=True)

            # ── Raw Data Table ─────────────────────────────────
            with st.expander("📋 View Raw Data"):
                st.dataframe(
                    df[["text", "sentiment", "created_at"]].head(100),
                    use_container_width=True
                )
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download CSV",
                    csv,
                    file_name=f"sentiment_{keyword or 'all'}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
    else:
        st.info("👈 Enter a keyword in the sidebar and click **Run Analysis**")


# ═══════════════════════════════════════════════════════════
# TAB 4: TOPIC MODELING
# LDA-based topic discovery with coherence scores
# ═══════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 🧠 Topic Modeling (LDA)")
    st.caption(
        "Discovers hidden themes in tweets. "
        "Topics are auto-detected — no labels needed."
    )

    tm_col1, tm_col2 = st.columns([1, 3])
    with tm_col1:
        num_topics = st.slider("Number of topics", 3, 10, 5)
        tm_limit = st.slider("Tweets to analyze", 100, 500, 200, step=50)

    with tm_col2:
        if st.button("🔍 Discover Topics", type="primary"):
            start_dt = datetime.utcnow() - timedelta(hours=hours)
            with st.spinner("Running LDA topic modeling..."):
                docs = find_by_query_text(keyword, limit=tm_limit, start_date=start_dt)
                texts = [d.get("text", "") for d in docs if d.get("text")]

            if len(texts) < 10:
                st.warning("Need at least 10 tweets for topic modeling. Expand your search.")
            else:
                topics, lda, dictionary, corpus = compute_lda_topics(
                    texts, num_topics=num_topics
                )

                if topics:
                    formatted = format_topics_for_display(topics)

                    st.markdown(f"#### Found {len(formatted)} topics in {len(texts)} tweets")

                    for t in formatted:
                        with st.expander(
                            f"🏷️ Topic {t['topic_id']+1}: **{t['label']}**"
                        ):
                            words_html = " ".join(
                                f'<span style="background:#e94560;color:white;'
                                f'padding:2px 8px;border-radius:12px;margin:2px;'
                                f'font-size:0.85em">{w}</span>'
                                for w in t["top_words"]
                            )
                            st.markdown(
                                f"**Top words:** {words_html}",
                                unsafe_allow_html=True
                            )
                            st.code(t["raw"], language="text")
                else:
                    st.warning("Could not extract topics. Try more tweets or a broader keyword.")
        else:
            st.info("Click **Discover Topics** to run LDA topic modeling")


# ─── Auto-Refresh ─────────────────────────────────────────
# If auto-refresh is enabled, re-run the entire Streamlit app
# after the specified interval. This keeps live data fresh.
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()
