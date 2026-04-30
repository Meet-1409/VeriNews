"""
VeriNews - editorial-style Streamlit app.
Run with: streamlit run app.py
"""

import html
import sys
import urllib.parse
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from config.settings import (
    CURSOR_API_KEY,
    NEWS_API_KEY,
    OPENAI_API_KEY,
    PERPLEXITY_API_KEY,
)

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@st.cache_resource
def get_model_components():
    from src.explainability.explainer import FakeNewsExplainer
    from src.inference.predictor import FakeNewsPredictor

    model_path = PROJECT_ROOT / "models" / "fake_news_model.pkl"
    vec_path = PROJECT_ROOT / "models" / "tfidf.pkl"

    if not model_path.exists() or not vec_path.exists():
        raise FileNotFoundError(f"Model files missing: {model_path}, {vec_path}")

    predictor = FakeNewsPredictor(str(model_path), str(vec_path))
    explainer = FakeNewsExplainer(str(model_path), str(vec_path))
    return predictor, explainer


@st.cache_resource
def get_db():
    from src.storage.database import VeriNewsDB

    return VeriNewsDB(str(PROJECT_ROOT / "verinews.db"))


@st.cache_resource
def get_vectorizer_path() -> str:
    return str(PROJECT_ROOT / "models" / "tfidf.pkl")


@st.cache_resource
def get_live_fact_checker():
    from src.research.live_factcheck import LiveFactChecker

    return LiveFactChecker()


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;600;700&family=Manrope:wght@400;500;600;700&display=swap');

        :root {
            --ink: #0f172a;
            --muted: #334155;
            --muted-strong: #475569;
            --gold: #9b7b45;
            --panel: rgba(255, 252, 246, 0.98);
            --line: rgba(123, 94, 43, 0.2);
            --shadow: 0 24px 60px rgba(29, 31, 36, 0.08);
            --green: #166534;
            --red: #b91c1c;
            --amber: #b45309;
        }

        html, body, [class*="css"] {
            font-family: 'Manrope', sans-serif;
            color: var(--ink);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(217, 203, 176, 0.32), transparent 30%),
                radial-gradient(circle at top right, rgba(158, 175, 154, 0.18), transparent 24%),
                linear-gradient(135deg, #f5efe5 0%, #efe6d7 48%, #e8ddcd 100%);
        }

        .main .block-container {
            max-width: 1220px;
            padding: 2rem 2.1rem 3rem 2.1rem;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(29, 34, 39, 0.98), rgba(43, 49, 55, 0.96));
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        section[data-testid="stSidebar"] * {
            color: #f1f5f9;
        }
        section[data-testid="stSidebar"] .stCaptionContainer,
        section[data-testid="stSidebar"] p {
            color: #cbd5e1 !important;
        }

        section[data-testid="stSidebar"] .stButton button {
            background: linear-gradient(135deg, #b8955b, #8e6d3e);
            color: #fff9ef;
            border: none;
            border-radius: 999px;
            font-weight: 700;
            min-height: 2.8rem;
            box-shadow: 0 10px 24px rgba(0, 0, 0, 0.18);
        }

        section[data-testid="stSidebar"] .stRadio label {
            padding: 0.6rem 0.8rem;
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 14px;
            background: rgba(255, 255, 255, 0.04);
        }

        .stTextArea textarea,
        .stSelectbox [data-baseweb="select"] > div {
            border-radius: 18px;
            border: 1px solid var(--line);
            background: rgba(255, 252, 246, 0.82);
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.65);
        }

        .stTextArea textarea {
            min-height: 240px;
            padding: 1rem 1rem 1.1rem 1rem;
            line-height: 1.75;
        }

        .stButton button[kind="primary"] {
            background: linear-gradient(135deg, #171c22, #314051);
            border: none;
            border-radius: 999px;
            font-weight: 700;
            min-height: 3rem;
            box-shadow: 0 14px 28px rgba(24, 32, 40, 0.18);
        }

        .stMetric,
        div[data-testid="metric-container"] {
            background: var(--panel) !important;
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 0.95rem 1rem;
            box-shadow: var(--shadow);
        }
        .stMetric *,
        div[data-testid="metric-container"] * {
            color: #0f172a !important;
        }

        .editorial-shell {
            background: transparent;
            border: none;
            border-radius: 0;
            padding: 0;
            box-shadow: none;
        }

        .hero-card,
        .panel-card,
        .story-card,
        .history-card,
        .fact-card {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 28px;
            box-shadow: var(--shadow);
        }

        .empty-card {
            background: rgba(248, 250, 252, 0.95);
            border: 1px dashed rgba(100, 116, 139, 0.4);
            border-radius: 16px;
            color: #475569 !important;
        }
        .empty-card .muted-copy {
            color: #475569 !important;
        }

        .hero-card {
            padding: 2.3rem 2.4rem;
            background:
                radial-gradient(circle at top right, rgba(155, 123, 69, 0.18), transparent 30%),
                linear-gradient(135deg, rgba(255, 250, 242, 0.98), rgba(249, 242, 231, 0.94));
        }

        .eyebrow {
            text-transform: uppercase;
            letter-spacing: 0.22em;
            font-size: 0.78rem;
            color: var(--gold);
            font-weight: 700;
            margin-bottom: 0.8rem;
        }

        .hero-title,
        .section-title,
        .page-title {
            font-family: 'Cormorant Garamond', serif;
            letter-spacing: -0.02em;
            color: #11161b;
        }

        .hero-title {
            font-size: clamp(2.7rem, 5vw, 4.4rem);
            line-height: 0.92;
            margin-bottom: 0.9rem;
        }

        .hero-copy,
        .muted-copy {
            color: var(--muted-strong);
            font-size: 1rem;
            line-height: 1.8;
        }

        .hero-stat-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.8rem;
            margin-top: 1.4rem;
        }

        .hero-stat {
            padding: 0.8rem 1rem;
            min-width: 140px;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.58);
            border: 1px solid rgba(123, 94, 43, 0.14);
        }

        .hero-stat strong {
            display: block;
            font-size: 1.05rem;
            margin-bottom: 0.12rem;
            color: var(--ink);
        }
        .hero-stat .muted-copy {
            color: var(--muted-strong);
            font-size: 0.9rem;
        }

        .panel-card,
        .fact-card,
        .history-card {
            padding: 1.35rem 1.45rem;
        }

        .section-title,
        .page-title {
            font-size: 2.1rem;
            margin-bottom: 0.2rem;
        }

        .section-kicker {
            color: var(--gold);
            font-size: 0.82rem;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }

        .panel-list {
            margin: 0;
            padding-left: 1rem;
            color: var(--muted-strong);
            line-height: 1.75;
        }

        .story-stack {
            display: grid;
            gap: 0.9rem;
        }

        .story-card,
        .history-card {
            padding: 1.2rem 1.3rem;
        }

        .story-source,
        .chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            align-items: center;
            margin-bottom: 0.55rem;
        }

        .chip,
        .pill {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.32rem 0.7rem;
            border-radius: 999px;
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }

        .chip {
            background: rgba(17, 22, 27, 0.92);
            color: #f8f2e9;
        }

        .pill {
            background: rgba(15, 23, 42, 0.08);
            color: #334155;
            border: 1px solid rgba(15, 23, 42, 0.12);
        }

        .pill.fake {
            background: rgba(141, 59, 57, 0.12);
            color: var(--red);
            border-color: rgba(141, 59, 57, 0.18);
        }

        .pill.real {
            background: rgba(47, 107, 79, 0.12);
            color: var(--green);
            border-color: rgba(47, 107, 79, 0.18);
        }

        .pill.uncertain {
            background: rgba(156, 107, 47, 0.14);
            color: var(--amber);
            border-color: rgba(156, 107, 47, 0.2);
        }

        .story-title {
            color: var(--ink);
            font-weight: 700;
            font-size: 1.08rem;
            margin-bottom: 0.35rem;
        }

        .story-copy {
            color: var(--muted-strong);
            line-height: 1.7;
            margin-bottom: 0.85rem;
        }

        .story-link a,
        .inline-link a {
            color: #6e5228;
            font-weight: 700;
            text-decoration: none;
        }

        .result-banner {
            padding: 1.2rem 1.35rem;
            border-radius: 22px;
            border: 1px solid var(--line);
            margin-bottom: 1rem;
        }

        .result-banner.fake {
            background: linear-gradient(135deg, rgba(141,59,57,0.12), rgba(255,252,246,0.95));
        }

        .result-banner.real {
            background: linear-gradient(135deg, rgba(47,107,79,0.12), rgba(255,252,246,0.95));
        }

        .result-banner.uncertain {
            background: linear-gradient(135deg, rgba(156,107,47,0.12), rgba(255,252,246,0.95));
        }

        .result-title {
            font-weight: 800;
            font-size: 1.15rem;
            margin-bottom: 0.3rem;
            color: var(--ink);
        }
        .result-banner .muted-copy,
        .result-banner .pill {
            color: var(--muted-strong) !important;
        }

        .query-preview {
            background: rgba(255,255,255,0.7);
            border-radius: 20px;
            border: 1px solid rgba(123, 94, 43, 0.15);
            padding: 1rem 1.05rem;
            line-height: 1.8;
            color: var(--ink);
        }

        .divider-space {
            height: 0.5rem;
        }

        [data-testid="stExpander"],
        [data-testid="stAlert"],
        .stAlert,
        [data-baseweb="notification"] {
            color: var(--ink) !important;
        }
        [data-testid="stExpander"] div,
        [data-testid="stAlert"] div,
        .stAlert div {
            color: var(--muted-strong) !important;
        }
        div[data-testid="stMetricValue"],
        div[data-testid="stMetricLabel"],
        label[data-testid="stMetricLabel"],
        label[data-testid="stMetricLabel"] div,
        div[data-testid="metric-container"],
        div[data-testid="metric-container"] label,
        div[data-testid="metric-container"] div {
            color: #0f172a !important;
        }
        [data-testid="stAlert"] p, [data-testid="stAlert"] label,
        [data-testid="stAlert"] div {
            color: #1e293b !important;
        }
        .stSuccess, .stWarning, .stError, .stInfo {
            color: #0f172a !important;
        }
        [data-testid="stAlert"] {
            background-color: rgba(255, 252, 246, 0.95) !important;
        }
        [data-testid="stAlert"] a {
            color: #1d4ed8 !important;
        }

        @media (max-width: 900px) {
            .main .block-container {
                padding: 1rem 0.9rem 2rem 0.9rem;
            }

            .hero-card {
                padding: 1.5rem 1.3rem;
            }

            .hero-title {
                font-size: 2.5rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def format_confidence(value: float) -> str:
    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return "N/A"


def label_class(label: str) -> str:
    key = str(label).strip().lower()
    if key == "fake":
        return "fake"
    if key == "real":
        return "real"
    return "uncertain"


def escape(value: str) -> str:
    return html.escape(value or "")


def render_shell_start() -> None:
    st.markdown('<div class="editorial-shell">', unsafe_allow_html=True)


def render_shell_end() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown(
            """
            <div style="padding-top:0.4rem;">
              <div style="font-family:'Cormorant Garamond', serif; font-size:2.1rem; font-weight:700; line-height:0.9;">VeriNews</div>
              <div style="margin-top:0.45rem; color:#cbd5e1; line-height:1.65;">
                Editorial-grade fake news detection with source-backed verification.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Navigate")
        page = st.radio(
            "Navigation",
            ["Analyze news", "Previous searches"],
            index=0,
            label_visibility="collapsed",
        )

        st.markdown("### Controls")
        if st.button("Refresh App Cache", use_container_width=True):
            st.cache_resource.clear()
            st.success("Cached resources were refreshed.")

        st.markdown("### Tips")
        st.caption(
            "Paste a full headline or paragraph for better classification and source matching."
        )

    return page


def render_hero() -> None:
    hero_col, side_col = st.columns([1.9, 1.1], gap="large")

    with hero_col:
        st.markdown(
            """
            <div class="hero-card">
              <div class="eyebrow">Intelligence Desk</div>
              <div class="hero-title">Verify the story before the story spreads.</div>
              <div class="hero-copy">
                VeriNews blends machine learning, explainable AI, and trusted-source retrieval into a single
                newsroom-inspired workspace. Paste a headline or article to get a decision, understand why it was made,
                and compare it against reputable coverage.
              </div>
              <div class="hero-stat-row">
                <div class="hero-stat">
                  <strong>ML Classification</strong>
                  <span class="muted-copy">Fake, real, or uncertain</span>
                </div>
                <div class="hero-stat">
                  <strong>Transparent Evidence</strong>
                  <span class="muted-copy">Word-level influence and sources</span>
                </div>
                <div class="hero-stat">
                  <strong>Search Archive</strong>
                  <span class="muted-copy">Saved verification history</span>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with side_col:
        render_trending_panel()


def render_trending_panel() -> None:
    st.markdown(
        """
        <div class="panel-card">
          <div class="section-kicker">Trusted Desk</div>
          <div class="section-title">Trending verified coverage</div>
          <div class="muted-copy">A quick view into current source material from reputable outlets.</div>
        """,
        unsafe_allow_html=True,
    )

    try:
        from src.retrieval.news_fetcher import fetch_verified_news

        trending_articles = fetch_verified_news("world headlines verified news", page_size=4)
    except Exception:
        trending_articles = []

    if not trending_articles:
        trending_articles = [
            {
                "title": "Reuters World News",
                "description": "Verified coverage from Reuters.",
                "url": "https://www.reuters.com/world/",
                "source": "Reuters",
            },
            {
                "title": "BBC News - World",
                "description": "Top world stories from BBC News.",
                "url": "https://www.bbc.com/news/world",
                "source": "BBC News",
            },
        ]

    st.markdown('<div class="story-stack">', unsafe_allow_html=True)
    for article in trending_articles[:3]:
        title = escape(article.get("title", "Untitled story"))
        desc = escape(article.get("description", "Verified reporting from a trusted source."))
        source = escape(article.get("source", "Trusted source"))
        url = article.get("url", "#")
        st.markdown(
            f"""
            <div class="story-card">
              <div class="story-source"><span class="chip">{source}</span></div>
              <div class="story-title">{title}</div>
              <div class="story-copy">{desc}</div>
              <div class="story-link"><a href="{url}" target="_blank">Open coverage</a></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div></div>", unsafe_allow_html=True)


def render_input_panel() -> tuple[str, bool]:
    input_col, guide_col = st.columns([1.55, 1], gap="large")

    with input_col:
        st.markdown(
            """
            <div class="panel-card">
              <div class="section-kicker">Analysis Workspace</div>
              <div class="section-title">Run a verification</div>
              <div class="muted-copy">Paste a headline, social post, or article excerpt. Longer, clearer inputs usually produce better retrieval and explanation quality.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        news_text = st.text_area(
            "Enter a news article or headline",
            height=260,
            placeholder="Example: Officials announce a major policy shift after reports of a confidential memo...",
        )
        analyze_clicked = st.button("Analyze Story", type="primary", use_container_width=True)

    with guide_col:
        st.markdown(
            """
            <div class="panel-card">
              <div class="section-kicker">Method</div>
              <div class="section-title">How the check works</div>
              <ol class="panel-list">
                <li>Text is cleaned and transformed with TF-IDF.</li>
                <li>A trained classifier predicts fake, real, or uncertain.</li>
                <li>LIME highlights the words influencing that decision.</li>
                <li>Trusted coverage is retrieved and ranked by relevance.</li>
              </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="panel-card">
              <div class="section-kicker">Prompt Ideas</div>
              <div class="section-title">Good starting inputs</div>
              <div class="muted-copy">
                Government announces new vaccination drive in rural areas.<br><br>
                Celebrity involved in secret space mission leaks.<br><br>
                Files reveal new details about a public investigation.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return news_text, analyze_clicked


def render_result_banner(result: dict) -> None:
    label = result.get("label", "Uncertain")
    confidence = format_confidence(result.get("confidence", 0))
    status_class = label_class(label)
    messages = {
        "fake": "This claim shows strong signals commonly associated with misinformation.",
        "real": "This claim aligns more closely with patterns found in legitimate reporting.",
        "uncertain": "The model is cautious here. Use the supporting sources before drawing a conclusion.",
    }
    st.markdown(
        f"""
        <div class="result-banner {status_class}">
          <div class="chip-row">
            <span class="pill {status_class}">{escape(label)}</span>
            <span class="pill">Confidence {escape(confidence)}</span>
          </div>
          <div class="result-title">Model assessment</div>
          <div class="muted-copy">{messages[status_class]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_explanation_section(explainer, news_text: str, label: str) -> None:
    st.markdown(
        """
        <div class="panel-card">
          <div class="section-kicker">Explainability</div>
          <div class="section-title">Why the model leaned this way</div>
          <div class="muted-copy">The chart below shows the words that most strongly influenced the current decision.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        explanation = explainer.explain(news_text)
        if not explanation:
            st.info("Explanation data is not available for this input.")
            return

        words = [word for word, _ in explanation]
        weights = [float(weight) for _, weight in explanation]
        colors = ["#8d3b39" if value < 0 else "#2f6b4f" for value in weights]
        fig = go.Figure(
            go.Bar(
                x=weights,
                y=words,
                orientation="h",
                marker_color=colors,
                hovertemplate="%{y}: %{x:.3f}<extra></extra>",
            )
        )
        fig.update_layout(
            height=360,
            margin=dict(l=120, r=20, t=30, b=50),
            paper_bgcolor="rgba(255,255,255,0.95)",
            plot_bgcolor="rgba(255,252,246,0.9)",
            font=dict(color="#0f172a", size=13),
            xaxis=dict(
                title=f"Influence score ({'negative → fake' if label == 'Fake' else 'relative contribution'})",
                title_font=dict(color="#0f172a", size=12),
                tickfont=dict(color="#0f172a", size=11),
                gridcolor="rgba(15,23,42,0.1)",
            ),
            yaxis=dict(
                autorange="reversed",
                tickfont=dict(color="#0f172a", size=12),
                title_font=dict(color="#0f172a"),
            ),
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as ex:
        st.warning("Explanation could not be generated for this input.")
        st.exception(ex)


# Demo/placeholder images when articles have no image_url (for slideshow-style previews)
DEMO_ARTICLE_IMAGES = [
    "https://images.unsplash.com/photo-1504711434969-e33886168f5c?w=400&q=80",
    "https://images.unsplash.com/photo-1495020689067-958852a7765e?w=400&q=80",
    "https://images.unsplash.com/photo-1585829365295-ab7cd400c167?w=400&q=80",
]


def _demo_image_for_index(index: int) -> str:
    return DEMO_ARTICLE_IMAGES[index % len(DEMO_ARTICLE_IMAGES)]


def render_verified_articles(db, query_id: int, news_text: str) -> None:
    st.markdown(
        """
        <div class="panel-card">
          <div class="section-kicker">Source Comparison</div>
          <div class="section-title">Trusted reporting related to this claim</div>
          <div class="muted-copy">Coverage is retrieved from reputable outlets and ranked by similarity to the submitted text.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        from src.retrieval.news_fetcher import fetch_verified_news
        from src.retrieval.similarity_ranker import rank_articles_by_similarity

        articles = fetch_verified_news(news_text)
        ranked = rank_articles_by_similarity(
            news_text,
            articles,
            vectorizer_path=get_vectorizer_path(),
        )
    except Exception:
        ranked = []

    if not ranked:
        st.info("Live source retrieval is unavailable right now. You can still continue with a manual web search.")
        return

    db.save_verified_articles(query_id, ranked)

    for idx, article in enumerate(ranked):
        image_col, content_col = st.columns([1, 1.8], gap="medium")
        with image_col:
            img_url = article.get("image_url") or _demo_image_for_index(idx)
            try:
                st.image(img_url, use_container_width=True)
            except Exception:
                st.markdown(
                    f'<div class="empty-card" style="min-height:180px; display:flex; align-items:center; justify-content:center;"><span class="muted-copy">No preview</span></div>',
                    unsafe_allow_html=True,
                )
        with content_col:
            title = escape(article.get("title") or "Untitled coverage")
            description = escape(article.get("description") or "No description available.")
            source = escape(article.get("source") or "Trusted source")
            similarity = article.get("similarity_score", 0)
            try:
                similarity = round(float(similarity), 2)
            except (TypeError, ValueError):
                similarity = 0.0
            url = article.get("url") or "#"
            st.markdown(
                f"""
                <div class="story-card">
                  <div class="chip-row">
                    <span class="chip">{source}</span>
                    <span class="pill">Similarity {similarity:.2f}</span>
                  </div>
                  <div class="story-title">{title}</div>
                  <div class="story-copy">{description}</div>
                  <div class="story-link"><a href="{url}" target="_blank">Read source article</a></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Contradiction check: does any retrieved source contradict the claim? (Novel feature.)
    if OPENAI_API_KEY and ranked:
        from src.research.live_factcheck import check_contradiction
        contradiction = check_contradiction(news_text, ranked, OPENAI_API_KEY)
        if contradiction:
            st.markdown(
                """
                <div class="panel-card" style="margin-top:1rem;">
                  <div class="section-kicker">New in VeriNews</div>
                  <div class="section-title">Contradiction check</div>
                  <div class="muted-copy">We asked whether any of the trusted sources above contradict your claim.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.info(contradiction)

    encoded_q = urllib.parse.quote(news_text)
    google_url = f"https://www.google.com/search?q={encoded_q}+news"
    st.markdown(
        f'<div class="inline-link" style="margin-top:0.7rem;"><a href="{google_url}" target="_blank">Open broader news search</a></div>',
        unsafe_allow_html=True,
    )


def render_live_factcheck(news_text: str, result: dict) -> None:
    if not OPENAI_API_KEY:
        return

    st.markdown(
        """
        <div class="panel-card">
          <div class="section-kicker">Model + ChatGPT</div>
          <div class="section-title">Combined verdict</div>
          <div class="muted-copy">The verdict above is from our ML model. Below, trusted news is fetched and summarized by ChatGPT so you get both the model’s classification and an AI evidence brief.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        checker = get_live_fact_checker()
        with st.spinner("Reviewing live evidence and drafting the brief..."):
            live = checker.analyze(news_text, result)
    except Exception as ex:
        st.warning("Live fact-check is unavailable right now.")
        st.exception(ex)
        return

    evidence = live.get("evidence", {})
    ppl = live.get("perplexity", {})
    gpt = live.get("openai", {})

    metric_col1, metric_col2 = st.columns(2)
    with metric_col1:
        st.metric("Live Verdict", str(gpt.get("verdict", "N/A")))
    with metric_col2:
        st.metric("Live Confidence", format_confidence(gpt.get("confidence", 0)))

    summary = (gpt.get("summary") or gpt.get("refined_summary") or "").strip()
    if summary:
        st.markdown(
            f"""
            <div class="fact-card">
              <div class="section-kicker">Summary</div>
              <div class="muted-copy">{escape(summary)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    key_points = gpt.get("key_points") or []
    if key_points:
        items = "".join(f"<li>{escape(point)}</li>" for point in key_points)
        st.markdown(
            f"""
            <div class="fact-card">
              <div class="section-kicker">Key Findings</div>
              <ol class="panel-list">{items}</ol>
            </div>
            """,
            unsafe_allow_html=True,
        )

    citations = evidence.get("citations", [])
    if citations:
        links = "".join(
            f'<li><a href="{url}" target="_blank">{escape(url)}</a></li>'
            for url in citations[:6]
        )
        st.markdown(
            f"""
            <div class="fact-card">
              <div class="section-kicker">Evidence Links</div>
              <ol class="panel-list">{links}</ol>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if ppl.get("available") and ppl.get("summary"):
        st.markdown(
            f"""
            <div class="fact-card">
              <div class="section-kicker">Perplexity Notes</div>
              <div class="muted-copy">{escape(ppl.get("summary"))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if not gpt.get("available") and not summary and not key_points and not citations:
        st.warning("The AI brief could not be generated for this request.")


def render_analysis_page() -> None:
    render_hero()
    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
    news_text, analyze_clicked = render_input_panel()

    if not analyze_clicked:
        return

    if not news_text.strip():
        st.warning("Please enter a headline or article before analyzing.")
        return

    try:
        with st.spinner("Loading model and verification services..."):
            predictor, explainer = get_model_components()
            db = get_db()

        with st.spinner("Analyzing the submitted story..."):
            result = predictor.predict(news_text)
            query_id = db.save_query(news_text, result["label"], result["confidence"])
    except Exception as ex:
        st.error("The analysis pipeline could not be started.")
        st.exception(ex)
        return

    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
    render_result_banner(result)

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        st.metric("Classification", result.get("label", "N/A"))
    with metric_col2:
        st.metric("Confidence", format_confidence(result.get("confidence", 0)))
    with metric_col3:
        st.metric("Input Length", f"{len(news_text.split())} words")

    st.markdown(
        f"""
        <div class="panel-card" style="margin-top:0.8rem;">
          <div class="section-kicker">Submitted Text</div>
          <div class="query-preview">{escape(news_text)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
    render_explanation_section(explainer, news_text, result.get("label", "Uncertain"))

    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
    render_verified_articles(db, query_id, news_text)

    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
    render_live_factcheck(news_text, result)

    # User feedback: "Was this helpful?" – stored for improvement (novel feature).
    st.markdown(
        """
        <div class="panel-card" style="margin-top:1rem;">
          <div class="section-kicker">New in VeriNews</div>
          <div class="section-title">Was this result helpful?</div>
          <div class="muted-copy">Your feedback helps us improve. It is stored locally and not shared.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    fb_col1, fb_col2, _ = st.columns([1, 1, 2])
    with fb_col1:
        if st.button("Yes, helpful", key="feedback_yes", use_container_width=True):
            db.save_feedback(query_id, "yes")
            st.success("Thanks for your feedback!")
    with fb_col2:
        if st.button("No, not helpful", key="feedback_no", use_container_width=True):
            db.save_feedback(query_id, "no")
            st.info("We'll use this to improve.")


def render_history_page() -> None:
    st.markdown(
        """
        <div class="panel-card">
          <div class="section-kicker">Archive</div>
          <div class="page-title">Previous searches</div>
          <div class="muted-copy">Review saved classifications and reopen the supporting source material connected to each verification.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    db = get_db()
    queries = db.get_recent_queries(limit=50)

    if not queries:
        st.markdown(
            """
            <div class="empty-card" style="margin-top:1rem;">
              <div class="section-title">No saved searches yet</div>
              <div class="muted-copy">Run a verification from the main workspace and it will appear here with its related source material.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    options = {
        f"{q['created_at'][:16]} | {q['label']:<9} | {q['query_text'][:65]}": q
        for q in queries
    }
    selected_label = st.selectbox(
        "Select a saved verification",
        list(options.keys()),
        label_visibility="collapsed",
    )
    selected = options[selected_label]

    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
    status_class = label_class(selected["label"])
    st.markdown(
        f"""
        <div class="history-card">
          <div class="chip-row">
            <span class="pill {status_class}">{escape(selected['label'])}</span>
            <span class="pill">Saved {escape(selected['created_at'][:10])}</span>
            <span class="pill">Confidence {escape(format_confidence(selected['confidence']))}</span>
          </div>
          <div class="section-title" style="font-size:1.7rem;">Saved verification snapshot</div>
          <div class="query-preview" style="margin-top:0.8rem;">{escape(selected['query_text'])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    articles = db.get_verified_articles_for_query(selected["id"])
    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="panel-card">
          <div class="section-kicker">Stored Sources</div>
          <div class="section-title">Related articles</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not articles:
        st.info("No verified articles were stored for this search.")
        return

    for idx, article in enumerate(articles):
        title = escape(article.get("title", "Untitled coverage"))
        description = escape(article.get("description", "No description stored for this article."))
        source = escape(article.get("source", "Trusted source"))
        similarity = article.get("similarity_score", 0)
        url = article.get("url", "#")
        image_url = article.get("image_url") or _demo_image_for_index(idx)

        image_col, content_col = st.columns([1, 1.8], gap="medium")
        with image_col:
            try:
                st.image(image_url, use_container_width=True)
            except Exception:
                st.markdown(
                    '<div class="empty-card" style="min-height:180px; display:flex; align-items:center; justify-content:center;"><span class="muted-copy">No preview</span></div>',
                    unsafe_allow_html=True,
                )
        with content_col:
            st.markdown(
                f"""
                <div class="story-card">
                  <div class="chip-row">
                    <span class="chip">{source}</span>
                    <span class="pill">Similarity {similarity:.2f}</span>
                  </div>
                  <div class="story-title">{title}</div>
                  <div class="story-copy">{description}</div>
                  <div class="story-link"><a href="{url}" target="_blank">Read full article</a></div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def main() -> None:
    st.set_page_config(page_title="VeriNews", page_icon="VN", layout="wide")
    inject_styles()
    page = render_sidebar()
    render_shell_start()
    if page == "Analyze news":
        render_analysis_page()
    else:
        render_history_page()
    render_shell_end()


if __name__ == "__main__":
    main()
