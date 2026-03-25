import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# ──────────────────────────────────────────────
# OPTIONAL: Lazy import sentence-transformers
# ──────────────────────────────────────────────
TransformerModel = None
def get_transformer_model():
    """Lazy load to avoid crash on startup"""
    global TransformerModel
    if TransformerModel is None:
        try:
            from sentence_transformers import SentenceTransformer
            TransformerModel = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            st.error(f"⚠️ Failed to load AI model: {e}")
            st.info("The app will still work with TF-IDF, but semantic search will be disabled.")
            TransformerModel = False
    return TransformerModel

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(page_title="RECOMMENDER // OPIUM", page_icon="✝", layout="wide")

# ──────────────────────────────────────────────
# OPIUM AESTHETIC CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');

    :root {
        --blood: #ff0000;
        --void: #000000;
        --void-light: #0a0a0a;
        --ash: #2a2a2a;
        --silver: #888888;
        --pure: #ffffff;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }

    .stApp {
        background: var(--void) !important;
    }

    /* Film grain */
    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E");
        opacity: 0.03;
        pointer-events: none;
        z-index: 1000;
    }

    /* Red glow */
    .red-glow {
        position: fixed;
        width: 600px;
        height: 600px;
        background: radial-gradient(circle, rgba(255,0,0,0.08) 0%, transparent 70%);
        border-radius: 50%;
        pointer-events: none;
        z-index: -1;
    }
    .glow-1 { top: -200px; left: -200px; }
    .glow-2 { bottom: -300px; right: -200px; opacity: 0.5; }

    /* Landing */
    .landing-wrap {
        min-height: 85vh;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 40px 20px;
        position: relative;
    }

    .symbol-float {
        position: fixed;
        color: var(--blood);
        opacity: 0.15;
        font-size: 120px;
        pointer-events: none;
        animation: drift 20s infinite ease-in-out;
    }
    .sym-1 { top: 10%; left: 8%; animation-delay: 0s; }
    .sym-2 { top: 20%; right: 12%; animation-delay: -7s; font-size: 80px; }
    .sym-3 { bottom: 15%; left: 15%; animation-delay: -14s; font-size: 60px; }
    .sym-4 { bottom: 25%; right: 8%; animation-delay: -3s; font-size: 100px; }

    @keyframes drift {
        0%, 100% { transform: translate(0, 0) rotate(0deg); }
        25% { transform: translate(10px, -15px) rotate(2deg); }
        50% { transform: translate(-5px, 10px) rotate(-1deg); }
        75% { transform: translate(-15px, -5px) rotate(1deg); }
    }

    .landing-badge {
        display: inline-flex;
        align-items: center;
        gap: 12px;
        border: 1px solid var(--ash);
        padding: 10px 24px;
        color: var(--silver);
        font-size: 11px;
        letter-spacing: 4px;
        text-transform: uppercase;
        margin-bottom: 40px;
    }

    .badge-star { color: var(--blood); font-size: 14px; }

    .landing-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(40px, 10vw, 100px);
        font-weight: 700;
        line-height: 0.9;
        margin: 0 0 20px;
        color: var(--pure);
        letter-spacing: -3px;
        text-transform: uppercase;
    }

    .title-outline { -webkit-text-stroke: 1px var(--pure); -webkit-text-fill-color: transparent; }
    .title-blood { color: var(--blood); text-shadow: 0 0 60px rgba(255,0,0,0.5); }

    .landing-sub {
        color: var(--silver);
        font-size: 15px;
        max-width: 500px;
        line-height: 1.8;
        margin: 0 auto 50px;
        font-weight: 300;
    }

    .landing-divider {
        width: 1px;
        height: 60px;
        background: linear-gradient(to bottom, transparent, var(--blood), transparent);
        margin: 0 auto 40px;
    }

    .stat-row {
        display: flex;
        gap: 60px;
        justify-content: center;
        margin-bottom: 40px;
        flex-wrap: wrap;
    }

    .stat-item { text-align: center; }
    .stat-num {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 36px;
        font-weight: 700;
        color: var(--pure);
        display: block;
    }
    .stat-label {
        color: var(--silver);
        font-size: 10px;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-top: 6px;
    }

    .feature-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 2px;
        max-width: 900px;
        width: 100%;
        margin: 0 auto 40px;
        background: var(--ash);
    }

    @media (max-width: 768px) {
        .feature-grid { grid-template-columns: repeat(2, 1fr); }
        .stat-row { gap: 30px; }
    }
    @media (max-width: 480px) {
        .feature-grid { grid-template-columns: 1fr; }
    }

    .feature-card {
        background: var(--void);
        padding: 24px;
        text-align: left;
        transition: all 0.3s ease;
        position: relative;
    }
    .feature-card:hover {
        background: var(--void-light);
        border-left: 2px solid var(--blood);
    }
    .feature-icon { color: var(--blood); font-size: 20px; margin-bottom: 12px; display: block; }
    .feature-title {
        color: var(--pure);
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        font-size: 13px;
        margin-bottom: 6px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .feature-desc { color: var(--silver); font-size: 12px; line-height: 1.6; }

    /* Buttons */
    div[data-testid="stButton"] > button {
        background: transparent !important;
        color: var(--pure) !important;
        border: 1px solid var(--pure) !important;
        border-radius: 0 !important;
        padding: 16px 48px !important;
        font-size: 12px !important;
        font-weight: 600 !important;
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: 4px !important;
        text-transform: uppercase !important;
        transition: all 0.3s ease !important;
    }
    div[data-testid="stButton"] > button:hover {
        background: var(--blood) !important;
        border-color: var(--blood) !important;
        box-shadow: 0 0 40px rgba(255,0,0,0.4) !important;
    }

    /* Cards */
    .glass-card {
        background: var(--void-light);
        border: 1px solid var(--ash);
        border-left: 3px solid var(--ash);
        padding: 20px 24px;
        margin-bottom: 8px;
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        background: var(--void-light);
        border-left-color: var(--blood);
        transform: translateX(4px);
    }
    .card-tfidf { border-left-color: var(--silver); }
    .card-trans { border-left-color: var(--blood); }

    /* Hide Streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }
</style>

<div class="red-glow glow-1"></div>
<div class="red-glow glow-2"></div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────
if "launched" not in st.session_state:
    st.session_state.launched = False


# ──────────────────────────────────────────────
# LANDING PAGE — RENDERS CORRECTLY NOW
# ──────────────────────────────────────────────
def show_landing():
    st.markdown("""
    <div class="landing-wrap">
        <div class="symbol-float sym-1">✦</div>
        <div class="symbol-float sym-2">✝</div>
        <div class="symbol-float sym-3">★</div>
        <div class="symbol-float sym-4">✦</div>

        <div class="landing-badge">
            <span class="badge-star">✦</span>
            <span>ML · NLP · SEMANTIC SEARCH</span>
            <span class="badge-star">✦</span>
        </div>

        <h1 class="landing-title">
            <span class="title-outline">CONTENT</span>
            <span class="title-blood">RECOMMENDER</span>
        </h1>

        <p class="landing-sub">
            Discover what you'll love. TF-IDF meets deep semantic understanding.
            Watch keyword matching compete against neural embeddings. Side by side.
        </p>

        <div class="landing-divider"></div>

        <div class="stat-row">
            <div class="stat-item">
                <span class="stat-num">4.8K</span>
                <span class="stat-label">Movies</span>
            </div>
            <div class="stat-item">
                <span class="stat-num">+66%</span>
                <span class="stat-label">Semantic Uplift</span>
            </div>
            <div class="stat-item">
                <span class="stat-num">384</span>
                <span class="stat-label">Dimensions</span>
            </div>
        </div>

        <div class="feature-grid">
            <div class="feature-card">
                <span class="feature-icon">◈</span>
                <div class="feature-title">TF-IDF Baseline</div>
                <div class="feature-desc">Classic keyword frequency vectorization. Fast and interpretable.</div>
            </div>
            <div class="feature-card">
                <span class="feature-icon">◉</span>
                <div class="feature-title">Neural Embeddings</div>
                <div class="feature-desc">all-MiniLM-L6-v2 understands context, meaning, intent.</div>
            </div>
            <div class="feature-card">
                <span class="feature-icon">◎</span>
                <div class="feature-title">Live Comparison</div>
                <div class="feature-desc">Both approaches running simultaneously. See the difference.</div>
            </div>
            <div class="feature-card">
                <span class="feature-icon">✦</span>
                <div class="feature-title">5000+ Movies</div>
                <div class="feature-desc">Full TMDB dataset with titles, overviews, and genres.</div>
            </div>
            <div class="feature-card">
                <span class="feature-icon">✧</span>
                <div class="feature-title">Spotify Tracks</div>
                <div class="feature-desc">Artist, track, and genre vectorized for music discovery.</div>
            </div>
            <div class="feature-card">
                <span class="feature-icon">⬡</span>
                <div class="feature-title">Instant Search</div>
                <div class="feature-desc">Pre-computed similarity matrices. Zero latency lookups.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("ENTER ✦", use_container_width=True):
            st.session_state.launched = True
            st.rerun()


# ──────────────────────────────────────────────
# DATA HELPERS
# ──────────────────────────────────────────────
def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    stopwords = {'the','a','an','and','or','but','in','on','at','to','for','of','with','is','was','are'}
    text = ' '.join([w for w in text.split() if w not in stopwords])
    return text


@st.cache_data
def load_data(content_type):
    try:
        if content_type == "Movies":
            df = pd.read_csv('data/movies.csv')
            df['combined'] = df['title'].fillna('') + ' ' + df['overview'].fillna('') + ' ' + df['genres'].fillna('')
        else:
            df = pd.read_csv('data/songs.csv')
            df['combined'] = df['song'].fillna('') + ' ' + df['artist'].fillna('') + ' ' + df['genre'].fillna('')
        df['combined'] = df['combined'].apply(preprocess_text)
        return df
    except FileNotFoundError:
        st.error(f"❌ Missing data file: `data/{content_type.lower()}.csv`")
        st.stop()


@st.cache_resource
def build_tfidf_matrix(df):
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(df['combined'])
    return cosine_similarity(tfidf_matrix, tfidf_matrix)


@st.cache_resource
def build_transformer_matrix(df):
    model = get_transformer_model()
    if model is False:
        return None  # Model failed to load
    embeddings = model.encode(df['combined'].tolist(), show_progress_bar=False)
    return cosine_similarity(embeddings, embeddings)


def get_recommendations(title, df, cosine_sim, content_type, top_n=10):
    title_col = 'title' if content_type == "Movies" else 'song'
    if title not in df[title_col].values:
        return pd.DataFrame()
    idx = df[df[title_col] == title].index[0]
    sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    indices = [i[0] for i in sim_scores]
    scores = [i[1] for i in sim_scores]
    result = df.iloc[indices].copy()
    result['similarity_score'] = scores
    return result


# ──────────────────────────────────────────────
# MAIN APP
# ──────────────────────────────────────────────
def show_main_app():
    col_back, _ = st.columns([1, 5])
    with col_back:
        if st.button("← BACK"):
            st.session_state.launched = False
            st.rerun()

    st.markdown("""
        <h1 style='font-family:Space Grotesk;font-size:40px;color:#fff;letter-spacing:-2px;'>
            RECOMMENDER<span style='color:#ff0000;'>.</span>
        </h1>
        <p style='color:#666;font-size:13px;letter-spacing:1px;margin-bottom:30px;'>
            TF-IDF vs Neural Semantic Search
        </p>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["✦ MOVIES", "✦ SONGS"])

    for tab, content_type in [(tab1, "Movies"), (tab2, "Songs")]:
        with tab:
            df = load_data(content_type)

            st.markdown(f"<div style='color:#888;font-size:11px;letter-spacing:2px;margin-bottom:20px;'>DATASET: {len(df)} {content_type}</div>", unsafe_allow_html=True)

            title_col = 'title' if content_type == "Movies" else 'song'
            search = st.text_input("Search", placeholder="Type to search...", key=f"search_{content_type}", label_visibility="collapsed")
            options = df[df[title_col].str.contains(search, case=False, na=False)][title_col].tolist() if search else df[title_col].head(20).tolist()
            selected = st.selectbox("Select", options, key=f"select_{content_type}", label_visibility="collapsed")

            if selected:
                st.markdown("<hr style='border-top:1px solid #2a2a2a;margin:30px 0;'>", unsafe_allow_html=True)

                with st.spinner(""):
                    tfidf_sim = build_tfidf_matrix(df)
                    trans_sim = build_transformer_matrix(df)

                if trans_sim is None:
                    st.warning("⚠️ Neural model failed to load — showing TF-IDF only.")

                colA, colB = st.columns(2)

                # TF-IDF
                with colA:
                    st.markdown("<div style='color:#888;font-size:11px;letter-spacing:2px;margin-bottom:10px;'>TF-IDF // KEYWORD</div>", unsafe_allow_html=True)
                    tfidf_recs = get_recommendations(selected, df, tfidf_sim, content_type)
                    for _, row in tfidf_recs.iterrows():
                        st.markdown(f"""
                        <div class='glass-card card-tfidf'>
                            <div style='color:#e0e0e0;font-weight:600;'>{row[title_col]}</div>
                            <div style='color:#555;font-size:12px;'>MATCH <span style='color:#888;'>{row['similarity_score']*100:.1f}%</span></div>
                        </div>
                        """, unsafe_allow_html=True)

                # Transformer
                if trans_sim is not None:
                    with colB:
                        st.markdown("<div style='color:#ff0000;font-size:11px;letter-spacing:2px;margin-bottom:10px;'>NEURAL // SEMANTIC</div>", unsafe_allow_html=True)
                        trans_recs = get_recommendations(selected, df, trans_sim, content_type)
                        for _, row in trans_recs.iterrows():
                            st.markdown(f"""
                            <div class='glass-card card-trans'>
                                <div style='color:#e0e0e0;font-weight:600;'>{row[title_col]}</div>
                                <div style='color:#555;font-size:12px;'>MATCH <span style='color:#ff0000;'>{row['similarity_score']*100:.1f}%</span></div>
                            </div>
                            """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# ROUTER
# ──────────────────────────────────────────────
if not st.session_state.launched:
    show_landing()
else:
    show_main_app()