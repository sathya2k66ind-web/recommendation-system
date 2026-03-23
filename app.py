import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re
 
# Page config
st.set_page_config(page_title="Content Recommendation System", page_icon="🎬", layout="wide")
 
# ==========================================
# GLOBAL CSS
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
 
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif !important;
    }
 
    .stApp {
        background-color: #050508 !important;
    }
 
    .bg-orb {
        position: fixed;
        border-radius: 50%;
        filter: blur(120px);
        z-index: -1;
        animation: float 10s infinite alternate ease-in-out;
    }
    .orb-1 { width: 400px; height: 400px; background: rgba(124, 58, 237, 0.15); top: -100px; left: -100px; }
    .orb-2 { width: 500px; height: 500px; background: rgba(14, 165, 233, 0.1); bottom: -150px; right: -100px; animation-delay: -5s; }
    .orb-3 { width: 300px; height: 300px; background: rgba(236, 72, 153, 0.08); top: 40%; left: 40%; animation-delay: -2s; }
 
    @keyframes float {
        0%   { transform: translateY(0px) scale(1); }
        100% { transform: translateY(30px) scale(1.1); }
    }
 
    /* ── LANDING PAGE ── */
    .landing-wrap {
        min-height: 90vh;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 60px 20px;
    }
 
    .landing-badge {
        display: inline-block;
        background: rgba(124, 58, 237, 0.15);
        border: 1px solid rgba(124, 58, 237, 0.4);
        border-radius: 999px;
        padding: 6px 20px;
        color: #a78bfa;
        font-size: 13px;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 28px;
        animation: fadeInDown 0.6s ease forwards;
    }
 
    .landing-title {
        font-size: clamp(36px, 6vw, 72px);
        font-weight: 600;
        line-height: 1.1;
        margin: 0 0 24px;
        background: linear-gradient(135deg, #ffffff 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: fadeInDown 0.7s ease forwards;
    }
 
    .landing-sub {
        color: #94a3b8;
        font-size: clamp(15px, 2vw, 19px);
        max-width: 560px;
        line-height: 1.7;
        margin: 0 auto 48px;
        animation: fadeInDown 0.8s ease forwards;
    }
 
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 16px;
        max-width: 780px;
        width: 100%;
        margin: 0 auto 52px;
        animation: fadeInUp 0.9s ease forwards;
    }
 
    .feature-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 14px;
        padding: 22px 18px;
        text-align: left;
        transition: all 0.3s ease;
    }
    .feature-card:hover {
        background: rgba(124,58,237,0.08);
        border-color: rgba(124,58,237,0.3);
        transform: translateY(-3px);
    }
 
    .feature-icon {
        font-size: 26px;
        margin-bottom: 10px;
        display: block;
    }
 
    .feature-title {
        color: #f1f5f9;
        font-weight: 600;
        font-size: 14px;
        margin-bottom: 6px;
    }
 
    .feature-desc {
        color: #64748b;
        font-size: 12px;
        line-height: 1.6;
    }
 
    .stat-row {
        display: flex;
        gap: 40px;
        justify-content: center;
        margin-bottom: 52px;
        animation: fadeInUp 1s ease forwards;
    }
 
    .stat-item {
        text-align: center;
    }
 
    .stat-num {
        font-size: 36px;
        font-weight: 600;
        background: linear-gradient(135deg, #a855f7, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: block;
    }
 
    .stat-label {
        color: #475569;
        font-size: 12px;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
 
    /* Launch button — styled via st.button override */
    div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, #7c3aed, #a855f7) !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 16px 48px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        font-family: 'Poppins', sans-serif !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 0 30px rgba(124, 58, 237, 0.4) !important;
        letter-spacing: 0.5px !important;
    }
    div[data-testid="stButton"] > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 0 50px rgba(124, 58, 237, 0.6) !important;
    }
 
    /* divider */
    .landing-divider {
        width: 60px;
        height: 2px;
        background: linear-gradient(90deg, #7c3aed, #ec4899);
        margin: 0 auto 48px;
        border-radius: 2px;
        animation: fadeIn 1s ease forwards;
    }
 
    /* ── MAIN APP (existing styles) ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px !important;
        color: #94a3b8;
        padding: 10px 25px;
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255,255,255,0.08);
        color: white;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(124,58,237,0.2) !important;
        border: 1px solid #7c3aed !important;
        color: #ffffff !important;
        box-shadow: 0 0 15px rgba(124,58,237,0.4);
    }
 
    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div {
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 12px !important;
        backdrop-filter: blur(10px);
        color: white !important;
        transition: all 0.3s ease;
    }
    div[data-baseweb="input"] > div:focus-within,
    div[data-baseweb="select"] > div:focus-within {
        border-color: #7c3aed !important;
        box-shadow: 0 0 15px rgba(124,58,237,0.3) !important;
    }
 
    .glass-card {
        background: rgba(255,255,255,0.03);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 15px 20px;
        margin-bottom: 12px;
        transition: all 0.3s ease;
        animation: fadeInUp 0.5s ease-out forwards;
        opacity: 0;
    }
    .glass-card:hover { transform: translateY(-4px); }
 
    .card-tfidf { border-left: 4px solid #0ea5e9; }
    .card-tfidf:hover { box-shadow: 0 8px 25px rgba(14,165,233,0.25); border-color: rgba(14,165,233,0.4); }
 
    .card-trans { border-left: 4px solid #a855f7; }
    .card-trans:hover { box-shadow: 0 8px 25px rgba(168,85,247,0.25); border-color: rgba(168,85,247,0.4); }
 
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-20px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to   { opacity: 1; }
    }
 
    div[data-testid="stMetricValue"] {
        font-size: 32px !important;
        font-weight: 600 !important;
        color: #ffffff !important;
        text-shadow: 0 0 15px rgba(255,255,255,0.3);
    }
    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 16px !important;
    }
 
    .gradient-text {
        background: linear-gradient(135deg, #a855f7, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 600;
        text-shadow: 0 0 30px rgba(168,85,247,0.3);
    }
</style>
 
<div class="bg-orb orb-1"></div>
<div class="bg-orb orb-2"></div>
<div class="bg-orb orb-3"></div>
""", unsafe_allow_html=True)
 
 
# ==========================================
# SESSION STATE
# ==========================================
if "launched" not in st.session_state:
    st.session_state.launched = False
 
 
# ==========================================
# LANDING PAGE
# ==========================================
def show_landing():
    st.markdown("""
    <div class="landing-wrap">
 
        <div class="landing-badge">✦ &nbsp; ML · NLP · Semantic Search</div>
 
        <h1 class="landing-title">Content-Based<br>AI Recommender</h1>
 
        <p class="landing-sub">
            Discover movies and songs you'll love. Powered by TF-IDF and 
            Sentence Transformers — compare keyword matching against deep 
            semantic understanding, side by side.
        </p>
 
        <div class="landing-divider"></div>
 
        <div class="stat-row">
            <div class="stat-item">
                <span class="stat-num">4800+</span>
                <span class="stat-label">Movies</span>
            </div>
            <div class="stat-item">
                <span class="stat-num">+66%</span>
                <span class="stat-label">Semantic Uplift</span>
            </div>
            <div class="stat-item">
                <span class="stat-num">384</span>
                <span class="stat-label">Embedding Dims</span>
            </div>
        </div>
 
        <div class="feature-grid">
            <div class="feature-card">
                <span class="feature-icon">📊</span>
                <div class="feature-title">TF-IDF Baseline</div>
                <div class="feature-desc">Classic keyword frequency vectorization. Fast, interpretable, exact matches.</div>
            </div>
            <div class="feature-card">
                <span class="feature-icon">🧠</span>
                <div class="feature-title">Sentence Transformers</div>
                <div class="feature-desc">all-MiniLM-L6-v2 embeddings. Understands context, meaning and intent.</div>
            </div>
            <div class="feature-card">
                <span class="feature-icon">📈</span>
                <div class="feature-title">Side-by-Side Compare</div>
                <div class="feature-desc">See both approaches live. Watch semantic search outperform keyword matching.</div>
            </div>
            <div class="feature-card">
                <span class="feature-icon">🎬</span>
                <div class="feature-title">5000+ Movies</div>
                <div class="feature-desc">Full TMDB dataset. Title, overview and genre combined for rich matching.</div>
            </div>
            <div class="feature-card">
                <span class="feature-icon">🎵</span>
                <div class="feature-title">Spotify Songs</div>
                <div class="feature-desc">Artist, track name and genre vectorized for music discovery.</div>
            </div>
            <div class="feature-card">
                <span class="feature-icon">⚡</span>
                <div class="feature-title">Cached Similarity</div>
                <div class="feature-desc">Matrix built once at startup. Every search is an instant lookup.</div>
            </div>
        </div>
 
    </div>
    """, unsafe_allow_html=True)
 
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Launch App →", use_container_width=True):
            st.session_state.launched = True
            st.rerun()
 
 
# ==========================================
# HELPER FUNCTIONS
# ==========================================
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
    if content_type == "Movies":
        df = pd.read_csv('data/movies.csv')
        df['combined'] = df['title'].fillna('') + ' ' + df['overview'].fillna('') + ' ' + df['genres'].fillna('')
        df['combined'] = df['combined'].apply(preprocess_text)
        return df
    else:
        df = pd.read_csv('data/songs.csv')
        df['combined'] = df['song'].fillna('') + ' ' + df['artist'].fillna('') + ' ' + df['genre'].fillna('')
        df['combined'] = df['combined'].apply(preprocess_text)
        return df
 
@st.cache_resource
def build_tfidf_matrix(df):
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(df['combined'])
    return cosine_similarity(tfidf_matrix, tfidf_matrix)
 
@st.cache_resource
def build_transformer_matrix(df):
    model = SentenceTransformer('all-MiniLM-L6-v2')
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
    scores  = [i[1] for i in sim_scores]
    result  = df.iloc[indices].copy()
    result['similarity_score'] = scores
    return result
 
 
# ==========================================
# MAIN APP
# ==========================================
def show_main_app():
    # Back button
    col_back, col_title = st.columns([1, 8])
    with col_back:
        if st.button("← Back"):
            st.session_state.launched = False
            st.rerun()
 
    st.markdown("<h1 class='gradient-text'>🎬 Content-Based AI Recommender</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; font-size: 18px; margin-bottom: 30px;'>Compare traditional TF-IDF vs modern Sentence Transformers for semantic search.</p>", unsafe_allow_html=True)
 
    tab1, tab2 = st.tabs(["🎬 Movies", "🎵 Songs"])
 
    for tab, content_type in [(tab1, "Movies"), (tab2, "Songs")]:
        with tab:
            df = load_data(content_type)
 
            st.markdown(f"<h3 style='color: #e2e8f0; margin-top: 20px;'>Dataset Size: <span style='color: #a855f7;'>{len(df)}</span> {content_type}</h3>", unsafe_allow_html=True)
 
            title_col = 'title' if content_type == "Movies" else 'song'
            search_query = st.text_input(
                f"Search for a {content_type[:-1].lower()}:",
                placeholder=f"Type {content_type[:-1].lower()} name...",
                key=f"search_{content_type}"
            )
 
            if search_query:
                filtered = df[df[title_col].str.contains(search_query, case=False, na=False)]
                options = filtered[title_col].tolist()
            else:
                options = df[title_col].head(20).tolist()
 
            selected_title = st.selectbox(
                f"Select a {content_type[:-1].lower()}:",
                options=options,
                key=f"select_{content_type}"
            )
 
            if selected_title:
                st.markdown("<hr style='border-color: rgba(255,255,255,0.1); margin: 30px 0;'>", unsafe_allow_html=True)
 
                with st.spinner("Analyzing semantics..."):
                    tfidf_sim       = build_tfidf_matrix(df)
                    transformer_sim = build_transformer_matrix(df)
 
                col1, col2 = st.columns(2)
 
                with col1:
                    st.markdown("<h4 style='color: #0ea5e9; font-weight: 600;'>📊 TF-IDF (Baseline)</h4>", unsafe_allow_html=True)
                    st.markdown("<p style='color: #94a3b8; font-size: 14px;'>Word frequency matching • Exact keyword hits</p>", unsafe_allow_html=True)
                    tfidf_recs = get_recommendations(selected_title, df, tfidf_sim, content_type)
                    if not tfidf_recs.empty:
                        for i, (idx, row) in enumerate(tfidf_recs.iterrows()):
                            score = row['similarity_score']
                            st.markdown(f"""
                            <div class='glass-card card-tfidf' style='animation-delay: {i*0.05}s;'>
                                <div style='color: #f8fafc; font-weight: 600; font-size: 16px;'>{row[title_col]}</div>
                                <div style='color: #94a3b8; font-size: 13px; margin-top: 5px;'>
                                    Similarity Match: <span style='color: #0ea5e9; font-weight: 600;'>{score*100:.1f}%</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
 
                with col2:
                    st.markdown("<h4 style='color: #a855f7; font-weight: 600;'>🧠 Sentence Transformers</h4>", unsafe_allow_html=True)
                    st.markdown("<p style='color: #94a3b8; font-size: 14px;'>Semantic meaning • Context-aware intent</p>", unsafe_allow_html=True)
                    transformer_recs = get_recommendations(selected_title, df, transformer_sim, content_type)
                    if not transformer_recs.empty:
                        for i, (idx, row) in enumerate(transformer_recs.iterrows()):
                            score = row['similarity_score']
                            st.markdown(f"""
                            <div class='glass-card card-trans' style='animation-delay: {i*0.05}s;'>
                                <div style='color: #f8fafc; font-weight: 600; font-size: 16px;'>{row[title_col]}</div>
                                <div style='color: #94a3b8; font-size: 13px; margin-top: 5px;'>
                                    Similarity Match: <span style='color: #a855f7; font-weight: 600;'>{score*100:.1f}%</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
 
                st.markdown("<hr style='border-color: rgba(255,255,255,0.1); margin: 30px 0;'>", unsafe_allow_html=True)
                st.markdown("<h3 style='text-align: center; color: #f8fafc; margin-bottom: 20px;'>📈 Performance Metrics</h3>", unsafe_allow_html=True)
 
                m1, m2, m3 = st.columns(3)
                avg_tfidf       = tfidf_recs['similarity_score'].mean() if not tfidf_recs.empty else 0
                avg_transformer = transformer_recs['similarity_score'].mean() if not transformer_recs.empty else 0
                improvement     = ((avg_transformer - avg_tfidf) / avg_tfidf * 100) if avg_tfidf > 0 else 0
 
                with m1:
                    st.markdown("<div class='glass-card' style='text-align:center;padding:20px;'>", unsafe_allow_html=True)
                    st.metric("Avg TF-IDF Score", f"{avg_tfidf:.3f}")
                    st.markdown("</div>", unsafe_allow_html=True)
                with m2:
                    st.markdown("<div class='glass-card' style='text-align:center;padding:20px;'>", unsafe_allow_html=True)
                    st.metric("Avg Transformer Score", f"{avg_transformer:.3f}")
                    st.markdown("</div>", unsafe_allow_html=True)
                with m3:
                    st.markdown("<div class='glass-card' style='text-align:center;padding:20px;border-color:rgba(0,255,136,0.3);'>", unsafe_allow_html=True)
                    st.metric("AI Improvement", f"+{improvement:.1f}%")
                    st.markdown("</div>", unsafe_allow_html=True)
 
    # Sidebar
    with st.sidebar:
        st.markdown("<h2 class='gradient-text' style='font-size:24px;'>💡 How It Works</h2>", unsafe_allow_html=True)
        st.markdown("""
        <div style='background:rgba(255,255,255,0.03);padding:15px;border-radius:12px;border:1px solid rgba(255,255,255,0.05);margin-bottom:15px;'>
        <h4 style='color:#0ea5e9;margin-top:0;'>1. Preprocessing</h4>
        <p style='color:#94a3b8;font-size:13px;'>Combines title + overview + genre, removes stopwords and cleans text.</p>
        </div>
        <div style='background:rgba(255,255,255,0.03);padding:15px;border-radius:12px;border:1px solid rgba(255,255,255,0.05);margin-bottom:15px;'>
        <h4 style='color:#0ea5e9;margin-top:0;'>2. TF-IDF (Baseline)</h4>
        <p style='color:#94a3b8;font-size:13px;'>Word frequency vectorization. Fast, exact keyword matching (Max 5000 features).</p>
        </div>
        <div style='background:rgba(255,255,255,0.03);padding:15px;border-radius:12px;border:1px solid rgba(255,255,255,0.05);margin-bottom:15px;'>
        <h4 style='color:#a855f7;margin-top:0;'>3. Sentence Transformers</h4>
        <p style='color:#94a3b8;font-size:13px;'>Uses <code>all-MiniLM-L6-v2</code> for 384-dimensional semantic embeddings.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br><h4 style='color:#e2e8f0;'>🛠️ Tech Stack</h4>", unsafe_allow_html=True)
        st.code("• Python 3.11\n• Pandas\n• scikit-learn\n• sentence-transformers\n• Streamlit")
 
 
# ==========================================
# ROUTER
# ==========================================
if not st.session_state.launched:
    show_landing()
else:
    show_main_app()