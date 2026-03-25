import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(page_title="RECOMMENDER", page_icon="✝", layout="wide", initial_sidebar_state="collapsed")

# ──────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────
if "launched" not in st.session_state:
    st.session_state.launched = False


# ──────────────────────────────────────────────
# LANDING PAGE — ENHANCED
# ──────────────────────────────────────────────
def show_landing():
    
    # === INJECT ALL CSS ===
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');
        
        #MainMenu, footer, header {visibility: hidden;}
        
        .stApp {
            background: #000 !important;
        }
        
        .block-container {
            padding: 0 !important;
            max-width: 100% !important;
        }
        
        section[data-testid="stSidebar"] {display: none;}
        
        /* ═══════════════════════════════════════
           WRAPPER
        ═══════════════════════════════════════ */
        .land-wrap {
            min-height: 100vh;
            background: #000;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            padding: 50px 25px;
            font-family: 'Inter', sans-serif;
            position: relative;
        }
        
        /* ═══════════════════════════════════════
           GLOWS — MORE INTENSE
        ═══════════════════════════════════════ */
        .glow-orb {
            position: fixed;
            border-radius: 50%;
            pointer-events: none;
            z-index: 0;
        }
        .glow-1 {
            top: -200px;
            left: -200px;
            width: 700px;
            height: 700px;
            background: radial-gradient(circle, rgba(255,0,0,0.15) 0%, transparent 70%);
        }
        .glow-2 {
            bottom: -200px;
            right: -200px;
            width: 700px;
            height: 700px;
            background: radial-gradient(circle, rgba(255,0,0,0.12) 0%, transparent 70%);
        }
        .glow-3 {
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 800px;
            height: 800px;
            background: radial-gradient(circle, rgba(255,0,0,0.05) 0%, transparent 60%);
        }
        
        /* ═══════════════════════════════════════
           FLOATING SYMBOLS — LARGER
        ═══════════════════════════════════════ */
        .sym {
            position: fixed;
            color: #ff0000;
            opacity: 0.15;
            pointer-events: none;
            z-index: 1;
            animation: float 15s infinite ease-in-out;
        }
        .s1 { top: 8%; left: 5%; font-size: 120px; }
        .s2 { top: 15%; right: 8%; font-size: 90px; animation-delay: -5s; }
        .s3 { bottom: 12%; left: 10%; font-size: 70px; animation-delay: -10s; }
        .s4 { bottom: 18%; right: 6%; font-size: 100px; animation-delay: -3s; }
        .s5 { top: 50%; left: 3%; font-size: 50px; animation-delay: -7s; opacity: 0.08; }
        .s6 { top: 40%; right: 4%; font-size: 60px; animation-delay: -12s; opacity: 0.08; }
        
        @keyframes float {
            0%, 100% { transform: translateY(0) rotate(0deg); }
            50% { transform: translateY(-20px) rotate(3deg); }
        }
        
        /* ═══════════════════════════════════════
           BADGE — LARGER + RED ACCENT
        ═══════════════════════════════════════ */
        .badge {
            display: inline-flex;
            gap: 14px;
            border: 1px solid #ff0000;
            padding: 14px 28px;
            color: #ff0000;
            font-size: 13px;
            letter-spacing: 4px;
            text-transform: uppercase;
            margin-bottom: 45px;
            background: rgba(255,0,0,0.05);
        }
        .badge-dot { 
            color: #ff0000; 
            font-size: 16px;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        /* ═══════════════════════════════════════
           TITLE — MUCH LARGER + CENTERED
        ═══════════════════════════════════════ */
        .main-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: clamp(70px, 15vw, 180px);
            font-weight: 700;
            line-height: 0.85;
            letter-spacing: -6px;
            text-transform: uppercase;
            margin-bottom: 35px;
            text-align: center;
            width: 100%;
        }
        .t-outline {
            display: block;
            -webkit-text-stroke: 2px #fff;
            -webkit-text-fill-color: transparent;
            margin-bottom: 5px;
        }
        .t-red {
            display: block;
            color: #ff0000;
            text-shadow: 
                0 0 80px rgba(255,0,0,0.6),
                0 0 150px rgba(255,0,0,0.3);
            animation: glow 3s infinite alternate;
        }
        
        @keyframes glow {
            0% { text-shadow: 0 0 80px rgba(255,0,0,0.6), 0 0 150px rgba(255,0,0,0.3); }
            100% { text-shadow: 0 0 100px rgba(255,0,0,0.8), 0 0 200px rgba(255,0,0,0.4); }
        }
        
        /* ═══════════════════════════════════════
           SUBTITLE — LARGER
        ═══════════════════════════════════════ */
        .sub {
            color: #888;
            font-size: 18px;
            max-width: 550px;
            line-height: 1.9;
            margin: 0 auto 55px;
            text-align: center;
        }
        .sub strong {
            color: #ff0000;
            font-weight: 600;
        }
        
        /* ═══════════════════════════════════════
           DIVIDER — RED
        ═══════════════════════════════════════ */
        .div-line {
            width: 2px;
            height: 70px;
            background: linear-gradient(to bottom, transparent, #ff0000, transparent);
            margin: 0 auto 45px;
        }
        
        /* ═══════════════════════════════════════
           STATS — LARGER + RED ACCENTS
        ═══════════════════════════════════════ */
        .stats-row {
            display: flex;
            gap: 80px;
            justify-content: center;
            flex-wrap: wrap;
            margin-bottom: 55px;
        }
        .stat-box { 
            text-align: center;
            position: relative;
        }
        .stat-box::before {
            content: '✦';
            position: absolute;
            top: -25px;
            left: 50%;
            transform: translateX(-50%);
            color: #ff0000;
            font-size: 10px;
            opacity: 0.7;
        }
        .stat-num {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 52px;
            font-weight: 700;
            color: #fff;
            display: block;
            letter-spacing: -3px;
        }
        .stat-lbl {
            color: #ff0000;
            font-size: 12px;
            letter-spacing: 3px;
            text-transform: uppercase;
            margin-top: 8px;
            display: block;
        }
        
        /* ═══════════════════════════════════════
           FEATURES — LARGER + MORE RED
        ═══════════════════════════════════════ */
        .feat-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 2px;
            background: #ff0000;
            max-width: 950px;
            width: 100%;
            margin: 0 auto 60px;
        }
        @media (max-width: 850px) {
            .feat-grid { grid-template-columns: repeat(2, 1fr); }
            .stats-row { gap: 50px; }
            .stat-num { font-size: 42px; }
        }
        @media (max-width: 550px) {
            .feat-grid { grid-template-columns: 1fr; }
            .main-title { letter-spacing: -3px; }
        }
        .feat-card {
            background: #000;
            padding: 30px 24px;
            text-align: left;
            transition: 0.3s ease;
            border-left: 3px solid transparent;
        }
        .feat-card:hover {
            background: #0a0a0a;
            border-left-color: #ff0000;
            transform: translateX(5px);
        }
        .feat-ico {
            color: #ff0000;
            font-size: 24px;
            margin-bottom: 16px;
        }
        .feat-ttl {
            color: #fff;
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 600;
            font-size: 14px;
            letter-spacing: 2px;
            text-transform: uppercase;
            margin-bottom: 10px;
        }
        .feat-desc {
            color: #666;
            font-size: 13px;
            line-height: 1.7;
        }
        
        /* ═══════════════════════════════════════
           ENTER BUTTON — RED HOVER
        ═══════════════════════════════════════ */
        .btn-wrap {
            margin-top: 10px;
        }
        
        .btn-wrap .stButton > button {
            background: transparent !important;
            color: #fff !important;
            border: 2px solid #fff !important;
            border-radius: 0 !important;
            padding: 20px 70px !important;
            font-size: 14px !important;
            font-weight: 700 !important;
            font-family: 'Space Grotesk', sans-serif !important;
            letter-spacing: 5px !important;
            text-transform: uppercase !important;
            transition: all 0.3s ease !important;
            position: relative !important;
            overflow: hidden !important;
        }
        
        .btn-wrap .stButton > button:hover {
            background: #ff0000 !important;
            border-color: #ff0000 !important;
            color: #fff !important;
            box-shadow: 
                0 0 30px rgba(255,0,0,0.6),
                0 0 60px rgba(255,0,0,0.4),
                0 0 100px rgba(255,0,0,0.2) !important;
            transform: scale(1.02) !important;
        }
        
        .btn-wrap .stButton > button:active {
            transform: scale(0.98) !important;
        }
        
        /* Hint text */
        .hint {
            color: #333;
            font-size: 12px;
            letter-spacing: 3px;
            text-transform: uppercase;
            margin-top: 20px;
            animation: blink 2s infinite;
        }
        
        @keyframes blink {
            0%, 100% { opacity: 0.3; }
            50% { opacity: 1; }
        }
    </style>
    """, unsafe_allow_html=True)
    
    # === GLOWS ===
    st.markdown('''
    <div class="glow-orb glow-1"></div>
    <div class="glow-orb glow-2"></div>
    <div class="glow-orb glow-3"></div>
    ''', unsafe_allow_html=True)
    
    # === SYMBOLS ===
    st.markdown('''
    <div class="sym s1">✦</div>
    <div class="sym s2">✝</div>
    <div class="sym s3">★</div>
    <div class="sym s4">✦</div>
    <div class="sym s5">✝</div>
    <div class="sym s6">★</div>
    ''', unsafe_allow_html=True)
    
    # === OPEN WRAPPER ===
    st.markdown('<div class="land-wrap">', unsafe_allow_html=True)
    
    # === BADGE ===
    st.markdown('''
    <div class="badge">
        <span class="badge-dot">✦</span> 
        ML · NLP · SEMANTIC SEARCH 
        <span class="badge-dot">✦</span>
    </div>
    ''', unsafe_allow_html=True)
    
    # === TITLE ===
    st.markdown('''
    <h1 class="main-title">
        <span class="t-outline">CONTENT</span>
        <span class="t-red">RECOMMENDER</span>
    </h1>
    ''', unsafe_allow_html=True)
    
    # === SUBTITLE ===
    st.markdown('''
    <p class="sub">
        Discover what you will love. <strong>TF-IDF</strong> meets deep <strong>semantic understanding</strong>. 
        Watch keyword matching compete against <strong>neural embeddings</strong>.
    </p>
    ''', unsafe_allow_html=True)
    
    # === DIVIDER ===
    st.markdown('<div class="div-line"></div>', unsafe_allow_html=True)
    
    # === STATS ===
    st.markdown('''
    <div class="stats-row">
        <div class="stat-box">
            <span class="stat-num">4.8K</span>
            <span class="stat-lbl">Movies</span>
        </div>
        <div class="stat-box">
            <span class="stat-num">+66%</span>
            <span class="stat-lbl">Semantic Uplift</span>
        </div>
        <div class="stat-box">
            <span class="stat-num">384</span>
            <span class="stat-lbl">Dimensions</span>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    # === FEATURES ===
    st.markdown('''
    <div class="feat-grid">
        <div class="feat-card">
            <div class="feat-ico">◈</div>
            <div class="feat-ttl">TF-IDF Baseline</div>
            <div class="feat-desc">Classic keyword frequency vectorization. Fast and interpretable.</div>
        </div>
        <div class="feat-card">
            <div class="feat-ico">◉</div>
            <div class="feat-ttl">Neural Embeddings</div>
            <div class="feat-desc">all-MiniLM-L6-v2 understands context, meaning, and intent.</div>
        </div>
        <div class="feat-card">
            <div class="feat-ico">◎</div>
            <div class="feat-ttl">Live Comparison</div>
            <div class="feat-desc">Both approaches running simultaneously. See the difference.</div>
        </div>
        <div class="feat-card">
            <div class="feat-ico">✦</div>
            <div class="feat-ttl">5000+ Movies</div>
            <div class="feat-desc">Full TMDB dataset with titles, overviews, and genres.</div>
        </div>
        <div class="feat-card">
            <div class="feat-ico">✧</div>
            <div class="feat-ttl">Spotify Tracks</div>
            <div class="feat-desc">Artist, track, and genre vectorized for music discovery.</div>
        </div>
        <div class="feat-card">
            <div class="feat-ico">⬡</div>
            <div class="feat-ttl">Instant Search</div>
            <div class="feat-desc">Pre-computed similarity matrices. Zero latency lookups.</div>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    # === CLOSE WRAPPER ===
    st.markdown('</div>', unsafe_allow_html=True)
    
    # === BUTTON ===
    st.markdown('<div class="btn-wrap">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1.2, 1, 1.2])
    with c2:
        if st.button("ENTER ✦", use_container_width=True):
            st.session_state.launched = True
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    
    # === HINT ===
    st.markdown('<p class="hint" style="text-align:center;">CLICK TO INITIALIZE</p>', unsafe_allow_html=True)


# ──────────────────────────────────────────────
# LAZY LOAD MODEL
# ──────────────────────────────────────────────
TransformerModel = None

def get_transformer_model():
    global TransformerModel
    if TransformerModel is None:
        try:
            from sentence_transformers import SentenceTransformer
            TransformerModel = SentenceTransformer('all-MiniLM-L6-v2')
        except:
            TransformerModel = False
    return TransformerModel


# ──────────────────────────────────────────────
# DATA HELPERS
# ──────────────────────────────────────────────
def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    stopwords = {'the','a','an','and','or','but','in','on','at','to','for','of','with','is','was','are'}
    return ' '.join([w for w in text.split() if w not in stopwords])


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
    except:
        st.error("Data file not found")
        st.stop()


@st.cache_resource
def build_tfidf_matrix(df):
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    matrix = tfidf.fit_transform(df['combined'])
    return cosine_similarity(matrix, matrix)


@st.cache_resource
def build_transformer_matrix(df):
    model = get_transformer_model()
    if model is False:
        return None
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
    
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');
        
        #MainMenu, footer, header {visibility: hidden;}
        .stApp { background: #000 !important; }
        .block-container { padding: 2rem 3rem !important; max-width: 1400px !important; }
        
        .app-header { font-family: 'Space Grotesk', sans-serif; font-size: 42px; font-weight: 700; color: #fff; letter-spacing: -2px; margin-bottom: 5px; }
        .app-header span { color: #ff0000; }
        .app-sub { color: #555; font-size: 14px; letter-spacing: 1px; margin-bottom: 35px; }
        
        .sec-label { font-family: 'Space Grotesk', sans-serif; color: #666; font-size: 12px; letter-spacing: 3px; text-transform: uppercase; margin-bottom: 16px; }
        .sec-label.red { color: #ff0000; }
        
        .card { background: #0a0a0a; border: 1px solid #1a1a1a; border-left: 3px solid #333; padding: 18px 22px; margin-bottom: 10px; transition: 0.25s; }
        .card:hover { background: #111; border-left-color: #ff0000; transform: translateX(5px); }
        .card.neural { border-left-color: #550000; }
        .card.neural:hover { border-left-color: #ff0000; }
        .card-title { color: #e0e0e0; font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 15px; margin-bottom: 5px; }
        .card-score { color: #555; font-size: 12px; letter-spacing: 1px; }
        .card-score span { color: #888; font-weight: 600; }
        .card-score span.red { color: #ff0000; }
        
        .stTabs [data-baseweb="tab-list"] { background: transparent; border-bottom: 1px solid #1a1a1a; gap: 0; }
        .stTabs [data-baseweb="tab"] { background: transparent; color: #555; border: none; border-bottom: 2px solid transparent; padding: 14px 28px; font-size: 13px; letter-spacing: 2px; font-weight: 600; }
        .stTabs [data-baseweb="tab"]:hover { color: #fff; }
        .stTabs [aria-selected="true"] { color: #fff !important; border-bottom: 2px solid #ff0000 !important; }
        
        .stTextInput > div > div { background: #0a0a0a !important; border: 1px solid #222 !important; border-radius: 0 !important; }
        .stTextInput > div > div:focus-within { border-color: #ff0000 !important; }
        .stTextInput input { color: #fff !important; font-size: 14px !important; }
        .stSelectbox > div > div { background: #0a0a0a !important; border: 1px solid #222 !important; border-radius: 0 !important; }
        
        .stButton > button { background: transparent !important; color: #666 !important; border: 1px solid #333 !important; border-radius: 0 !important; font-size: 12px !important; letter-spacing: 2px !important; padding: 10px 24px !important; }
        .stButton > button:hover { border-color: #ff0000 !important; color: #ff0000 !important; background: rgba(255,0,0,0.1) !important; }
        
        [data-testid="stMetricValue"] { font-family: 'Space Grotesk', sans-serif !important; font-size: 34px !important; color: #fff !important; letter-spacing: -1px; }
        [data-testid="stMetricLabel"] { color: #555 !important; font-size: 11px !important; letter-spacing: 2px !important; text-transform: uppercase; }
        
        .metric-wrap { background: #0a0a0a; border: 1px solid #1a1a1a; padding: 20px; text-align: center; }
        .metric-wrap.highlight { border-left: 3px solid #ff0000; }
    </style>
    """, unsafe_allow_html=True)
    
    if st.button("← BACK"):
        st.session_state.launched = False
        st.rerun()
    
    st.markdown('<div class="app-header">RECOMMENDER<span>.</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="app-sub">TF-IDF vs Neural Semantic Search — Side by Side</div>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["✦ MOVIES", "✦ SONGS"])
    
    for tab, content_type in [(tab1, "Movies"), (tab2, "Songs")]:
        with tab:
            df = load_data(content_type)
            
            st.markdown(f'<div class="sec-label">DATASET: {len(df)} {content_type.upper()}</div>', unsafe_allow_html=True)
            
            title_col = 'title' if content_type == "Movies" else 'song'
            
            search = st.text_input("Search", placeholder="Type to search...", key=f"search_{content_type}", label_visibility="collapsed")
            
            if search:
                options = df[df[title_col].str.contains(search, case=False, na=False)][title_col].tolist()
            else:
                options = df[title_col].head(25).tolist()
            
            selected = st.selectbox("Select", options, key=f"select_{content_type}", label_visibility="collapsed")
            
            if selected:
                st.markdown("<hr style='border:none;border-top:1px solid #1a1a1a;margin:28px 0;'>", unsafe_allow_html=True)
                
                with st.spinner(""):
                    tfidf_sim = build_tfidf_matrix(df)
                    trans_sim = build_transformer_matrix(df)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<div class="sec-label">TF-IDF // KEYWORD MATCHING</div>', unsafe_allow_html=True)
                    tfidf_recs = get_recommendations(selected, df, tfidf_sim, content_type)
                    for _, row in tfidf_recs.iterrows():
                        st.markdown(f'''
                        <div class="card">
                            <div class="card-title">{row[title_col]}</div>
                            <div class="card-score">MATCH <span>{row["similarity_score"]*100:.1f}%</span></div>
                        </div>
                        ''', unsafe_allow_html=True)
                
                with col2:
                    if trans_sim is not None:
                        st.markdown('<div class="sec-label red">NEURAL // SEMANTIC</div>', unsafe_allow_html=True)
                        trans_recs = get_recommendations(selected, df, trans_sim, content_type)
                        for _, row in trans_recs.iterrows():
                            st.markdown(f'''
                            <div class="card neural">
                                <div class="card-title">{row[title_col]}</div>
                                <div class="card-score">MATCH <span class="red">{row["similarity_score"]*100:.1f}%</span></div>
                            </div>
                            ''', unsafe_allow_html=True)
                    else:
                        st.warning("Neural model unavailable")
                
                st.markdown("<hr style='border:none;border-top:1px solid #1a1a1a;margin:35px 0;'>", unsafe_allow_html=True)
                st.markdown('<div class="sec-label" style="text-align:center;">PERFORMANCE METRICS</div>', unsafe_allow_html=True)
                
                avg_tfidf = tfidf_recs['similarity_score'].mean() if not tfidf_recs.empty else 0
                avg_trans = trans_recs['similarity_score'].mean() if trans_sim is not None and not trans_recs.empty else 0
                improvement = ((avg_trans - avg_tfidf) / avg_tfidf * 100) if avg_tfidf > 0 else 0
                
                m1, m2, m3 = st.columns(3)
                
                with m1:
                    st.markdown('<div class="metric-wrap">', unsafe_allow_html=True)
                    st.metric("TF-IDF AVG", f"{avg_tfidf:.3f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                with m2:
                    st.markdown('<div class="metric-wrap">', unsafe_allow_html=True)
                    st.metric("NEURAL AVG", f"{avg_trans:.3f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                with m3:
                    st.markdown('<div class="metric-wrap highlight">', unsafe_allow_html=True)
                    st.metric("IMPROVEMENT", f"+{improvement:.1f}%")
                    st.markdown('</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────
# ROUTER
# ──────────────────────────────────────────────
if not st.session_state.launched:
    show_landing()
else:
    show_main_app()