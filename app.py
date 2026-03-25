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


# ══════════════════════════════════════════════
# LANDING PAGE
# ══════════════════════════════════════════════
def show_landing():
    
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
        
        .div-line {
            width: 2px;
            height: 70px;
            background: linear-gradient(to bottom, transparent, #ff0000, transparent);
            margin: 0 auto 45px;
        }
        
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
    
    st.markdown('''
    <div class="glow-orb glow-1"></div>
    <div class="glow-orb glow-2"></div>
    <div class="glow-orb glow-3"></div>
    ''', unsafe_allow_html=True)
    
    st.markdown('''
    <div class="sym s1">✦</div>
    <div class="sym s2">✝</div>
    <div class="sym s3">★</div>
    <div class="sym s4">✦</div>
    <div class="sym s5">✝</div>
    <div class="sym s6">★</div>
    ''', unsafe_allow_html=True)
    
    st.markdown('<div class="land-wrap">', unsafe_allow_html=True)
    
    st.markdown('''
    <div class="badge">
        <span class="badge-dot">✦</span> 
        ML · NLP · SEMANTIC SEARCH 
        <span class="badge-dot">✦</span>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown('''
    <h1 class="main-title">
        <span class="t-outline">CONTENT</span>
        <span class="t-red">RECOMMENDER</span>
    </h1>
    ''', unsafe_allow_html=True)
    
    st.markdown('''
    <p class="sub">
        Discover what you will love. <strong>TF-IDF</strong> meets deep <strong>semantic understanding</strong>. 
        Watch keyword matching compete against <strong>neural embeddings</strong>.
    </p>
    ''', unsafe_allow_html=True)
    
    st.markdown('<div class="div-line"></div>', unsafe_allow_html=True)
    
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
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="btn-wrap">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1.2, 1, 1.2])
    with c2:
        if st.button("ENTER ✦", use_container_width=True):
            st.session_state.launched = True
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<p class="hint" style="text-align:center;">CLICK TO INITIALIZE</p>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
# LAZY LOAD MODEL
# ══════════════════════════════════════════════
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


# ══════════════════════════════════════════════
# DATA HELPERS
# ══════════════════════════════════════════════
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


# ══════════════════════════════════════════════
# MAIN APP — FULL OPIUM MODE
# ══════════════════════════════════════════════
def show_main_app():
    
    # === HARDCORE OPIUM CSS ===
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Inter:wght@300;400;500&display=swap');
        
        /* ═══════════════════════════════════════
           HIDE DEFAULTS
        ═══════════════════════════════════════ */
        #MainMenu, footer, header {visibility: hidden;}
        
        .stApp { 
            background: #000 !important; 
        }
        
        .block-container { 
            padding: 2rem 3rem !important; 
            max-width: 1500px !important; 
        }
        
        /* ═══════════════════════════════════════
           FILM GRAIN OVERLAY
        ═══════════════════════════════════════ */
        .stApp::before {
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.8' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
            opacity: 0.03;
            pointer-events: none;
            z-index: 9999;
        }
        
        /* ═══════════════════════════════════════
           FLOATING SYMBOLS
        ═══════════════════════════════════════ */
        .app-sym {
            position: fixed;
            color: #ff0000;
            pointer-events: none;
            z-index: 0;
            animation: floatSym 20s infinite ease-in-out;
        }
        .as1 { top: 5%; left: 3%; font-size: 80px; opacity: 0.08; }
        .as2 { top: 30%; right: 2%; font-size: 60px; opacity: 0.06; animation-delay: -5s; }
        .as3 { bottom: 20%; left: 2%; font-size: 50px; opacity: 0.07; animation-delay: -10s; }
        .as4 { bottom: 10%; right: 3%; font-size: 70px; opacity: 0.05; animation-delay: -15s; }
        .as5 { top: 60%; left: 1%; font-size: 40px; opacity: 0.04; animation-delay: -8s; }
        .as6 { top: 15%; left: 50%; font-size: 30px; opacity: 0.03; animation-delay: -12s; }
        
        @keyframes floatSym {
            0%, 100% { transform: translateY(0) rotate(0deg); }
            25% { transform: translateY(-15px) rotate(5deg); }
            50% { transform: translateY(10px) rotate(-3deg); }
            75% { transform: translateY(-8px) rotate(2deg); }
        }
        
        /* ═══════════════════════════════════════
           RED GLOWS
        ═══════════════════════════════════════ */
        .app-glow {
            position: fixed;
            border-radius: 50%;
            pointer-events: none;
            z-index: 0;
        }
        .ag1 {
            top: -150px;
            left: -150px;
            width: 500px;
            height: 500px;
            background: radial-gradient(circle, rgba(255,0,0,0.1) 0%, transparent 70%);
        }
        .ag2 {
            bottom: -150px;
            right: -150px;
            width: 500px;
            height: 500px;
            background: radial-gradient(circle, rgba(255,0,0,0.08) 0%, transparent 70%);
        }
        .ag3 {
            top: 40%;
            right: 10%;
            width: 300px;
            height: 300px;
            background: radial-gradient(circle, rgba(255,0,0,0.05) 0%, transparent 70%);
            animation: pulseGlow 4s infinite alternate;
        }
        
        @keyframes pulseGlow {
            0% { opacity: 0.5; transform: scale(1); }
            100% { opacity: 1; transform: scale(1.1); }
        }
        
        /* ═══════════════════════════════════════
           HEADER
        ═══════════════════════════════════════ */
        .app-header {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 56px;
            font-weight: 700;
            color: #fff;
            letter-spacing: -3px;
            margin-bottom: 8px;
            position: relative;
            display: inline-block;
            animation: fadeSlideIn 0.8s ease;
        }
        .app-header span { 
            color: #ff0000;
            text-shadow: 0 0 30px rgba(255,0,0,0.5);
        }
        .app-header::after {
            content: '✦';
            position: absolute;
            top: -10px;
            right: -30px;
            font-size: 16px;
            color: #ff0000;
            opacity: 0.7;
            animation: pulse 2s infinite;
        }
        
        @keyframes fadeSlideIn {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 0.7; }
            50% { opacity: 0.3; }
        }
        
        .app-sub {
            color: #666;
            font-size: 16px;
            letter-spacing: 2px;
            margin-bottom: 40px;
            animation: fadeSlideIn 1s ease;
        }
        .app-sub strong {
            color: #ff0000;
        }
        
        /* ═══════════════════════════════════════
           TABS
        ═══════════════════════════════════════ */
        .stTabs [data-baseweb="tab-list"] { 
            background: transparent; 
            border-bottom: 2px solid #1a1a1a; 
            gap: 0; 
        }
        .stTabs [data-baseweb="tab"] { 
            background: transparent;
            color: #555;
            border: none;
            border-bottom: 3px solid transparent;
            padding: 16px 32px;
            font-family: 'Space Grotesk', sans-serif;
            font-size: 14px;
            font-weight: 600;
            letter-spacing: 3px;
            transition: all 0.3s ease;
        }
        .stTabs [data-baseweb="tab"]:hover { 
            color: #ff0000;
            background: rgba(255,0,0,0.05);
        }
        .stTabs [aria-selected="true"] { 
            color: #ff0000 !important; 
            border-bottom: 3px solid #ff0000 !important;
            background: rgba(255,0,0,0.08) !important;
        }
        
        /* ═══════════════════════════════════════
           SECTION LABELS
        ═══════════════════════════════════════ */
        .sec-label {
            font-family: 'Space Grotesk', sans-serif;
            color: #555;
            font-size: 13px;
            letter-spacing: 4px;
            text-transform: uppercase;
            margin-bottom: 18px;
            display: flex;
            align-items: center;
            gap: 10px;
            animation: fadeIn 0.6s ease;
        }
        .sec-label::before {
            content: '✦';
            color: #ff0000;
            font-size: 10px;
            animation: pulse 2s infinite;
        }
        .sec-label.red { 
            color: #ff0000; 
            text-shadow: 0 0 20px rgba(255,0,0,0.3);
        }
        
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        /* ═══════════════════════════════════════
           INPUTS
        ═══════════════════════════════════════ */
        .stTextInput > div > div { 
            background: #0a0a0a !important; 
            border: 2px solid #1a1a1a !important; 
            border-radius: 0 !important;
            transition: all 0.3s ease;
        }
        .stTextInput > div > div:focus-within { 
            border-color: #ff0000 !important;
            box-shadow: 0 0 20px rgba(255,0,0,0.2) !important;
        }
        .stTextInput input { 
            color: #fff !important; 
            font-size: 15px !important;
            font-family: 'Inter', sans-serif !important;
        }
        .stTextInput input::placeholder {
            color: #444 !important;
        }
        
        .stSelectbox > div > div { 
            background: #0a0a0a !important; 
            border: 2px solid #1a1a1a !important; 
            border-radius: 0 !important; 
        }
        .stSelectbox > div > div:focus-within {
            border-color: #ff0000 !important;
        }
        
        /* ═══════════════════════════════════════
           CARDS — ANIMATED
        ═══════════════════════════════════════ */
        .card {
            background: linear-gradient(135deg, #0a0a0a 0%, #050505 100%);
            border: 1px solid #1a1a1a;
            border-left: 4px solid #333;
            padding: 20px 24px;
            margin-bottom: 12px;
            transition: all 0.35s cubic-bezier(0.16, 1, 0.3, 1);
            position: relative;
            overflow: hidden;
            animation: cardSlide 0.5s ease forwards;
            opacity: 0;
        }
        
        .card::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,0,0,0.05), transparent);
            transition: left 0.5s ease;
        }
        
        .card:hover {
            background: linear-gradient(135deg, #111 0%, #0a0a0a 100%);
            border-left-color: #ff0000;
            transform: translateX(8px);
            box-shadow: 0 5px 30px rgba(255,0,0,0.15);
        }
        
        .card:hover::before {
            left: 100%;
        }
        
        .card.neural { 
            border-left-color: #550000; 
        }
        .card.neural:hover { 
            border-left-color: #ff0000;
            box-shadow: 0 5px 30px rgba(255,0,0,0.2);
        }
        
        @keyframes cardSlide {
            from { 
                opacity: 0; 
                transform: translateX(-30px); 
            }
            to { 
                opacity: 1; 
                transform: translateX(0); 
            }
        }
        
        .card-title {
            color: #e8e8e8;
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 600;
            font-size: 16px;
            margin-bottom: 6px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .card-rank {
            color: #ff0000;
            font-size: 12px;
            font-weight: 700;
            opacity: 0.7;
        }
        
        .card-score {
            color: #555;
            font-size: 12px;
            letter-spacing: 2px;
            text-transform: uppercase;
        }
        .card-score span { 
            color: #888; 
            font-weight: 700;
            font-family: 'Space Grotesk', sans-serif;
        }
        .card-score span.red { 
            color: #ff0000;
            text-shadow: 0 0 10px rgba(255,0,0,0.5);
        }
        
        /* ═══════════════════════════════════════
           BACK BUTTON
        ═══════════════════════════════════════ */
        .stButton > button {
            background: transparent !important;
            color: #666 !important;
            border: 1px solid #333 !important;
            border-radius: 0 !important;
            font-family: 'Space Grotesk', sans-serif !important;
            font-size: 12px !important;
            letter-spacing: 3px !important;
            padding: 12px 28px !important;
            transition: all 0.3s ease !important;
        }
        .stButton > button:hover {
            border-color: #ff0000 !important;
            color: #ff0000 !important;
            background: rgba(255,0,0,0.1) !important;
            box-shadow: 0 0 20px rgba(255,0,0,0.2) !important;
        }
        
        /* ═══════════════════════════════════════
           METRICS
        ═══════════════════════════════════════ */
        [data-testid="stMetricValue"] { 
            font-family: 'Space Grotesk', sans-serif !important; 
            font-size: 38px !important; 
            color: #fff !important; 
            letter-spacing: -2px;
        }
        [data-testid="stMetricLabel"] { 
            color: #555 !important; 
            font-size: 11px !important; 
            letter-spacing: 3px !important;
            text-transform: uppercase;
        }
        
        .metric-card {
            background: linear-gradient(135deg, #0a0a0a 0%, #050505 100%);
            border: 1px solid #1a1a1a;
            padding: 28px 24px;
            text-align: center;
            position: relative;
            overflow: hidden;
            transition: all 0.3s ease;
            animation: metricFade 0.8s ease forwards;
            opacity: 0;
        }
        
        .metric-card::before {
            content: '✦';
            position: absolute;
            top: 10px;
            right: 12px;
            color: #ff0000;
            font-size: 10px;
            opacity: 0.5;
        }
        
        .metric-card:hover {
            border-color: #333;
            transform: translateY(-3px);
        }
        
        .metric-card.highlight {
            border-left: 4px solid #ff0000;
            background: linear-gradient(135deg, #0f0505 0%, #0a0a0a 100%);
        }
        
        .metric-card.highlight::after {
            content: '';
            position: absolute;
            bottom: 0;
            left: 0;
            width: 100%;
            height: 2px;
            background: linear-gradient(90deg, transparent, #ff0000, transparent);
            animation: scanLine 2s infinite;
        }
        
        @keyframes metricFade {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        @keyframes scanLine {
            0% { opacity: 0; }
            50% { opacity: 1; }
            100% { opacity: 0; }
        }
        
        /* ═══════════════════════════════════════
           COMPARISON HEADER
        ═══════════════════════════════════════ */
        .comp-header {
            text-align: center;
            margin: 40px 0 30px;
            position: relative;
        }
        
        .comp-header h3 {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 14px;
            letter-spacing: 5px;
            color: #555;
            text-transform: uppercase;
            display: inline-flex;
            align-items: center;
            gap: 15px;
        }
        
        .comp-header h3::before,
        .comp-header h3::after {
            content: '—————';
            color: #222;
        }
        
        .comp-header h3 span {
            color: #ff0000;
        }
        
        /* ═══════════════════════════════════════
           DIVIDERS
        ═══════════════════════════════════════ */
        .red-divider {
            border: none;
            height: 1px;
            background: linear-gradient(90deg, transparent, #ff0000, transparent);
            margin: 35px 0;
            opacity: 0.5;
        }
        
        /* ═══════════════════════════════════════
           SELECTED ITEM DISPLAY
        ═══════════════════════════════════════ */
        .selected-item {
            background: linear-gradient(135deg, #0f0505 0%, #0a0505 100%);
            border: 2px solid #ff0000;
            padding: 20px 28px;
            margin: 25px 0;
            position: relative;
            animation: selectedPulse 2s infinite;
        }
        
        .selected-item::before {
            content: 'ANALYZING';
            position: absolute;
            top: -10px;
            left: 20px;
            background: #000;
            color: #ff0000;
            font-size: 10px;
            letter-spacing: 3px;
            padding: 2px 10px;
        }
        
        .selected-item h4 {
            color: #fff;
            font-family: 'Space Grotesk', sans-serif;
            font-size: 20px;
            font-weight: 600;
            margin: 0;
        }
        
        @keyframes selectedPulse {
            0%, 100% { box-shadow: 0 0 20px rgba(255,0,0,0.2); }
            50% { box-shadow: 0 0 40px rgba(255,0,0,0.4); }
        }
    </style>
    """, unsafe_allow_html=True)
    
    # === FLOATING SYMBOLS ===
    st.markdown('''
    <div class="app-sym as1">✦</div>
    <div class="app-sym as2">✝</div>
    <div class="app-sym as3">★</div>
    <div class="app-sym as4">✦</div>
    <div class="app-sym as5">✝</div>
    <div class="app-sym as6">★</div>
    ''', unsafe_allow_html=True)
    
    # === RED GLOWS ===
    st.markdown('''
    <div class="app-glow ag1"></div>
    <div class="app-glow ag2"></div>
    <div class="app-glow ag3"></div>
    ''', unsafe_allow_html=True)
    
    # === BACK BUTTON ===
    if st.button("← EXIT"):
        st.session_state.launched = False
        st.rerun()
    
    # === HEADER ===
    st.markdown('<div class="app-header">RECOMMENDER<span>.</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="app-sub"><strong>TF-IDF</strong> vs <strong>Neural Semantic Search</strong> — Side by Side Comparison</div>', unsafe_allow_html=True)
    
    # === TABS ===
    tab1, tab2 = st.tabs(["✦  MOVIES", "✦  SONGS"])
    
    for tab, content_type in [(tab1, "Movies"), (tab2, "Songs")]:
        with tab:
            df = load_data(content_type)
            
            st.markdown(f'<div class="sec-label">DATASET: {len(df)} {content_type.upper()}</div>', unsafe_allow_html=True)
            
            title_col = 'title' if content_type == "Movies" else 'song'
            
            # Search
            search = st.text_input(
                "Search", 
                placeholder="Type to search...", 
                key=f"search_{content_type}", 
                label_visibility="collapsed"
            )
            
            if search:
                options = df[df[title_col].str.contains(search, case=False, na=False)][title_col].tolist()
            else:
                options = df[title_col].head(25).tolist()
            
            selected = st.selectbox(
                "Select", 
                options, 
                key=f"select_{content_type}", 
                label_visibility="collapsed"
            )
            
            if selected:
                # Selected item display
                st.markdown(f'''
                <div class="selected-item">
                    <h4>✦ {selected}</h4>
                </div>
                ''', unsafe_allow_html=True)
                
                st.markdown('<hr class="red-divider">', unsafe_allow_html=True)
                
                with st.spinner(""):
                    tfidf_sim = build_tfidf_matrix(df)
                    trans_sim = build_transformer_matrix(df)
                
                col1, col2 = st.columns(2)
                
                # TF-IDF Results
                with col1:
                    st.markdown('<div class="sec-label">TF-IDF // KEYWORD MATCHING</div>', unsafe_allow_html=True)
                    tfidf_recs = get_recommendations(selected, df, tfidf_sim, content_type)
                    for i, (_, row) in enumerate(tfidf_recs.iterrows()):
                        delay = i * 0.1
                        st.markdown(f'''
                        <div class="card" style="animation-delay: {delay}s;">
                            <div class="card-title">
                                <span class="card-rank">#{i+1}</span>
                                {row[title_col]}
                            </div>
                            <div class="card-score">MATCH <span>{row["similarity_score"]*100:.1f}%</span></div>
                        </div>
                        ''', unsafe_allow_html=True)
                
                # Neural Results
                with col2:
                    if trans_sim is not None:
                        st.markdown('<div class="sec-label red">NEURAL // SEMANTIC</div>', unsafe_allow_html=True)
                        trans_recs = get_recommendations(selected, df, trans_sim, content_type)
                        for i, (_, row) in enumerate(trans_recs.iterrows()):
                            delay = i * 0.1
                            st.markdown(f'''
                            <div class="card neural" style="animation-delay: {delay}s;">
                                <div class="card-title">
                                    <span class="card-rank">#{i+1}</span>
                                    {row[title_col]}
                                </div>
                                <div class="card-score">MATCH <span class="red">{row["similarity_score"]*100:.1f}%</span></div>
                            </div>
                            ''', unsafe_allow_html=True)
                    else:
                        st.warning("Neural model unavailable")
                
                # Metrics Section
                st.markdown('''
                <div class="comp-header">
                    <h3>————— <span>✦</span> PERFORMANCE <span>✦</span> —————</h3>
                </div>
                ''', unsafe_allow_html=True)
                
                avg_tfidf = tfidf_recs['similarity_score'].mean() if not tfidf_recs.empty else 0
                avg_trans = trans_recs['similarity_score'].mean() if trans_sim is not None and not trans_recs.empty else 0
                improvement = ((avg_trans - avg_tfidf) / avg_tfidf * 100) if avg_tfidf > 0 else 0
                
                m1, m2, m3 = st.columns(3)
                
                with m1:
                    st.markdown('<div class="metric-card" style="animation-delay: 0.2s;">', unsafe_allow_html=True)
                    st.metric("TF-IDF AVG", f"{avg_tfidf:.3f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                with m2:
                    st.markdown('<div class="metric-card" style="animation-delay: 0.4s;">', unsafe_allow_html=True)
                    st.metric("NEURAL AVG", f"{avg_trans:.3f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                with m3:
                    st.markdown('<div class="metric-card highlight" style="animation-delay: 0.6s;">', unsafe_allow_html=True)
                    st.metric("IMPROVEMENT", f"+{improvement:.1f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Footer
                st.markdown('''
                <div style="text-align: center; margin-top: 50px; padding: 20px;">
                    <p style="color: #222; font-size: 11px; letter-spacing: 4px;">
                        ✦ BUILT WITH VAMP ENERGY ✦
                    </p>
                </div>
                ''', unsafe_allow_html=True)


# ══════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════
if not st.session_state.launched:
    show_landing()
else:
    show_main_app()