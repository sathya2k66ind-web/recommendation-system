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
# LANDING PAGE
# ──────────────────────────────────────────────
def show_landing():
    
    # === INJECT ALL CSS FIRST ===
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
        
        /* Wrapper */
        .land-wrap {
            min-height: 100vh;
            background: #000;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            padding: 40px 20px;
            font-family: 'Inter', sans-serif;
            position: relative;
        }
        
        /* Glows */
        .glow-orb {
            position: fixed;
            width: 500px;
            height: 500px;
            border-radius: 50%;
            pointer-events: none;
            z-index: 0;
        }
        .glow-1 {
            top: -150px;
            left: -150px;
            background: radial-gradient(circle, rgba(255,0,0,0.1) 0%, transparent 70%);
        }
        .glow-2 {
            bottom: -150px;
            right: -150px;
            background: radial-gradient(circle, rgba(255,0,0,0.07) 0%, transparent 70%);
        }
        
        /* Symbols */
        .sym {
            position: fixed;
            color: #ff0000;
            opacity: 0.1;
            font-size: 80px;
            pointer-events: none;
            z-index: 1;
            animation: float 15s infinite ease-in-out;
        }
        .s1 { top: 10%; left: 5%; }
        .s2 { top: 20%; right: 8%; font-size: 60px; animation-delay: -5s; }
        .s3 { bottom: 15%; left: 12%; font-size: 50px; animation-delay: -10s; }
        .s4 { bottom: 20%; right: 6%; font-size: 70px; animation-delay: -3s; }
        
        @keyframes float {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-15px); }
        }
        
        /* Badge */
        .badge {
            display: inline-flex;
            gap: 10px;
            border: 1px solid #333;
            padding: 10px 22px;
            color: #555;
            font-size: 11px;
            letter-spacing: 3px;
            text-transform: uppercase;
            margin-bottom: 35px;
        }
        .badge-dot { color: #ff0000; }
        
        /* Title */
        .main-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: clamp(45px, 11vw, 110px);
            font-weight: 700;
            line-height: 0.85;
            letter-spacing: -4px;
            text-transform: uppercase;
            margin-bottom: 25px;
        }
        .t-outline {
            display: block;
            -webkit-text-stroke: 1.5px #fff;
            -webkit-text-fill-color: transparent;
        }
        .t-red {
            display: block;
            color: #ff0000;
            text-shadow: 0 0 60px rgba(255,0,0,0.4);
        }
        
        /* Subtitle */
        .sub {
            color: #555;
            font-size: 14px;
            max-width: 460px;
            line-height: 1.8;
            margin: 0 auto 45px;
        }
        
        /* Divider */
        .div-line {
            width: 1px;
            height: 50px;
            background: linear-gradient(to bottom, transparent, #ff0000, transparent);
            margin: 0 auto 35px;
        }
        
        /* Stats */
        .stats-row {
            display: flex;
            gap: 60px;
            justify-content: center;
            flex-wrap: wrap;
            margin-bottom: 45px;
        }
        .stat-box { text-align: center; }
        .stat-num {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 38px;
            font-weight: 700;
            color: #fff;
            display: block;
            letter-spacing: -2px;
        }
        .stat-lbl {
            color: #444;
            font-size: 10px;
            letter-spacing: 2px;
            text-transform: uppercase;
            margin-top: 6px;
            display: block;
        }
        
        /* Features */
        .feat-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1px;
            background: #222;
            max-width: 850px;
            width: 100%;
            margin: 0 auto 50px;
        }
        @media (max-width: 768px) {
            .feat-grid { grid-template-columns: repeat(2, 1fr); }
            .stats-row { gap: 35px; }
        }
        @media (max-width: 500px) {
            .feat-grid { grid-template-columns: 1fr; }
        }
        .feat-card {
            background: #000;
            padding: 25px 20px;
            text-align: left;
            transition: 0.3s ease;
        }
        .feat-card:hover {
            background: #0a0a0a;
            border-left: 2px solid #ff0000;
        }
        .feat-ico {
            color: #ff0000;
            font-size: 18px;
            margin-bottom: 12px;
        }
        .feat-ttl {
            color: #fff;
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 600;
            font-size: 12px;
            letter-spacing: 1px;
            text-transform: uppercase;
            margin-bottom: 6px;
        }
        .feat-desc {
            color: #444;
            font-size: 11px;
            line-height: 1.6;
        }
        
        /* Button override */
        .btn-wrap .stButton > button {
            background: transparent !important;
            color: #fff !important;
            border: 1px solid #fff !important;
            border-radius: 0 !important;
            padding: 16px 50px !important;
            font-size: 12px !important;
            font-weight: 600 !important;
            font-family: 'Space Grotesk', sans-serif !important;
            letter-spacing: 4px !important;
            text-transform: uppercase !important;
        }
        .btn-wrap .stButton > button:hover {
            background: #ff0000 !important;
            border-color: #ff0000 !important;
            box-shadow: 0 0 40px rgba(255,0,0,0.5) !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # === GLOWS ===
    st.markdown('<div class="glow-orb glow-1"></div><div class="glow-orb glow-2"></div>', unsafe_allow_html=True)
    
    # === SYMBOLS ===
    st.markdown('<div class="sym s1">✦</div><div class="sym s2">✝</div><div class="sym s3">★</div><div class="sym s4">✦</div>', unsafe_allow_html=True)
    
    # === OPEN WRAPPER ===
    st.markdown('<div class="land-wrap">', unsafe_allow_html=True)
    
    # === BADGE ===
    st.markdown('<div class="badge"><span class="badge-dot">✦</span> ML · NLP · SEMANTIC SEARCH <span class="badge-dot">✦</span></div>', unsafe_allow_html=True)
    
    # === TITLE ===
    st.markdown('<h1 class="main-title"><span class="t-outline">CONTENT</span><span class="t-red">RECOMMENDER</span></h1>', unsafe_allow_html=True)
    
    # === SUBTITLE ===
    st.markdown('<p class="sub">Discover what you will love. TF-IDF meets deep semantic understanding. Watch keyword matching compete against neural embeddings.</p>', unsafe_allow_html=True)
    
    # === DIVIDER ===
    st.markdown('<div class="div-line"></div>', unsafe_allow_html=True)
    
    # === STATS ===
    st.markdown('''
    <div class="stats-row">
        <div class="stat-box"><span class="stat-num">4.8K</span><span class="stat-lbl">Movies</span></div>
        <div class="stat-box"><span class="stat-num">+66%</span><span class="stat-lbl">Semantic Uplift</span></div>
        <div class="stat-box"><span class="stat-num">384</span><span class="stat-lbl">Dimensions</span></div>
    </div>
    ''', unsafe_allow_html=True)
    
    # === FEATURES ===
    st.markdown('''
    <div class="feat-grid">
        <div class="feat-card"><div class="feat-ico">◈</div><div class="feat-ttl">TF-IDF Baseline</div><div class="feat-desc">Classic keyword frequency vectorization.</div></div>
        <div class="feat-card"><div class="feat-ico">◉</div><div class="feat-ttl">Neural Embeddings</div><div class="feat-desc">Context and meaning understanding.</div></div>
        <div class="feat-card"><div class="feat-ico">◎</div><div class="feat-ttl">Live Comparison</div><div class="feat-desc">Both approaches side by side.</div></div>
        <div class="feat-card"><div class="feat-ico">✦</div><div class="feat-ttl">5000+ Movies</div><div class="feat-desc">Full TMDB dataset included.</div></div>
        <div class="feat-card"><div class="feat-ico">✧</div><div class="feat-ttl">Spotify Tracks</div><div class="feat-desc">Music discovery enabled.</div></div>
        <div class="feat-card"><div class="feat-ico">⬡</div><div class="feat-ttl">Instant Search</div><div class="feat-desc">Pre-computed matrices.</div></div>
    </div>
    ''', unsafe_allow_html=True)
    
    # === CLOSE WRAPPER ===
    st.markdown('</div>', unsafe_allow_html=True)
    
    # === BUTTON (Streamlit native) ===
    st.markdown('<div class="btn-wrap">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1.3, 1, 1.3])
    with c2:
        if st.button("ENTER ✦", use_container_width=True):
            st.session_state.launched = True
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


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
        except Exception as e:
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
        
        .app-header { font-family: 'Space Grotesk', sans-serif; font-size: 38px; font-weight: 700; color: #fff; letter-spacing: -2px; margin-bottom: 5px; }
        .app-header span { color: #ff0000; }
        .app-sub { color: #444; font-size: 13px; letter-spacing: 1px; margin-bottom: 30px; }
        
        .sec-label { font-family: 'Space Grotesk', sans-serif; color: #555; font-size: 11px; letter-spacing: 3px; text-transform: uppercase; margin-bottom: 14px; }
        .sec-label.red { color: #ff0000; }
        
        .card { background: #0a0a0a; border: 1px solid #1a1a1a; border-left: 3px solid #333; padding: 16px 20px; margin-bottom: 8px; transition: 0.2s; }
        .card:hover { background: #111; border-left-color: #ff0000; transform: translateX(4px); }
        .card.neural { border-left-color: #440000; }
        .card.neural:hover { border-left-color: #ff0000; }
        .card-title { color: #e0e0e0; font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 14px; margin-bottom: 4px; }
        .card-score { color: #444; font-size: 11px; letter-spacing: 1px; }
        .card-score span { color: #666; font-weight: 600; }
        .card-score span.red { color: #ff0000; }
        
        .stTabs [data-baseweb="tab-list"] { background: transparent; border-bottom: 1px solid #1a1a1a; gap: 0; }
        .stTabs [data-baseweb="tab"] { background: transparent; color: #444; border: none; border-bottom: 2px solid transparent; padding: 12px 24px; font-size: 12px; letter-spacing: 2px; }
        .stTabs [data-baseweb="tab"]:hover { color: #fff; }
        .stTabs [aria-selected="true"] { color: #fff !important; border-bottom: 2px solid #ff0000 !important; }
        
        .stTextInput > div > div { background: #0a0a0a !important; border: 1px solid #222 !important; border-radius: 0 !important; }
        .stTextInput input { color: #fff !important; }
        .stSelectbox > div > div { background: #0a0a0a !important; border: 1px solid #222 !important; border-radius: 0 !important; }
        
        .stButton > button { background: transparent !important; color: #555 !important; border: 1px solid #333 !important; border-radius: 0 !important; font-size: 11px !important; letter-spacing: 2px !important; }
        .stButton > button:hover { border-color: #ff0000 !important; color: #ff0000 !important; }
        
        [data-testid="stMetricValue"] { font-family: 'Space Grotesk', sans-serif !important; font-size: 30px !important; color: #fff !important; }
        [data-testid="stMetricLabel"] { color: #444 !important; font-size: 10px !important; letter-spacing: 2px !important; }
    </style>
    """, unsafe_allow_html=True)
    
    if st.button("← BACK"):
        st.session_state.launched = False
        st.rerun()
    
    st.markdown('<div class="app-header">RECOMMENDER<span>.</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="app-sub">TF-IDF vs Neural Semantic Search</div>', unsafe_allow_html=True)
    
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
                st.markdown("<hr style='border:none;border-top:1px solid #1a1a1a;margin:25px 0;'>", unsafe_allow_html=True)
                
                with st.spinner(""):
                    tfidf_sim = build_tfidf_matrix(df)
                    trans_sim = build_transformer_matrix(df)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<div class="sec-label">TF-IDF // KEYWORD</div>', unsafe_allow_html=True)
                    tfidf_recs = get_recommendations(selected, df, tfidf_sim, content_type)
                    for _, row in tfidf_recs.iterrows():
                        st.markdown(f'<div class="card"><div class="card-title">{row[title_col]}</div><div class="card-score">MATCH <span>{row["similarity_score"]*100:.1f}%</span></div></div>', unsafe_allow_html=True)
                
                with col2:
                    if trans_sim is not None:
                        st.markdown('<div class="sec-label red">NEURAL // SEMANTIC</div>', unsafe_allow_html=True)
                        trans_recs = get_recommendations(selected, df, trans_sim, content_type)
                        for _, row in trans_recs.iterrows():
                            st.markdown(f'<div class="card neural"><div class="card-title">{row[title_col]}</div><div class="card-score">MATCH <span class="red">{row["similarity_score"]*100:.1f}%</span></div></div>', unsafe_allow_html=True)
                    else:
                        st.warning("Neural model unavailable")
                
                st.markdown("<hr style='border:none;border-top:1px solid #1a1a1a;margin:30px 0;'>", unsafe_allow_html=True)
                
                avg_tfidf = tfidf_recs['similarity_score'].mean() if not tfidf_recs.empty else 0
                avg_trans = trans_recs['similarity_score'].mean() if trans_sim is not None and not trans_recs.empty else 0
                improvement = ((avg_trans - avg_tfidf) / avg_tfidf * 100) if avg_tfidf > 0 else 0
                
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("TF-IDF AVG", f"{avg_tfidf:.3f}")
                with m2:
                    st.metric("NEURAL AVG", f"{avg_trans:.3f}")
                with m3:
                    st.metric("IMPROVEMENT", f"+{improvement:.1f}%")


# ──────────────────────────────────────────────
# ROUTER
# ──────────────────────────────────────────────
if not st.session_state.launched:
    show_landing()
else:
    show_main_app()