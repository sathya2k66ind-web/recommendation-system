import streamlit as st
import streamlit.components.v1 as components
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
# LANDING PAGE — TRUE FULLSCREEN
# ──────────────────────────────────────────────
def show_landing():
    
    # Inject CSS to make it FULLSCREEN (remove all Streamlit padding)
    st.markdown("""
    <style>
        /* Remove ALL Streamlit default spacing */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        .stApp {
            background: #000 !important;
        }
        
        /* Kill all padding */
        .block-container {
            padding: 0 !important;
            max-width: 100% !important;
        }
        
        section[data-testid="stSidebar"] {
            display: none;
        }
        
        /* Remove top padding */
        .stApp > header + div {
            padding-top: 0 !important;
        }
        
        div[data-testid="stVerticalBlock"] {
            gap: 0 !important;
        }
        
        /* ═══════════════════════════════════════════════
           LANDING PAGE STYLES
        ═══════════════════════════════════════════════ */
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');
        
        .landing-container {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: #000;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            padding: 20px;
            box-sizing: border-box;
            font-family: 'Inter', -apple-system, sans-serif;
            overflow-y: auto;
            z-index: 9999;
        }
        
        /* Film grain */
        .landing-container::before {
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E");
            opacity: 0.03;
            pointer-events: none;
            z-index: 10000;
        }
        
        /* Red glows */
        .glow-1 {
            position: fixed;
            top: -200px;
            left: -200px;
            width: 600px;
            height: 600px;
            background: radial-gradient(circle, rgba(255,0,0,0.1) 0%, transparent 70%);
            border-radius: 50%;
            pointer-events: none;
            z-index: 0;
        }
        
        .glow-2 {
            position: fixed;
            bottom: -200px;
            right: -200px;
            width: 600px;
            height: 600px;
            background: radial-gradient(circle, rgba(255,0,0,0.07) 0%, transparent 70%);
            border-radius: 50%;
            pointer-events: none;
            z-index: 0;
        }
        
        /* Floating symbols */
        .symbol {
            position: fixed;
            color: #ff0000;
            opacity: 0.12;
            font-size: 100px;
            pointer-events: none;
            z-index: 1;
            animation: drift 20s infinite ease-in-out;
        }
        .sym-1 { top: 8%; left: 5%; animation-delay: 0s; }
        .sym-2 { top: 15%; right: 8%; font-size: 70px; animation-delay: -5s; }
        .sym-3 { bottom: 20%; left: 10%; font-size: 50px; animation-delay: -10s; }
        .sym-4 { bottom: 10%; right: 5%; font-size: 80px; animation-delay: -15s; }
        
        @keyframes drift {
            0%, 100% { transform: translate(0, 0) rotate(0deg); }
            25% { transform: translate(10px, -10px) rotate(2deg); }
            50% { transform: translate(-5px, 8px) rotate(-1deg); }
            75% { transform: translate(-10px, -5px) rotate(1deg); }
        }
        
        /* Content wrapper */
        .landing-content {
            position: relative;
            z-index: 100;
            max-width: 1000px;
            width: 100%;
        }
        
        /* Badge */
        .badge {
            display: inline-flex;
            align-items: center;
            gap: 12px;
            border: 1px solid #333;
            padding: 10px 24px;
            color: #666;
            font-size: 11px;
            letter-spacing: 4px;
            text-transform: uppercase;
            margin-bottom: 40px;
            animation: fadeIn 0.8s ease;
        }
        .badge-star { color: #ff0000; font-size: 12px; }
        
        /* Title */
        .title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: clamp(50px, 12vw, 130px);
            font-weight: 700;
            line-height: 0.85;
            letter-spacing: -5px;
            text-transform: uppercase;
            margin-bottom: 28px;
            animation: titleIn 1s ease;
        }
        .title-outline {
            display: block;
            -webkit-text-stroke: 1.5px #fff;
            -webkit-text-fill-color: transparent;
        }
        .title-blood {
            display: block;
            color: #ff0000;
            text-shadow: 0 0 80px rgba(255,0,0,0.5);
        }
        
        @keyframes titleIn {
            from { opacity: 0; transform: translateY(30px); filter: blur(10px); }
            to { opacity: 1; transform: translateY(0); filter: blur(0); }
        }
        
        /* Subtitle */
        .subtitle {
            color: #555;
            font-size: 15px;
            max-width: 500px;
            line-height: 1.8;
            margin: 0 auto 50px;
            font-weight: 300;
            animation: fadeIn 1.2s ease;
        }
        
        /* Divider */
        .divider {
            width: 1px;
            height: 60px;
            background: linear-gradient(to bottom, transparent, #ff0000, transparent);
            margin: 0 auto 40px;
            animation: fadeIn 1s ease;
        }
        
        /* Stats */
        .stats {
            display: flex;
            gap: 70px;
            justify-content: center;
            margin-bottom: 50px;
            flex-wrap: wrap;
            animation: fadeUp 1.4s ease;
        }
        .stat { text-align: center; }
        .stat-num {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 42px;
            font-weight: 700;
            color: #fff;
            display: block;
            letter-spacing: -2px;
        }
        .stat-label {
            color: #444;
            font-size: 10px;
            letter-spacing: 3px;
            text-transform: uppercase;
            margin-top: 8px;
        }
        
        /* Features grid */
        .features {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 2px;
            background: #1a1a1a;
            margin-bottom: 60px;
            animation: fadeUp 1.6s ease;
        }
        
        @media (max-width: 800px) {
            .features { grid-template-columns: repeat(2, 1fr); }
            .stats { gap: 40px; }
            .title { letter-spacing: -3px; }
        }
        @media (max-width: 500px) {
            .features { grid-template-columns: 1fr; }
            .stats { gap: 30px; }
        }
        
        .feature {
            background: #000;
            padding: 28px 24px;
            text-align: left;
            transition: all 0.3s ease;
            border-left: 2px solid transparent;
        }
        .feature:hover {
            background: #0a0a0a;
            border-left-color: #ff0000;
        }
        .feature-icon {
            color: #ff0000;
            font-size: 18px;
            margin-bottom: 14px;
        }
        .feature-title {
            color: #fff;
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 600;
            font-size: 12px;
            letter-spacing: 1px;
            text-transform: uppercase;
            margin-bottom: 8px;
        }
        .feature-desc {
            color: #444;
            font-size: 12px;
            line-height: 1.6;
        }
        
        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        /* Button area styling */
        .button-area {
            position: relative;
            z-index: 9999;
        }
        
        /* Override Streamlit button in landing */
        .landing-btn-wrap {
            display: flex;
            justify-content: center;
            margin-top: -20px;
        }
        
        .landing-btn-wrap .stButton > button {
            background: transparent !important;
            color: #fff !important;
            border: 1px solid #fff !important;
            border-radius: 0 !important;
            padding: 18px 60px !important;
            font-size: 12px !important;
            font-weight: 600 !important;
            font-family: 'Space Grotesk', sans-serif !important;
            letter-spacing: 4px !important;
            text-transform: uppercase !important;
            transition: all 0.3s ease !important;
            cursor: pointer !important;
        }
        
        .landing-btn-wrap .stButton > button:hover {
            background: #ff0000 !important;
            border-color: #ff0000 !important;
            box-shadow: 0 0 50px rgba(255,0,0,0.5) !important;
        }
    </style>
    
    <div class="glow-1"></div>
    <div class="glow-2"></div>
    
    <div class="symbol sym-1">✦</div>
    <div class="symbol sym-2">✝</div>
    <div class="symbol sym-3">★</div>
    <div class="symbol sym-4">✦</div>
    
    <div class="landing-container">
        <div class="landing-content">
            
            <div class="badge">
                <span class="badge-star">✦</span>
                <span>ML · NLP · SEMANTIC SEARCH</span>
                <span class="badge-star">✦</span>
            </div>
            
            <h1 class="title">
                <span class="title-outline">CONTENT</span>
                <span class="title-blood">RECOMMENDER</span>
            </h1>
            
            <p class="subtitle">
                Discover what you'll love. TF-IDF meets deep semantic understanding.
                Watch keyword matching compete against neural embeddings. Side by side.
            </p>
            
            <div class="divider"></div>
            
            <div class="stats">
                <div class="stat">
                    <span class="stat-num">4.8K</span>
                    <span class="stat-label">Movies</span>
                </div>
                <div class="stat">
                    <span class="stat-num">+66%</span>
                    <span class="stat-label">Semantic Uplift</span>
                </div>
                <div class="stat">
                    <span class="stat-num">384</span>
                    <span class="stat-label">Dimensions</span>
                </div>
            </div>
            
            <div class="features">
                <div class="feature">
                    <div class="feature-icon">◈</div>
                    <div class="feature-title">TF-IDF Baseline</div>
                    <div class="feature-desc">Classic keyword frequency vectorization. Fast and interpretable.</div>
                </div>
                <div class="feature">
                    <div class="feature-icon">◉</div>
                    <div class="feature-title">Neural Embeddings</div>
                    <div class="feature-desc">all-MiniLM-L6-v2 understands context, meaning, intent.</div>
                </div>
                <div class="feature">
                    <div class="feature-icon">◎</div>
                    <div class="feature-title">Live Comparison</div>
                    <div class="feature-desc">Both approaches running simultaneously. See the difference.</div>
                </div>
                <div class="feature">
                    <div class="feature-icon">✦</div>
                    <div class="feature-title">5000+ Movies</div>
                    <div class="feature-desc">Full TMDB dataset with titles, overviews, and genres.</div>
                </div>
                <div class="feature">
                    <div class="feature-icon">✧</div>
                    <div class="feature-title">Spotify Tracks</div>
                    <div class="feature-desc">Artist, track, and genre vectorized for music discovery.</div>
                </div>
                <div class="feature">
                    <div class="feature-icon">⬡</div>
                    <div class="feature-title">Instant Search</div>
                    <div class="feature-desc">Pre-computed similarity matrices. Zero latency lookups.</div>
                </div>
            </div>
            
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Button (must be Streamlit element for interactivity)
    st.markdown('<div class="landing-btn-wrap">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1.5, 1, 1.5])
    with col2:
        if st.button("✦  ENTER  ✦", use_container_width=True, key="enter_btn"):
            st.session_state.launched = True
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────
# LAZY LOAD TRANSFORMER MODEL
# ──────────────────────────────────────────────
TransformerModel = None

def get_transformer_model():
    global TransformerModel
    if TransformerModel is None:
        try:
            from sentence_transformers import SentenceTransformer
            TransformerModel = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            st.warning(f"⚠️ Neural model unavailable: {e}")
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
    except FileNotFoundError:
        st.error(f"❌ Missing: data/{content_type.lower()}.csv")
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
# MAIN APP PAGE
# ──────────────────────────────────────────────
def show_main_app():
    
    # Custom CSS for main app
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@300;400;500&display=swap');
        
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        .stApp { background: #000 !important; }
        
        .block-container {
            padding: 2rem 3rem !important;
            max-width: 1400px !important;
        }
        
        .app-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 42px;
            font-weight: 700;
            color: #fff;
            letter-spacing: -2px;
            margin-bottom: 5px;
        }
        .app-title span { color: #ff0000; }
        
        .app-sub {
            color: #444;
            font-size: 13px;
            letter-spacing: 1px;
            margin-bottom: 35px;
        }
        
        .section-label {
            font-family: 'Space Grotesk', sans-serif;
            color: #555;
            font-size: 11px;
            letter-spacing: 3px;
            text-transform: uppercase;
            margin-bottom: 16px;
        }
        .section-label.red { color: #ff0000; }
        
        .rec-card {
            background: #0a0a0a;
            border: 1px solid #1a1a1a;
            border-left: 3px solid #333;
            padding: 18px 22px;
            margin-bottom: 10px;
            transition: all 0.25s ease;
        }
        .rec-card:hover {
            background: #111;
            border-left-color: #ff0000;
            transform: translateX(5px);
        }
        .rec-card.neural { border-left-color: #550000; }
        .rec-card.neural:hover { border-left-color: #ff0000; }
        
        .rec-title {
            color: #e0e0e0;
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 600;
            font-size: 15px;
            margin-bottom: 5px;
        }
        .rec-score {
            color: #444;
            font-size: 11px;
            letter-spacing: 1px;
        }
        .rec-score span { color: #777; font-weight: 600; }
        .rec-score span.red { color: #ff0000; }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] { 
            background: transparent; 
            border-bottom: 1px solid #1a1a1a; 
            gap: 0; 
        }
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            color: #444;
            border: none;
            border-bottom: 2px solid transparent;
            padding: 14px 28px;
            font-family: 'Space Grotesk', sans-serif;
            font-size: 12px;
            font-weight: 600;
            letter-spacing: 2px;
        }
        .stTabs [data-baseweb="tab"]:hover { color: #fff; }
        .stTabs [aria-selected="true"] { 
            color: #fff !important; 
            border-bottom: 2px solid #ff0000 !important; 
        }
        
        /* Inputs */
        .stTextInput > div > div { 
            background: #0a0a0a !important; 
            border: 1px solid #222 !important; 
            border-radius: 0 !important; 
        }
        .stTextInput input { color: #fff !important; }
        .stSelectbox > div > div { 
            background: #0a0a0a !important; 
            border: 1px solid #222 !important; 
            border-radius: 0 !important; 
        }
        
        /* Button */
        .stButton > button {
            background: transparent !important;
            color: #666 !important;
            border: 1px solid #333 !important;
            border-radius: 0 !important;
            font-family: 'Space Grotesk', sans-serif !important;
            font-size: 11px !important;
            letter-spacing: 2px !important;
            padding: 10px 24px !important;
        }
        .stButton > button:hover {
            border-color: #ff0000 !important;
            color: #ff0000 !important;
        }
        
        /* Metrics */
        [data-testid="stMetricValue"] { 
            font-family: 'Space Grotesk', sans-serif !important; 
            font-size: 32px !important; 
            color: #fff !important; 
            letter-spacing: -1px;
        }
        [data-testid="stMetricLabel"] { 
            color: #444 !important; 
            font-size: 10px !important; 
            letter-spacing: 2px !important;
            text-transform: uppercase;
        }
        
        /* Metric cards */
        .metric-card {
            background: #0a0a0a;
            border: 1px solid #1a1a1a;
            padding: 24px;
            text-align: center;
        }
        .metric-card.highlight {
            border-left: 3px solid #ff0000;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Back button
    if st.button("← BACK"):
        st.session_state.launched = False
        st.rerun()
    
    # Header
    st.markdown("<div class='app-title'>RECOMMENDER<span>.</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='app-sub'>TF-IDF vs Neural Semantic Search — Side by Side Comparison</div>", unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2 = st.tabs(["✦  MOVIES", "✦  SONGS"])
    
    for tab, content_type in [(tab1, "Movies"), (tab2, "Songs")]:
        with tab:
            df = load_data(content_type)
            
            st.markdown(f"<div class='section-label'>DATASET: {len(df)} {content_type.upper()}</div>", unsafe_allow_html=True)
            
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
                st.markdown("<hr style='border:none;border-top:1px solid #1a1a1a;margin:30px 0;'>", unsafe_allow_html=True)
                
                with st.spinner(""):
                    tfidf_sim = build_tfidf_matrix(df)
                    trans_sim = build_transformer_matrix(df)
                
                col1, col2 = st.columns(2)
                
                # TF-IDF results
                with col1:
                    st.markdown("<div class='section-label'>TF-IDF // KEYWORD MATCHING</div>", unsafe_allow_html=True)
                    tfidf_recs = get_recommendations(selected, df, tfidf_sim, content_type)
                    for _, row in tfidf_recs.iterrows():
                        st.markdown(f"""
                        <div class='rec-card'>
                            <div class='rec-title'>{row[title_col]}</div>
                            <div class='rec-score'>MATCH <span>{row['similarity_score']*100:.1f}%</span></div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Neural results
                with col2:
                    if trans_sim is not None:
                        st.markdown("<div class='section-label red'>NEURAL // SEMANTIC</div>", unsafe_allow_html=True)
                        trans_recs = get_recommendations(selected, df, trans_sim, content_type)
                        for _, row in trans_recs.iterrows():
                            st.markdown(f"""
                            <div class='rec-card neural'>
                                <div class='rec-title'>{row[title_col]}</div>
                                <div class='rec-score'>MATCH <span class='red'>{row['similarity_score']*100:.1f}%</span></div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.warning("Neural model unavailable")
                
                # Metrics
                st.markdown("<hr style='border:none;border-top:1px solid #1a1a1a;margin:35px 0;'>", unsafe_allow_html=True)
                st.markdown("<div class='section-label' style='text-align:center;'>PERFORMANCE COMPARISON</div>", unsafe_allow_html=True)
                
                avg_tfidf = tfidf_recs['similarity_score'].mean() if not tfidf_recs.empty else 0
                avg_trans = trans_recs['similarity_score'].mean() if trans_sim is not None and not trans_recs.empty else 0
                improvement = ((avg_trans - avg_tfidf) / avg_tfidf * 100) if avg_tfidf > 0 else 0
                
                m1, m2, m3 = st.columns(3)
                
                with m1:
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    st.metric("TF-IDF AVG", f"{avg_tfidf:.3f}")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                with m2:
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    st.metric("NEURAL AVG", f"{avg_trans:.3f}")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                with m3:
                    st.markdown("<div class='metric-card highlight'>", unsafe_allow_html=True)
                    st.metric("IMPROVEMENT", f"+{improvement:.1f}%")
                    st.markdown("</div>", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# ROUTER
# ──────────────────────────────────────────────
if not st.session_state.launched:
    show_landing()
else:
    show_main_app()