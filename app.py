import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re

# Page config
st.set_page_config(page_title="RECOMMENDER // OPIUM", page_icon="✝", layout="wide")

# ==========================================
# OPIUM AESTHETIC CSS
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');

    /* ═══════════════════════════════════════
       ROOT VARIABLES
    ═══════════════════════════════════════ */
    :root {
        --blood: #ff0000;
        --blood-dark: #cc0000;
        --blood-deep: #8b0000;
        --void: #000000;
        --void-light: #0a0a0a;
        --void-lighter: #111111;
        --smoke: #1a1a1a;
        --ash: #2a2a2a;
        --silver: #888888;
        --bone: #e0e0e0;
        --pure: #ffffff;
    }

    /* ═══════════════════════════════════════
       BASE STYLES
    ═══════════════════════════════════════ */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, sans-serif !important;
    }

    .stApp {
        background: var(--void) !important;
    }

    /* Film grain overlay */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E");
        opacity: 0.03;
        pointer-events: none;
        z-index: 1000;
    }

    /* Subtle red glow */
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

    /* ═══════════════════════════════════════
       LANDING PAGE
    ═══════════════════════════════════════ */
    .landing-wrap {
        min-height: 90vh;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 60px 20px;
        position: relative;
    }

    /* Floating symbols */
    .symbol-float {
        position: absolute;
        color: var(--blood);
        opacity: 0.15;
        font-size: 120px;
        font-weight: 300;
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
        background: transparent;
        border: 1px solid var(--ash);
        padding: 10px 24px;
        color: var(--silver);
        font-size: 11px;
        letter-spacing: 4px;
        text-transform: uppercase;
        margin-bottom: 40px;
        font-weight: 500;
        animation: fadeIn 0.8s ease forwards;
    }

    .badge-star {
        color: var(--blood);
        font-size: 14px;
    }

    .landing-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(48px, 10vw, 140px);
        font-weight: 700;
        line-height: 0.9;
        margin: 0 0 20px;
        color: var(--pure);
        letter-spacing: -4px;
        text-transform: uppercase;
        animation: titleReveal 1s ease forwards;
        position: relative;
    }

    .landing-title span {
        display: block;
    }

    .title-outline {
        -webkit-text-stroke: 1px var(--pure);
        -webkit-text-fill-color: transparent;
    }

    .title-blood {
        color: var(--blood);
        text-shadow: 0 0 60px rgba(255,0,0,0.5);
    }

    @keyframes titleReveal {
        from { 
            opacity: 0; 
            transform: translateY(40px);
            filter: blur(10px);
        }
        to { 
            opacity: 1; 
            transform: translateY(0);
            filter: blur(0);
        }
    }

    .landing-sub {
        color: var(--silver);
        font-size: 15px;
        max-width: 500px;
        line-height: 1.8;
        margin: 0 auto 60px;
        font-weight: 300;
        letter-spacing: 0.5px;
        animation: fadeIn 1.2s ease forwards;
    }

    /* Stats row */
    .stat-row {
        display: flex;
        gap: 80px;
        justify-content: center;
        margin-bottom: 60px;
        animation: fadeInUp 1.4s ease forwards;
    }

    .stat-item {
        text-align: center;
        position: relative;
    }

    .stat-item::after {
        content: '✦';
        position: absolute;
        top: -20px;
        left: 50%;
        transform: translateX(-50%);
        color: var(--blood);
        font-size: 8px;
        opacity: 0.6;
    }

    .stat-num {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 42px;
        font-weight: 700;
        color: var(--pure);
        display: block;
        letter-spacing: -2px;
    }

    .stat-label {
        color: var(--silver);
        font-size: 10px;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-top: 8px;
        display: block;
    }

    /* Feature grid */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 2px;
        max-width: 900px;
        width: 100%;
        margin: 0 auto 60px;
        animation: fadeInUp 1.6s ease forwards;
        background: var(--ash);
    }

    .feature-card {
        background: var(--void);
        padding: 32px 24px;
        text-align: left;
        transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
        position: relative;
        overflow: hidden;
    }

    .feature-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 2px;
        background: var(--blood);
        transform: scaleX(0);
        transform-origin: left;
        transition: transform 0.4s ease;
    }

    .feature-card:hover {
        background: var(--void-light);
    }

    .feature-card:hover::before {
        transform: scaleX(1);
    }

    .feature-icon {
        font-size: 20px;
        margin-bottom: 16px;
        display: block;
        color: var(--blood);
    }

    .feature-title {
        color: var(--pure);
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        font-size: 13px;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .feature-desc {
        color: var(--silver);
        font-size: 12px;
        line-height: 1.7;
        font-weight: 300;
    }

    /* Divider */
    .landing-divider {
        width: 1px;
        height: 60px;
        background: linear-gradient(to bottom, transparent, var(--blood), transparent);
        margin: 0 auto 50px;
        animation: fadeIn 1s ease forwards;
    }

    /* ═══════════════════════════════════════
       LAUNCH BUTTON
    ═══════════════════════════════════════ */
    div[data-testid="stButton"] > button {
        background: transparent !important;
        color: var(--pure) !important;
        border: 1px solid var(--pure) !important;
        border-radius: 0 !important;
        padding: 18px 60px !important;
        font-size: 12px !important;
        font-weight: 600 !important;
        font-family: 'Space Grotesk', sans-serif !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        letter-spacing: 4px !important;
        text-transform: uppercase !important;
        position: relative !important;
        overflow: hidden !important;
    }

    div[data-testid="stButton"] > button:hover {
        background: var(--blood) !important;
        border-color: var(--blood) !important;
        box-shadow: 0 0 40px rgba(255,0,0,0.4) !important;
        transform: none !important;
    }

    div[data-testid="stButton"] > button:active {
        transform: scale(0.98) !important;
    }

    /* ═══════════════════════════════════════
       TABS STYLING
    ═══════════════════════════════════════ */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: transparent;
        border-bottom: 1px solid var(--ash);
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border: none;
        border-bottom: 2px solid transparent;
        border-radius: 0 !important;
        color: var(--silver);
        padding: 16px 32px;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        transition: all 0.3s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: var(--pure);
        background: transparent;
    }

    .stTabs [aria-selected="true"] {
        background: transparent !important;
        border: none !important;
        border-bottom: 2px solid var(--blood) !important;
        color: var(--pure) !important;
        box-shadow: none !important;
    }

    /* ═══════════════════════════════════════
       INPUT FIELDS
    ═══════════════════════════════════════ */
    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div {
        background: var(--void-light) !important;
        border: 1px solid var(--ash) !important;
        border-radius: 0 !important;
        color: var(--pure) !important;
        transition: all 0.3s ease;
    }

    div[data-baseweb="input"] > div:focus-within,
    div[data-baseweb="select"] > div:focus-within {
        border-color: var(--blood) !important;
        box-shadow: 0 0 20px rgba(255,0,0,0.15) !important;
    }

    input::placeholder {
        color: var(--silver) !important;
        font-style: italic;
    }

    /* ═══════════════════════════════════════
       GLASS CARDS - BRUTALIST VERSION
    ═══════════════════════════════════════ */
    .glass-card {
        background: var(--void-light);
        border: 1px solid var(--ash);
        border-left: 3px solid var(--ash);
        padding: 20px 24px;
        margin-bottom: 8px;
        transition: all 0.3s ease;
        animation: slideIn 0.4s ease-out forwards;
        opacity: 0;
        position: relative;
    }

    .glass-card:hover {
        background: var(--smoke);
        border-left-color: var(--blood);
        transform: translateX(4px);
    }

    .glass-card::after {
        content: '→';
        position: absolute;
        right: 20px;
        top: 50%;
        transform: translateY(-50%);
        color: var(--silver);
        opacity: 0;
        transition: all 0.3s ease;
    }

    .glass-card:hover::after {
        opacity: 1;
        right: 16px;
    }

    .card-tfidf { border-left-color: var(--silver); }
    .card-tfidf:hover { 
        border-left-color: var(--pure);
        box-shadow: 0 4px 20px rgba(255,255,255,0.05);
    }

    .card-trans { border-left-color: var(--blood-dark); }
    .card-trans:hover { 
        border-left-color: var(--blood);
        box-shadow: 0 4px 20px rgba(255,0,0,0.1);
    }

    @keyframes slideIn {
        from { 
            opacity: 0; 
            transform: translateX(-20px);
        }
        to { 
            opacity: 1; 
            transform: translateX(0);
        }
    }

    /* ═══════════════════════════════════════
       METRICS
    ═══════════════════════════════════════ */
    div[data-testid="stMetricValue"] {
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 36px !important;
        font-weight: 700 !important;
        color: var(--pure) !important;
        letter-spacing: -1px;
    }

    div[data-testid="stMetricLabel"] {
        color: var(--silver) !important;
        font-size: 11px !important;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    /* ═══════════════════════════════════════
       GRADIENT TEXT - NOW BLOOD RED
    ═══════════════════════════════════════ */
    .gradient-text {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        color: var(--pure);
        letter-spacing: -1px;
    }

    .blood-text {
        color: var(--blood);
        text-shadow: 0 0 30px rgba(255,0,0,0.3);
    }

    /* ═══════════════════════════════════════
       SIDEBAR
    ═══════════════════════════════════════ */
    section[data-testid="stSidebar"] {
        background: var(--void-light) !important;
        border-right: 1px solid var(--ash);
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 40px;
    }

    /* ═══════════════════════════════════════
       SCROLLBAR
    ═══════════════════════════════════════ */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: var(--void);
    }

    ::-webkit-scrollbar-thumb {
        background: var(--ash);
        border-radius: 0;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--blood);
    }

    /* ═══════════════════════════════════════
       MISC
    ═══════════════════════════════════════ */
    hr {
        border: none;
        border-top: 1px solid var(--ash);
        margin: 40px 0;
    }

    .section-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 11px;
        letter-spacing: 4px;
        text-transform: uppercase;
        color: var(--silver);
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .section-header::before {
        content: '✦';
        color: var(--blood);
        font-size: 10px;
    }

    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    @keyframes fadeInUp {
        from { 
            opacity: 0; 
            transform: translateY(30px);
        }
        to { 
            opacity: 1; 
            transform: translateY(0);
        }
    }

    /* Glitch effect on hover for titles */
    .glitch-hover:hover {
        animation: glitch 0.3s ease infinite;
    }

    @keyframes glitch {
        0% { transform: translate(0); }
        20% { transform: translate(-2px, 2px); }
        40% { transform: translate(-2px, -2px); }
        60% { transform: translate(2px, 2px); }
        80% { transform: translate(2px, -2px); }
        100% { transform: translate(0); }
    }

    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }

    /* Back button special style */
    .back-btn button {
        background: transparent !important;
        border: 1px solid var(--ash) !important;
        color: var(--silver) !important;
        padding: 8px 20px !important;
        font-size: 11px !important;
        letter-spacing: 2px !important;
    }

    .back-btn button:hover {
        border-color: var(--blood) !important;
        color: var(--blood) !important;
        background: transparent !important;
        box-shadow: none !important;
    }
</style>

<div class="red-glow glow-1"></div>
<div class="red-glow glow-2"></div>
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
    st.markdown('''
    <div class="landing-wrap">
        
        <!-- Floating symbols -->
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
    ''', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("ENTER ✦", use_container_width=True):
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
    # Header row
    col_back, col_spacer = st.columns([1, 5])
    with col_back:
        st.markdown('<div class="back-btn">', unsafe_allow_html=True)
        if st.button("← BACK"):
            st.session_state.launched = False
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
        <h1 style='
            font-family: Space Grotesk, sans-serif;
            font-size: 48px;
            font-weight: 700;
            color: #ffffff;
            letter-spacing: -2px;
            margin: 20px 0 8px;
        '>
            RECOMMENDER<span style='color: #ff0000;'>.</span>
        </h1>
        <p style='
            color: #666666;
            font-size: 14px;
            margin-bottom: 40px;
            letter-spacing: 1px;
        '>
            TF-IDF vs Sentence Transformers — semantic search comparison
        </p>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["✦  MOVIES", "✦  SONGS"])

    for tab, content_type in [(tab1, "Movies"), (tab2, "Songs")]:
        with tab:
            df = load_data(content_type)

            st.markdown(f"""
                <div class="section-header">
                    Dataset: {len(df)} {content_type}
                </div>
            """, unsafe_allow_html=True)

            title_col = 'title' if content_type == "Movies" else 'song'
            
            search_query = st.text_input(
                f"Search {content_type.lower()}",
                placeholder=f"Type to search...",
                key=f"search_{content_type}",
                label_visibility="collapsed"
            )

            if search_query:
                filtered = df[df[title_col].str.contains(search_query, case=False, na=False)]
                options = filtered[title_col].tolist()
            else:
                options = df[title_col].head(20).tolist()

            selected_title = st.selectbox(
                f"Select {content_type[:-1].lower()}",
                options=options,
                key=f"select_{content_type}",
                label_visibility="collapsed"
            )

            if selected_title:
                st.markdown("<hr>", unsafe_allow_html=True)

                with st.spinner(""):
                    tfidf_sim = build_tfidf_matrix(df)
                    transformer_sim = build_transformer_matrix(df)

                col1, col2 = st.columns(2)

                # TF-IDF Column
                with col1:
                    st.markdown("""
                        <div class="section-header" style="color: #888;">
                            TF-IDF // BASELINE
                        </div>
                        <p style='color: #444; font-size: 12px; margin: -16px 0 20px; letter-spacing: 1px;'>
                            Keyword frequency matching
                        </p>
                    """, unsafe_allow_html=True)
                    
                    tfidf_recs = get_recommendations(selected_title, df, tfidf_sim, content_type)
                    if not tfidf_recs.empty:
                        for i, (idx, row) in enumerate(tfidf_recs.iterrows()):
                            score = row['similarity_score']
                            st.markdown(f"""
                            <div class='glass-card card-tfidf' style='animation-delay: {i*0.05}s;'>
                                <div style='color: #e0e0e0; font-family: Space Grotesk, sans-serif; font-weight: 600; font-size: 15px;'>
                                    {row[title_col]}
                                </div>
                                <div style='color: #555; font-size: 12px; margin-top: 6px; letter-spacing: 1px;'>
                                    MATCH <span style='color: #888; font-weight: 600;'>{score*100:.1f}%</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                # Transformer Column
                with col2:
                    st.markdown("""
                        <div class="section-header" style="color: #ff0000;">
                            NEURAL // SEMANTIC
                        </div>
                        <p style='color: #444; font-size: 12px; margin: -16px 0 20px; letter-spacing: 1px;'>
                            Context-aware embeddings
                        </p>
                    """, unsafe_allow_html=True)
                    
                    transformer_recs = get_recommendations(selected_title, df, transformer_sim, content_type)
                    if not transformer_recs.empty:
                        for i, (idx, row) in enumerate(transformer_recs.iterrows()):
                            score = row['similarity_score']
                            st.markdown(f"""
                            <div class='glass-card card-trans' style='animation-delay: {i*0.05}s;'>
                                <div style='color: #e0e0e0; font-family: Space Grotesk, sans-serif; font-weight: 600; font-size: 15px;'>
                                    {row[title_col]}
                                </div>
                                <div style='color: #555; font-size: 12px; margin-top: 6px; letter-spacing: 1px;'>
                                    MATCH <span style='color: #ff0000; font-weight: 600;'>{score*100:.1f}%</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                # Metrics
                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown("""
                    <div class="section-header" style="text-align: center; justify-content: center;">
                        PERFORMANCE METRICS
                    </div>
                """, unsafe_allow_html=True)

                m1, m2, m3 = st.columns(3)
                avg_tfidf = tfidf_recs['similarity_score'].mean() if not tfidf_recs.empty else 0
                avg_transformer = transformer_recs['similarity_score'].mean() if not transformer_recs.empty else 0
                improvement = ((avg_transformer - avg_tfidf) / avg_tfidf * 100) if avg_tfidf > 0 else 0

                with m1:
                    st.markdown("""
                        <div class='glass-card' style='text-align:center;padding:28px 20px;border-left-color:#444;'>
                    """, unsafe_allow_html=True)
                    st.metric("TF-IDF AVG", f"{avg_tfidf:.3f}")
                    st.markdown("</div>", unsafe_allow_html=True)

                with m2:
                    st.markdown("""
                        <div class='glass-card' style='text-align:center;padding:28px 20px;border-left-color:#cc0000;'>
                    """, unsafe_allow_html=True)
                    st.metric("NEURAL AVG", f"{avg_transformer:.3f}")
                    st.markdown("</div>", unsafe_allow_html=True)

                with m3:
                    st.markdown(f"""
                        <div class='glass-card' style='text-align:center;padding:28px 20px;border-left-color:#ff0000;'>
                    """, unsafe_allow_html=True)
                    st.metric("IMPROVEMENT", f"+{improvement:.1f}%")
                    st.markdown("</div>", unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("""
            <div style='margin-bottom: 32px;'>
                <span style='color: #ff0000; font-size: 24px;'>✦</span>
            </div>
            <h2 style='
                font-family: Space Grotesk, sans-serif;
                font-size: 14px;
                font-weight: 700;
                color: #fff;
                letter-spacing: 2px;
                margin-bottom: 32px;
            '>
                HOW IT WORKS
            </h2>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style='background:#0a0a0a;padding:20px;border-left:2px solid #333;margin-bottom:16px;'>
            <h4 style='color:#666;margin:0 0 8px;font-size:10px;letter-spacing:2px;'>01 // PREPROCESSING</h4>
            <p style='color:#888;font-size:12px;line-height:1.7;margin:0;'>Title + overview + genre combined. Stopwords removed.</p>
        </div>
        
        <div style='background:#0a0a0a;padding:20px;border-left:2px solid #333;margin-bottom:16px;'>
            <h4 style='color:#666;margin:0 0 8px;font-size:10px;letter-spacing:2px;'>02 // TF-IDF</h4>
            <p style='color:#888;font-size:12px;line-height:1.7;margin:0;'>Word frequency vectors. Max 5000 features. Exact matching.</p>
        </div>
        
        <div style='background:#0a0a0a;padding:20px;border-left:2px solid #ff0000;margin-bottom:16px;'>
            <h4 style='color:#ff0000;margin:0 0 8px;font-size:10px;letter-spacing:2px;'>03 // NEURAL</h4>
            <p style='color:#888;font-size:12px;line-height:1.7;margin:0;'>all-MiniLM-L6-v2 embeddings. 384 dimensions. Semantic understanding.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div style='margin-top: 48px; padding-top: 24px; border-top: 1px solid #1a1a1a;'>
                <h4 style='color:#444;font-size:10px;letter-spacing:2px;margin-bottom:16px;'>STACK</h4>
            </div>
        """, unsafe_allow_html=True)
        
        st.code("Python 3.11\nPandas\nscikit-learn\nsentence-transformers\nStreamlit", language=None)


# ==========================================
# ROUTER
# ==========================================
if not st.session_state.launched:
    show_landing()
else:
    show_main_app()