import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Page config
st.set_page_config(page_title="Content Recommendation System", page_icon="🎬", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #0a0e1a; color: #e2e0da; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1f2e;
        border-radius: 8px;
        color: #64748b;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #7c3aed22;
        border: 1px solid #7c3aed;
        color: #7c3aed;
    }
    div[data-testid="stMetricValue"] { font-size: 24px; color: #7c3aed; }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='color: #7c3aed;'>🎬 Content-Based Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #64748b;'>Compare TF-IDF vs Enhanced Vectorization for semantic search</p>", unsafe_allow_html=True)

# Preprocessing function
def preprocess_text(text):
    """Clean and normalize text"""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'is', 'was', 'are'}
    text = ' '.join([word for word in text.split() if word not in stopwords])
    return text

# Load and cache data
@st.cache_data
def load_data(content_type):
    """Load and preprocess datasets"""
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

# Build similarity matrices
@st.cache_resource
def build_tfidf_matrix(df):
    """Build TF-IDF cosine similarity matrix (baseline)"""
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(df['combined'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

@st.cache_resource
def build_enhanced_matrix(df):
    """Build Enhanced TF-IDF with character n-grams (upgraded)"""
    tfidf = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 3),
        analyzer='char_wb',
        min_df=2,
        sublinear_tf=True
    )
    tfidf_matrix = tfidf.fit_transform(df['combined'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

# Recommendation function
def get_recommendations(title, df, cosine_sim, content_type, top_n=10):
    """Get top N recommendations"""
    title_col = 'title' if content_type == "Movies" else 'song'
    
    if title not in df[title_col].values:
        return pd.DataFrame()
    
    idx = df[df[title_col] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    
    indices = [i[0] for i in sim_scores]
    scores = [i[1] for i in sim_scores]
    
    result = df.iloc[indices].copy()
    result['similarity_score'] = scores
    return result

# Main app
def main():
    tab1, tab2 = st.tabs(["🎬 Movies", "🎵 Songs"])
    
    for tab, content_type in [(tab1, "Movies"), (tab2, "Songs")]:
        with tab:
            df = load_data(content_type)
            
            st.markdown(f"<h3 style='color: #7c3aed;'>Dataset: {len(df)} {content_type}</h3>", unsafe_allow_html=True)
            
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
                st.markdown("---")
                
                with st.spinner("Building recommendation matrices..."):
                    tfidf_sim = build_tfidf_matrix(df)
                    enhanced_sim = build_enhanced_matrix(df)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("<h4 style='color: #475569;'>📊 TF-IDF (Baseline)</h4>", unsafe_allow_html=True)
                    st.caption("Word frequency matching • Fast • Exact keyword matches")
                    
                    tfidf_recs = get_recommendations(selected_title, df, tfidf_sim, content_type, top_n=10)
                    
                    if not tfidf_recs.empty:
                        for idx, row in tfidf_recs.iterrows():
                            score = row['similarity_score']
                            st.markdown(f"""
                            <div style='background: #0d0f14; padding: 12px; border-radius: 8px; 
                                        border-left: 3px solid #475569; margin-bottom: 8px;'>
                                <div style='color: #e2e0da; font-weight: 600;'>{row[title_col]}</div>
                                <div style='color: #64748b; font-size: 12px; margin-top: 4px;'>
                                    Similarity: <span style='color: #7c3aed;'>{score:.3f}</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("<h4 style='color: #7c3aed;'>🧠 Enhanced TF-IDF (Upgraded)</h4>", unsafe_allow_html=True)
                    st.caption("Character n-grams • Context-aware • Better matching")
                    
                    enhanced_recs = get_recommendations(selected_title, df, enhanced_sim, content_type, top_n=10)
                    
                    if not enhanced_recs.empty:
                        for idx, row in enhanced_recs.iterrows():
                            score = row['similarity_score']
                            st.markdown(f"""
                            <div style='background: #0d0f14; padding: 12px; border-radius: 8px; 
                                        border-left: 3px solid #7c3aed; margin-bottom: 8px;'>
                                <div style='color: #e2e0da; font-weight: 600;'>{row[title_col]}</div>
                                <div style='color: #64748b; font-size: 12px; margin-top: 4px;'>
                                    Similarity: <span style='color: #7c3aed;'>{score:.3f}</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                
                st.markdown("---")
                st.markdown("<h4 style='color: #be185d;'>📈 Performance Comparison</h4>", unsafe_allow_html=True)
                
                m1, m2, m3 = st.columns(3)
                
                avg_tfidf = tfidf_recs['similarity_score'].mean() if not tfidf_recs.empty else 0
                avg_enhanced = enhanced_recs['similarity_score'].mean() if not enhanced_recs.empty else 0
                
                m1.metric("Avg TF-IDF Score", f"{avg_tfidf:.3f}")
                m2.metric("Avg Enhanced Score", f"{avg_enhanced:.3f}")
                improvement = ((avg_enhanced - avg_tfidf) / avg_tfidf * 100) if avg_tfidf > 0 else 0
                m3.metric("Improvement", f"{improvement:.1f}%")

# Sidebar
with st.sidebar:
    st.markdown("<h3 style='color: #7c3aed;'>💡 How It Works</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    **1. Preprocessing**
    - Combine fields (title + overview + genre)
    - Lowercase, remove stopwords
    - Clean special characters
    
    **2. TF-IDF (Baseline)**
    - Word frequency vectorization
    - Fast, exact keyword matching
    - 5000 max features, bigrams
    
    **3. Enhanced TF-IDF (Upgraded)**
    - Character n-grams (1-3)
    - Sublinear term frequency
    - 10000 max features
    - Better semantic matching
    
    **4. Cosine Similarity**
    - Compute similarity matrix
    - Built once at startup (cached)
    - Recommendations = top 10 similar items
    """)
    
    st.markdown("---")
    st.markdown("<h4 style='color: #0f766e;'>🛠️ Tech Stack</h4>", unsafe_allow_html=True)
    st.code("""
    • Python 3.11
    • Pandas (data)
    • scikit-learn (TF-IDF)
    • Streamlit (UI)
    """)
    
    st.markdown("---")
    st.info("💡 **Note:** Once deployed to Streamlit Cloud, this will use Sentence Transformers for even better results!")

if __name__ == "__main__":
    main()