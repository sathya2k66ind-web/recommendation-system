# 🎬 Content-Based Recommendation System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sathya-recommender.streamlit.app)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.0-orange.svg)

A dual-approach recommendation engine comparing **TF-IDF** vs **Sentence Transformers** for semantic similarity search on movies and songs.

## 🚀 Live Demo

**[Try it here → https://sathya-recommender.streamlit.app](https://sathya-recommender.streamlit.app)**

## 🎯 Features

- 🎬 Movie Recommendations (5000+ movies)
- 🎵 Song Recommendations (Spotify dataset)
- 📊 TF-IDF Baseline (keyword matching)
- 🧠 Enhanced Vectorization (semantic matching)
- 📈 Side-by-side Comparison
- ☁️ Live Deployment on Streamlit Cloud

## 🛠️ Tech Stack

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| **Python 3.11** | Core language |
| **Pandas** | Data preprocessing & manipulation |
| **scikit-learn** | TF-IDF vectorization & cosine similarity |
| **sentence-transformers** | Semantic embeddings with `all-MiniLM-L6-v2` |
| **Streamlit** | Interactive web dashboard |
| **NumPy** | Numerical computations |

## 📊 How It Works
Raw Data (CSV) → Preprocessing → Vectorization → Cosine Similarity → Top 10 Recommendations

text


1. **Preprocessing**: Combine title, overview, genres → lowercase → remove stopwords
2. **TF-IDF**: Word frequency vectorization (baseline)
**3. Sentence Transformers (Upgraded)**
- **Model**: `all-MiniLM-L6-v2` (pre-trained)
- **Embedding size**: 384 dimensions
- **Pros**: Understands context, captures semantic similarity
- **Example**: Matches "action thriller" with "intense crime film"
- **Cons**: Slower than TF-IDF, requires more memory
4. **Cosine Similarity**: Cached matrix for instant lookups
5. **Streamlit UI**: Search, compare, explore

## 🚀 Quick Start

```bash
git clone https://github.com/sathya2k66ind-web/recommendation-system.git
cd recommendation-system
pip install -r requirements.txt
streamlit run app.py