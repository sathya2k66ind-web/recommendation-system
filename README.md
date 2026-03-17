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

| Technology | Purpose |
|------------|---------|
| Python 3.11 | Core language |
| Pandas | Data preprocessing |
| scikit-learn | TF-IDF & cosine similarity |
| Streamlit | Interactive dashboard |

## 📊 How It Works
Raw Data (CSV) → Preprocessing → Vectorization → Cosine Similarity → Top 10 Recommendations

text


1. **Preprocessing**: Combine title, overview, genres → lowercase → remove stopwords
2. **TF-IDF**: Word frequency vectorization (baseline)
3. **Enhanced TF-IDF**: Character n-grams for better matching
4. **Cosine Similarity**: Cached matrix for instant lookups
5. **Streamlit UI**: Search, compare, explore

## 🚀 Quick Start

```bash
git clone https://github.com/sathya2k66ind-web/recommendation-system.git
cd recommendation-system
pip install -r requirements.txt
streamlit run app.py