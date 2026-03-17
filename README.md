# 🎬 Content-Based Recommendation System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_LIVE_URL_HERE)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)

A dual-approach recommendation engine comparing **TF-IDF** (baseline) vs **Sentence Transformers** (upgraded) for semantic similarity search on movies and songs.

## 🎯 Features

- 🎬 **Movie Recommendations** — TMDB dataset with 5000+ movies
- 🎵 **Song Recommendations** — Spotify top hits dataset
- 📊 **TF-IDF Baseline** — Fast keyword-based matching
- 🧠 **Sentence Transformers** — Semantic embeddings with `all-MiniLM-L6-v2`
- 📈 **Side-by-side Comparison** — Compare both approaches in real-time
- ☁️ **Live Demo** — Deployed on Streamlit Cloud

## 🚀 Live Demo

**[Try it here →](YOUR_LIVE_URL_HERE)**

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| **Python 3.11** | Core language |
| **Pandas** | Data preprocessing & manipulation |
| **scikit-learn** | TF-IDF vectorization & cosine similarity |
| **sentence-transformers** | Semantic embeddings (production version) |
| **Streamlit** | Interactive web dashboard |

## 📊 How It Works

### Architecture Flow
Raw Data (CSV files)
↓
Preprocessing (clean · combine · normalize · stopwords)
↓
┌─────────────────────┬─────────────────────────┐
│ TF-IDF (Baseline) │ Sentence Transformers │
│ Word frequency │ Semantic embeddings │
└─────────────────────┴─────────────────────────┘
↓ ↓
Cosine Similarity Matrix (cached at startup)
↓
get_recommendations(title) → Top 10 results
↓
Streamlit UI (search · tabs · result cards · scores)

text


### Key Components

**1. Data Preprocessing**
- Combine title, overview/description, and genre fields
- Lowercase normalization
- Stopword removal (common words like "the", "and", "is")
- Special character cleaning

**2. TF-IDF Vectorizer (Baseline)**
- Converts text to numerical vectors based on word frequency
- 5000 max features
- Bigram support (1-2 word phrases)
- Fast but only matches exact keywords

**3. Sentence Transformers (Upgraded)**
- Pre-trained model: `all-MiniLM-L6-v2`
- Generates 384-dimensional semantic embeddings
- Understands context and meaning
- Matches "action thriller" with "intense crime film" even with zero word overlap

**4. Cosine Similarity**
- Computes similarity score for every item pair
- O(n²) matrix built once at startup and cached
- Recommendations are just fast lookups and sorting

## 📈 Performance Comparison

| Metric | TF-IDF | Sentence Transformers |
|--------|--------|----------------------|
| **Speed** | ⚡ Very Fast | 🐢 Moderate (first load) |
| **Accuracy** | Keyword-dependent | Context-aware |
| **Semantic Understanding** | ❌ No | ✅ Yes |
| **Training Required** | ❌ No | ❌ No (pre-trained) |
| **Typical Avg Score** | ~0.25 | ~0.35-0.45 |
| **Best For** | Exact matches | Conceptual similarity |

## 🚀 Installation & Setup

### Prerequisites
- Python 3.11 or higher
- pip package manager

### 1. Clone the repository
```bash
git clone https://github.com/sathya2k66ind-web/recommendation-system.git
cd recommendation-system
