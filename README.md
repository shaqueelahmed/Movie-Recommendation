# 🎬 Full-Stack Movie Recommendation Engine

A modern, full-stack movie recommendation system that provides personalized film suggestions using Machine Learning (TF-IDF vectorization) and metadata analysis. The project features a responsive Streamlit frontend communicating with a FastAPI backend to deliver real-time search, dynamic routing, and intelligent recommendations.

## 🚀 Overview
This application goes beyond basic static datasets by integrating with live APIs and deploying a dual-recommendation strategy. When a user searches for or selects a movie, the engine calculates the mathematical similarity between titles using two distinct methods:
1. **Content-Based Filtering (TF-IDF):** Analyzes plot overviews, keywords, and tags to find narratively similar films.
2. **Metadata Filtering (Genre):** Retrieves and matches films within the same primary categories to ensure a diverse set of "More Like This" recommendations.

## 🛠️ Tech Stack
* **Frontend:** Streamlit (Custom UI styling, dynamic state-based routing)
* **Backend Framework:** FastAPI / Uvicorn (REST API)
* **Machine Learning & Data Processing:** Scikit-Learn (TF-IDF Vectorizer), Pandas, NumPy, SciPy
* **External APIs:** The Movie Database (TMDB) API for dynamic poster fetching and metadata.
* **Deployment Integration:** Configured to run locally or connect to a hosted backend (e.g., Render).

## ⚙️ Key Features
* **Smart Autocomplete Search:** Fast, character-based search that queries the backend and returns instant suggestions and poster grids.
* **Dynamic Single-Page Routing:** Uses Streamlit's `st.session_state` and query parameters to seamlessly navigate between the "Home Feed" (Trending, Popular, Top Rated) and detailed "Movie View" without page reloads.
* **Dual-Algorithm Recommendations:** Surfaces both narratively similar films (via Machine Learning) and genre-aligned films to give the user better choices.
* **Robust API Handling:** Includes built-in error handling, timeouts, and fallback logic if the primary recommendation engine is unavailable.
