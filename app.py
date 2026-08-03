from dotenv import load_dotenv
load_dotenv()

import json
import random
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.predict import predict_rating, UnknownMovieError, UnknownUserError, load_user_features
from src.tmdb_client import get_poster_and_overview

app = FastAPI()

with open("data/processed/movies_lookup.json") as f:
    MOVIES = json.load(f)  # list of {movie_id, title, year, genres}

MOVIES_BY_ID = {m["movie_id"]: m for m in MOVIES}

_movie_features = pd.read_csv("data/processed/movie_features.csv").set_index("movie_id")
_popular_movie_ids = _movie_features.sort_values("num_ratings", ascending=False).index.tolist()


class PredictRequest(BaseModel):
    user_id: int
    movie_id: int


@app.get("/api/movies/popular")
def popular_movies(limit: int = 48):
    results = []
    for movie_id in _popular_movie_ids[:limit]:
        movie = MOVIES_BY_ID.get(int(movie_id))
        if not movie:
            continue
        poster_info = get_poster_and_overview(movie.get("tmdb_id"))
        results.append({**movie, **poster_info})
    return {"results": results}


@app.get("/api/movies/search")
def search_movies(q: str, limit: int = 24):
    if not q or len(q) < 2:
        return {"results": []}
    q_lower = q.lower()
    matches = [m for m in MOVIES if q_lower in m["title"].lower()]
    results = []
    for movie in matches[:limit]:
        poster_info = get_poster_and_overview(movie.get("tmdb_id"))
        results.append({**movie, **poster_info})
    return {"results": results}


@app.get("/api/movies/{movie_id}")
def get_movie(movie_id: int):
    movie = MOVIES_BY_ID.get(movie_id)
    if not movie:
        raise HTTPException(status_code=404, detail="Movie not found")
    poster_info = get_poster_and_overview(movie.get("tmdb_id"))
    return {**movie, **poster_info}


@app.get("/api/users/random")
def random_user():
    user_features = load_user_features()
    user_id = int(random.choice(user_features.index.tolist()))
    return {"user_id": user_id}


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        rating = predict_rating(user_id=req.user_id, movie_id=req.movie_id)
    except UnknownMovieError:
        raise HTTPException(status_code=404, detail=f"movie_id {req.movie_id} not found in dataset")
    except UnknownUserError:
        raise HTTPException(status_code=404, detail=f"user_id {req.user_id} not found in dataset (try 1-943)")

    movie = MOVIES_BY_ID.get(req.movie_id, {})
    poster_info = get_poster_and_overview(movie.get("tmdb_id")) if movie else {}

    return {
        "predicted_rating": rating,
        "movie": {**movie, **poster_info},
    }


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Movie Rating Predictor</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: #14141f;
                color: #eee;
                min-height: 100vh;
            }
            header {
                position: sticky; top: 0; z-index: 20;
                background: linear-gradient(180deg, #1c1c2b 0%, #1c1c2bcc 90%, transparent);
                padding: 18px 28px 24px;
                display: flex; flex-wrap: wrap; gap: 16px; align-items: center;
                justify-content: space-between;
            }
            .title { font-size: 20px; font-weight: 700; white-space: nowrap; }
            .title span { color: #a78bfa; }
            .controls { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
            .user-box {
                display: flex; align-items: center; gap: 6px;
                background: #24243a; border-radius: 8px; padding: 6px 10px;
            }
            .user-box label { font-size: 12px; color: #999; white-space: nowrap; }
            .user-box input {
                width: 60px; background: transparent; border: none; color: #fff;
                font-size: 14px; outline: none;
            }
            .dice-btn {
                background: #2d2d45; border: none; border-radius: 6px;
                width: 32px; height: 32px; cursor: pointer; font-size: 15px;
            }
            .dice-btn:hover { background: #3a3a58; }
            .search-box {
                background: #24243a; border-radius: 8px; padding: 8px 14px;
                min-width: 220px; flex: 1; max-width: 340px;
            }
            .search-box input {
                width: 100%; background: transparent; border: none; color: #fff;
                font-size: 14px; outline: none;
            }
            .search-box input::placeholder { color: #777; }
            main { padding: 8px 28px 60px; }
            .section-label {
                font-size: 14px; color: #999; margin: 18px 0 14px; letter-spacing: 0.3px;
            }
            .grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
                gap: 18px;
            }
            .card {
                background: #1c1c2b; border-radius: 10px; overflow: hidden;
                cursor: pointer; transition: transform 0.15s;
                position: relative;
            }
            .card:hover { transform: translateY(-4px); }
            .poster-wrap {
                width: 100%; aspect-ratio: 2 / 3; background: #2a2a40;
                display: flex; align-items: center; justify-content: center;
                font-size: 34px; overflow: hidden; position: relative;
            }
            .poster-wrap img { width: 100%; height: 100%; object-fit: cover; }
            .rating-badge {
                position: absolute; top: 8px; right: 8px;
                background: rgba(20,20,31,0.9); color: #ffd166;
                font-size: 12px; font-weight: 700; padding: 4px 8px; border-radius: 6px;
                display: none;
            }
            .rating-badge.active { display: block; }
            .card-info { padding: 10px 10px 12px; }
            .card-info .title { font-size: 12.5px; font-weight: 600; line-height: 1.3;
                display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
            .card-info .meta { font-size: 11px; color: #888; margin-top: 4px; }
            .empty-state { text-align: center; color: #777; padding: 60px 20px; font-size: 14px; }

            .modal-overlay {
                display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.7);
                z-index: 100; align-items: center; justify-content: center; padding: 20px;
            }
            .modal-overlay.active { display: flex; }
            .modal {
                background: #1c1c2b; border-radius: 14px; max-width: 420px; width: 100%;
                padding: 24px; text-align: center;
            }
            .modal .poster-wrap { width: 140px; aspect-ratio: 2/3; margin: 0 auto 16px; border-radius: 8px; }
            .modal h2 { font-size: 17px; margin-bottom: 4px; }
            .modal .meta { font-size: 12.5px; color: #999; margin-bottom: 18px; }
            .modal .predicted { font-size: 42px; font-weight: 800; color: #ffd166; margin: 8px 0 4px; }
            .modal .predicted-label { font-size: 12px; color: #999; text-transform: uppercase; letter-spacing: 0.5px; }
            .modal .close-btn {
                margin-top: 18px; background: #2d2d45; border: none; color: #eee;
                padding: 10px 22px; border-radius: 8px; cursor: pointer; font-size: 13px;
            }
            .modal .close-btn:hover { background: #3a3a58; }
            .modal .loading-spinner { font-size: 13px; color: #999; padding: 20px 0; }
        </style>
    </head>
    <body>
        <header>
            <div class="title">🎬 Movie <span>Rating Predictor</span></div>
            <div class="controls">
                <div class="user-box">
                    <label for="userId">User</label>
                    <input type="number" id="userId" placeholder="1-943" min="1" max="943">
                </div>
                <button class="dice-btn" id="randomUserBtn" title="Random user">🎲</button>
                <div class="search-box">
                    <input type="text" id="movieSearch" placeholder="Search movies..." autocomplete="off">
                </div>
            </div>
        </header>

        <main>
            <div class="section-label" id="sectionLabel">Most rated</div>
            <div class="grid" id="grid"></div>
            <div class="empty-state" id="emptyState" style="display:none;">No movies found.</div>
        </main>

        <div class="modal-overlay" id="modalOverlay">
            <div class="modal">
                <div class="poster-wrap" id="modalPosterWrap">🎞️</div>
                <h2 id="modalTitle"></h2>
                <div class="meta" id="modalMeta"></div>
                <div id="modalBody">
                    <div class="loading-spinner">Predicting...</div>
                </div>
                <button class="close-btn" id="closeModalBtn">Close</button>
            </div>
        </div>

        <script>
            const grid = document.getElementById('grid');
            const emptyState = document.getElementById('emptyState');
            const sectionLabel = document.getElementById('sectionLabel');
            const userIdInput = document.getElementById('userId');
            const movieSearch = document.getElementById('movieSearch');
            const modalOverlay = document.getElementById('modalOverlay');

            function posterCell(movie) {
                if (movie.poster_url) {
                    return `<img src="${movie.poster_url}" loading="lazy">`;
                }
                return '🎞️';
            }

            function renderGrid(movies) {
                grid.innerHTML = '';
                emptyState.style.display = movies.length ? 'none' : 'block';
                movies.forEach(movie => {
                    const card = document.createElement('div');
                    card.className = 'card';
                    card.innerHTML = `
                        <div class="poster-wrap">
                            ${posterCell(movie)}
                            <div class="rating-badge"></div>
                        </div>
                        <div class="card-info">
                            <div class="title">${movie.title}</div>
                            <div class="meta">${[movie.year, (movie.genres||[]).slice(0,2).join(', ')].filter(Boolean).join(' · ')}</div>
                        </div>
                    `;
                    card.addEventListener('click', () => openModal(movie));
                    grid.appendChild(card);
                });
            }

            async function loadPopular() {
                sectionLabel.textContent = 'Most rated';
                const res = await fetch('/api/movies/popular?limit=48');
                const data = await res.json();
                renderGrid(data.results);
            }

            let debounceTimer = null;
            movieSearch.addEventListener('input', () => {
                clearTimeout(debounceTimer);
                const q = movieSearch.value.trim();
                if (!q) { loadPopular(); return; }
                debounceTimer = setTimeout(async () => {
                    sectionLabel.textContent = `Results for "${q}"`;
                    const res = await fetch(`/api/movies/search?q=${encodeURIComponent(q)}`);
                    const data = await res.json();
                    renderGrid(data.results);
                }, 300);
            });

            document.getElementById('randomUserBtn').addEventListener('click', async () => {
                const res = await fetch('/api/users/random');
                const data = await res.json();
                userIdInput.value = data.user_id;
            });

            async function openModal(movie) {
                modalOverlay.classList.add('active');
                document.getElementById('modalTitle').textContent = movie.title;
                document.getElementById('modalMeta').textContent =
                    [movie.year, (movie.genres||[]).join(', ')].filter(Boolean).join(' · ');
                const posterWrap = document.getElementById('modalPosterWrap');
                posterWrap.innerHTML = movie.poster_url ? `<img src="${movie.poster_url}">` : '🎞️';

                const modalBody = document.getElementById('modalBody');
                modalBody.innerHTML = '<div class="loading-spinner">Predicting...</div>';

                const userId = parseInt(userIdInput.value);
                if (!userId) {
                    modalBody.innerHTML = '<div class="loading-spinner">Pick a User ID up top first (or hit 🎲), then click a poster again.</div>';
                    return;
                }

                try {
                    const res = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ user_id: userId, movie_id: movie.movie_id })
                    });
                    if (!res.ok) {
                        const err = await res.json();
                        throw new Error(err.detail || 'Prediction failed');
                    }
                    const data = await res.json();
                    modalBody.innerHTML = `
                        <div class="predicted-label">Predicted Rating for User ${userId}</div>
                        <div class="predicted">${data.predicted_rating} / 5</div>
                    `;
                } catch (err) {
                    modalBody.innerHTML = `<div class="loading-spinner">Error: ${err.message}</div>`;
                }
            }

            document.getElementById('closeModalBtn').addEventListener('click', () => {
                modalOverlay.classList.remove('active');
            });
            modalOverlay.addEventListener('click', (e) => {
                if (e.target === modalOverlay) modalOverlay.classList.remove('active');
            });

            loadPopular();
        </script>
    </body>
    </html>
    """