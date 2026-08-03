import os
import json
import requests

TMDB_API_KEY = os.environ.get("TMDB_API_KEY")
TMDB_MOVIE_URL = "https://api.themoviedb.org/3/movie/{tmdb_id}"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w342"
CACHE_PATH = "data/processed/poster_cache.json"

_cache = None


def _load_cache():
    global _cache
    if _cache is not None:
        return _cache
    try:
        with open(CACHE_PATH) as f:
            _cache = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        _cache = {}
    return _cache


def _save_cache():
    with open(CACHE_PATH, "w") as f:
        json.dump(_cache, f)


def get_poster_and_overview(tmdb_id) -> dict:
    """Look up a movie on TMDB directly by its tmdb_id (from links.csv).
    Direct ID lookup, not fuzzy title search — much more reliable."""
    result = {"poster_url": None, "overview": None}

    if tmdb_id is None or (isinstance(tmdb_id, float) and tmdb_id != tmdb_id):  # NaN check
        return result

    tmdb_id = int(tmdb_id)
    cache = _load_cache()
    cache_key = str(tmdb_id)
    if cache_key in cache:
        return cache[cache_key]

    if not TMDB_API_KEY:
        return result

    try:
        url = TMDB_MOVIE_URL.format(tmdb_id=tmdb_id)
        response = requests.get(url, params={"api_key": TMDB_API_KEY}, timeout=5)
        response.raise_for_status()
        data = response.json()
        if data.get("poster_path"):
            result["poster_url"] = TMDB_IMAGE_BASE + data["poster_path"]
        result["overview"] = data.get("overview")
    except (requests.RequestException, ValueError):
        pass

    cache[cache_key] = result
    _save_cache()
    return result