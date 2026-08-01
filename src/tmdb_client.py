import os
import json
import requests
import re

def _clean_title(title: str) -> str:
    """MovieLens titles look like 'Toy Story (1995)' — TMDB search
    wants just 'Toy Story', so strip the trailing year."""
    return re.sub(r'\s*\(\d{4}\)\s*$', '', title).strip()

TMDB_API_KEY = os.environ.get("TMDB_API_KEY")
TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
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


def get_poster_and_overview(title: str, year: int | None) -> dict:
    """Look up a movie on TMDB by title (+ optional year) and return
    {poster_url, overview}. Falls back to Nones if not found, no API
    key is configured, or the request fails for any reason — the app
    should keep working, just without a poster image."""
    cache = _load_cache()
    cache_key = f"{title}|{year}"
    if cache_key in cache:
        return cache[cache_key]

    result = {"poster_url": None, "overview": None}

    if not TMDB_API_KEY:
        return result

    try:
        params = {"api_key": TMDB_API_KEY, "query": _clean_title(title)}
        if year:
            params["year"] = int(year)
        response = requests.get(TMDB_SEARCH_URL, params=params, timeout=5)
        response.raise_for_status()
        results = response.json().get("results", [])
        if results:
            top = results[0]
            if top.get("poster_path"):
                result["poster_url"] = TMDB_IMAGE_BASE + top["poster_path"]
            result["overview"] = top.get("overview")
    except (requests.RequestException, ValueError):
        # network hiccup or bad response — degrade gracefully, don't crash the app
        pass

    cache[cache_key] = result
    _save_cache()
    return result