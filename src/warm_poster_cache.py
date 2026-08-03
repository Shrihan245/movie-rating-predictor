"""
Run this once locally (with TMDB_API_KEY set) to pre-fetch posters for the
most popular movies. Commit the resulting poster_cache.json so the grid
loads instantly on Render without hitting TMDB live on every cold start.

Usage:
    python3 -m src.warm_poster_cache
"""
from dotenv import load_dotenv
load_dotenv()

import json
import time
import pandas as pd
from src.tmdb_client import get_poster_and_overview

TOP_N = 200


def main():
    movie_features = pd.read_csv('data/processed/movie_features.csv')
    with open('data/processed/movies_lookup.json') as f:
        movies_lookup = {m['movie_id']: m for m in json.load(f)}

    top_movies = movie_features.sort_values('num_ratings', ascending=False).head(TOP_N)

    fetched, skipped = 0, 0
    for _, row in top_movies.iterrows():
        movie = movies_lookup.get(int(row['movie_id']))
        if not movie:
            continue
        result = get_poster_and_overview(movie.get('tmdb_id'))
        if result.get('poster_url'):
            fetched += 1
        else:
            skipped += 1
        time.sleep(0.05)

    print(f"Done. Posters found: {fetched}, not found: {skipped}")
    print("Cached to data/processed/poster_cache.json")


if __name__ == "__main__":
    main()