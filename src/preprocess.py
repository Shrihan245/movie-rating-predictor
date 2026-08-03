import pandas as pd
import numpy as np
import re
import json

DATA_DIR = 'data/raw/ml-latest-small'


def load_data():
    movies = pd.read_csv(f'{DATA_DIR}/movies.csv')
    ratings = pd.read_csv(f'{DATA_DIR}/ratings.csv')
    links = pd.read_csv(f'{DATA_DIR}/links.csv')
    return movies, ratings, links


def extract_year(title):
    match = re.search(r'\((\d{4})\)', str(title))
    return int(match.group(1)) if match else None


def get_all_genres(movies):
    genre_set = set()
    for g in movies['genres']:
        if g == '(no genres listed)':
            continue
        genre_set.update(g.split('|'))
    return sorted(genre_set)


def build_movies_lookup(movies, links):
    """Reference table used for search / display / poster lookup, not training."""
    lookup = movies[['movieId', 'title']].copy()
    lookup = lookup.rename(columns={'movieId': 'movie_id'})
    lookup['year'] = movies['title'].apply(extract_year)
    lookup['year'] = lookup['year'].astype('Int64')
    lookup['genres'] = movies['genres'].apply(
        lambda g: [] if g == '(no genres listed)' else g.split('|')
    )
    links_renamed = links.rename(columns={'movieId': 'movie_id', 'tmdbId': 'tmdb_id'})
    lookup = lookup.merge(links_renamed[['movie_id', 'tmdb_id']], on='movie_id', how='left')
    lookup['tmdb_id'] = lookup['tmdb_id'].astype('Int64')
    return lookup


def preprocess(movies, ratings, links):
    all_genres = get_all_genres(movies)

    genre_data = {
        genre: movies['genres'].apply(lambda g: int(genre in g.split('|')) if g != '(no genres listed)' else 0)
        for genre in all_genres
    }
    movies_with_genres = pd.concat([movies[['movieId', 'title']], pd.DataFrame(genre_data)], axis=1)

    df = ratings.merge(movies_with_genres, on='movieId')
    df = df.rename(columns={'userId': 'user_id', 'movieId': 'movie_id'})

    movie_stats = df.groupby('movie_id')['rating'].agg(['count', 'mean']).reset_index()
    movie_stats.columns = ['movie_id', 'num_ratings', 'avg_rating']
    df = df.merge(movie_stats, on='movie_id')

    df['release_year'] = df['title'].apply(extract_year)
    df['release_year'] = df['release_year'].fillna(df['release_year'].median())

    # user rating-behavior features — this dataset has no age/gender/occupation,
    # so we use each user's own rating tendency instead
    user_stats = df.groupby('user_id')['rating'].agg(['count', 'mean']).reset_index()
    user_stats.columns = ['user_id', 'user_num_ratings', 'user_avg_rating']
    df = df.merge(user_stats, on='user_id')

    # user-genre affinity — how much THIS user tends to like THESE genres,
    # not just how popular the movie is overall. Build a per-user, per-genre
    # average rating table (their personal taste profile), then for every
    # rating compute the user's average affinity across the genres that
    # specific movie has.
    global_mean_rating = df['rating'].mean()

    genre_ratings = []
    for genre in all_genres:
        subset = df[df[genre] == 1][['user_id', 'rating']].copy()
        subset['genre'] = genre
        genre_ratings.append(subset)
    long_df = pd.concat(genre_ratings, ignore_index=True)

    user_genre_pref = long_df.groupby(['user_id', 'genre'])['rating'].mean().unstack('genre')
    genre_overall_avg = long_df.groupby('genre')['rating'].mean()
    user_genre_pref = user_genre_pref.reindex(columns=all_genres)
    user_genre_pref = user_genre_pref.reindex(df['user_id'].unique())
    user_genre_pref = user_genre_pref.fillna(genre_overall_avg)
    user_genre_pref = user_genre_pref.fillna(global_mean_rating)
    user_genre_pref.to_csv('data/processed/user_genre_pref.csv')

    user_pref_aligned = user_genre_pref.loc[df['user_id']].reset_index(drop=True)
    genre_flags = df[all_genres].values
    genre_counts = genre_flags.sum(axis=1)
    weighted_sum = (user_pref_aligned.values * genre_flags).sum(axis=1)
    affinity = np.divide(weighted_sum, genre_counts,
                        out=np.full_like(weighted_sum, global_mean_rating, dtype=float),
                        where=genre_counts != 0)
    df['user_genre_affinity'] = affinity

    with open('data/processed/global_mean_rating.json', 'w') as f:
        json.dump({'global_mean_rating': float(global_mean_rating)}, f)

    df = df.drop(columns=['timestamp', 'title'])

    df.to_csv('data/processed/processed_ratings.csv', index=False)
    print(f"Saved to data/processed/processed_ratings.csv  ({df.shape[0]} rows, {df.shape[1]} cols)")

    movie_feature_cols = ['movie_id', 'num_ratings', 'avg_rating', 'release_year'] + all_genres
    movie_features = df[movie_feature_cols].drop_duplicates(subset='movie_id')
    movie_features.to_csv('data/processed/movie_features.csv', index=False)

    user_feature_cols = ['user_id', 'user_num_ratings', 'user_avg_rating']
    user_features = df[user_feature_cols].drop_duplicates(subset='user_id')
    user_features.to_csv('data/processed/user_features.csv', index=False)

    print(f"Saved movie_features.csv ({len(movie_features)} movies), "
          f"user_features.csv ({len(user_features)} users)")

    return df


if __name__ == "__main__":
    movies, ratings, links = load_data()
    df = preprocess(movies, ratings, links)

    lookup = build_movies_lookup(movies, links)
    lookup.to_json('data/processed/movies_lookup.json', orient='records')
    print(f"Saved movies_lookup.json ({len(lookup)} movies)")

    print(df.head())
    print("\nFeature columns:", list(df.columns))