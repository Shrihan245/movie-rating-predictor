import pandas as pd
import re

GENRE_COLS = [
    'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy',
    'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
    'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
]


def load_data():
    ratings = pd.read_csv('data/raw/ml-100k/u.data',
                           sep='\t',
                           names=['user_id', 'movie_id', 'rating', 'timestamp'])

    item_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url'] + GENRE_COLS
    movies = pd.read_csv('data/raw/ml-100k/u.item',
                          sep='|',
                          encoding='latin-1',
                          names=item_cols)

    users = pd.read_csv('data/raw/ml-100k/u.user',
                         sep='|',
                         names=['user_id', 'age', 'gender', 'occupation', 'zip_code'])

    return ratings, movies, users


def extract_year(title):
    # Titles look like "Toy Story (1995)"
    match = re.search(r'\((\d{4})\)', str(title))
    return int(match.group(1)) if match else None


def build_movies_lookup(movies):
    """Small reference table used later for search / display, not for training."""
    lookup = movies[['movie_id', 'title']].copy()
    lookup['year'] = movies['title'].apply(extract_year)
    genre_lists = movies[GENRE_COLS].apply(
        lambda row: [g for g, v in zip(GENRE_COLS, row) if v == 1], axis=1
    )
    lookup['genres'] = genre_lists
    return lookup


def preprocess(ratings, movies, users):
    df = ratings.merge(movies, on='movie_id').merge(users, on='user_id')

    # movie popularity / quality features (as before)
    movie_stats = df.groupby('movie_id')['rating'].agg(['count', 'mean']).reset_index()
    movie_stats.columns = ['movie_id', 'num_ratings', 'avg_rating']
    df = df.merge(movie_stats, on='movie_id')

    # release year
    df['release_year'] = df['title'].apply(extract_year)
    df['release_year'] = df['release_year'].fillna(df['release_year'].median())

    # gender as binary
    df['is_male'] = (df['gender'] == 'M').astype(int)

    # occupation as one-hot (21 categories in ml-100k, small enough to one-hot)
    occupation_dummies = pd.get_dummies(df['occupation'], prefix='occ').astype(int)
    df = pd.concat([df, occupation_dummies], axis=1)

    # drop columns not used as features
    df = df.drop(columns=[
        'timestamp', 'title', 'release_date', 'video_release_date',
        'imdb_url', 'gender', 'occupation', 'zip_code'
    ])

    df.to_csv('data/processed/processed_ratings.csv', index=False)
    print(f"Saved to data/processed/processed_ratings.csv  ({df.shape[0]} rows, {df.shape[1]} cols)")

    # one row per movie: everything predict.py needs to fill in movie-side features
    movie_feature_cols = ['movie_id', 'num_ratings', 'avg_rating', 'release_year'] + GENRE_COLS
    movie_features = df[movie_feature_cols].drop_duplicates(subset='movie_id')
    movie_features.to_csv('data/processed/movie_features.csv', index=False)

    # one row per user: everything predict.py needs to fill in user-side features
    occ_cols = [c for c in df.columns if c.startswith('occ_')]
    user_feature_cols = ['user_id', 'age', 'is_male'] + occ_cols
    user_features = df[user_feature_cols].drop_duplicates(subset='user_id')
    user_features.to_csv('data/processed/user_features.csv', index=False)

    print(f"Saved movie_features.csv ({len(movie_features)} movies), "
          f"user_features.csv ({len(user_features)} users)")

    return df


if __name__ == "__main__":
    ratings, movies, users = load_data()
    df = preprocess(ratings, movies, users)

    # also save a lightweight movie lookup table for search / display / prediction
    lookup = build_movies_lookup(movies)
    lookup.to_json('data/processed/movies_lookup.json', orient='records')
    print(f"Saved movies_lookup.json ({len(lookup)} movies)")

    print(df.head())
    print("\nFeature columns:", list(df.columns))