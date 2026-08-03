import pickle
import json
import pandas as pd
from functools import lru_cache

MODEL_PATH = 'model/movie_rating_model.pkl'
FEATURE_COLS_PATH = 'model/feature_columns.json'
MOVIE_FEATURES_PATH = 'data/processed/movie_features.csv'
USER_FEATURES_PATH = 'data/processed/user_features.csv'
USER_GENRE_PREF_PATH = 'data/processed/user_genre_pref.csv'
GLOBAL_MEAN_PATH = 'data/processed/global_mean_rating.json'

_NON_GENRE_COLS = {'movie_id', 'num_ratings', 'avg_rating', 'release_year'}


@lru_cache(maxsize=1)
def load_model():
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)


@lru_cache(maxsize=1)
def load_feature_columns():
    with open(FEATURE_COLS_PATH) as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_movie_features():
    return pd.read_csv(MOVIE_FEATURES_PATH).set_index('movie_id')


@lru_cache(maxsize=1)
def load_user_features():
    return pd.read_csv(USER_FEATURES_PATH).set_index('user_id')


@lru_cache(maxsize=1)
def load_user_genre_pref():
    return pd.read_csv(USER_GENRE_PREF_PATH, index_col=0)


@lru_cache(maxsize=1)
def load_global_mean():
    with open(GLOBAL_MEAN_PATH) as f:
        return json.load(f)['global_mean_rating']


class UnknownMovieError(Exception):
    pass


class UnknownUserError(Exception):
    pass


def compute_user_genre_affinity(user_id: int, movie_id: int) -> float:
    """How much this specific user tends to like the genres this specific
    movie has — computed fresh at predict time the same way as training."""
    movie_features = load_movie_features()
    user_genre_pref = load_user_genre_pref()
    global_mean = load_global_mean()

    genre_cols = [c for c in movie_features.columns if c not in _NON_GENRE_COLS]
    movie_genres = [g for g in genre_cols if movie_features.loc[movie_id, g] == 1]

    if not movie_genres:
        return global_mean
    if user_id not in user_genre_pref.index:
        return global_mean

    available = [g for g in movie_genres if g in user_genre_pref.columns]
    if not available:
        return global_mean

    return float(user_genre_pref.loc[user_id, available].mean())


def build_feature_row(user_id: int, movie_id: int) -> pd.DataFrame:
    feature_cols = load_feature_columns()
    movie_features = load_movie_features()
    user_features = load_user_features()

    if movie_id not in movie_features.index:
        raise UnknownMovieError(f"movie_id {movie_id} not found in training data")
    if user_id not in user_features.index:
        raise UnknownUserError(f"user_id {user_id} not found in training data")

    row = {'user_id': user_id, 'movie_id': movie_id}
    row.update(movie_features.loc[movie_id].to_dict())
    row.update(user_features.loc[user_id].to_dict())
    row['user_genre_affinity'] = compute_user_genre_affinity(user_id, movie_id)

    ordered = {col: row.get(col, 0) for col in feature_cols}
    return pd.DataFrame([ordered], columns=feature_cols)


def predict_rating(user_id: int, movie_id: int) -> float:
    model = load_model()
    features = build_feature_row(user_id, movie_id)
    prediction = model.predict(features)
    return round(float(min(5.0, max(0.5, prediction[0]))), 2)


if __name__ == "__main__":
    rating = predict_rating(user_id=1, movie_id=1)
    print(f"Predicted rating: {rating}")