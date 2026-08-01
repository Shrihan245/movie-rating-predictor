import pickle
import json
import pandas as pd
from functools import lru_cache

MODEL_PATH = 'model/movie_rating_model.pkl'
FEATURE_COLS_PATH = 'model/feature_columns.json'
MOVIE_FEATURES_PATH = 'data/processed/movie_features.csv'
USER_FEATURES_PATH = 'data/processed/user_features.csv'


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


class UnknownMovieError(Exception):
    pass


class UnknownUserError(Exception):
    pass


def build_feature_row(user_id: int, movie_id: int) -> pd.DataFrame:
    """Assemble a single feature row for the model, in the exact column
    order it was trained on, using the movie/user lookup tables."""
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

    # any column the model expects but that we didn't populate (e.g. an
    # occupation dummy that isn't this user's occupation) defaults to 0
    ordered = {col: row.get(col, 0) for col in feature_cols}
    return pd.DataFrame([ordered], columns=feature_cols)


def predict_rating(user_id: int, movie_id: int) -> float:
    model = load_model()
    features = build_feature_row(user_id, movie_id)
    prediction = model.predict(features)
    # MovieLens ratings are 1-5; clip so the UI never shows an out-of-range number
    return round(float(min(5.0, max(1.0, prediction[0]))), 2)


if __name__ == "__main__":
    rating = predict_rating(user_id=1, movie_id=50)
    print(f"Predicted rating: {rating}")