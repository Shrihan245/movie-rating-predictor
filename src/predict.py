import pickle
import numpy as np
import pandas as pd

def load_model():
    with open('model/movie_rating_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

def predict_rating(user_id, movie_id, num_ratings, avg_rating):
    model = load_model()
    features = pd.DataFrame([[user_id, movie_id, num_ratings, avg_rating]],
                             columns=['user_id', 'movie_id', 'num_ratings', 'avg_rating'])
    prediction = model.predict(features)
    return round(prediction[0], 2)

if __name__ == "__main__":
    # test prediction
    rating = predict_rating(
        user_id=1,
        movie_id=50,
        num_ratings=100,
        avg_rating=4.2
    )
    print(f"Predicted rating: {rating}")