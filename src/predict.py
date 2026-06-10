import pickle
import pandas as pd
import os

def load_model():
    model_path = 'model/movie_rating_model.pkl'
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except (FileNotFoundError, Exception) as e:
        print(f"Model load failed: {e}. Using fallback predictor.")
        return None

def predict_rating(user_id, movie_id, num_ratings, avg_rating):
    model = load_model()
    
    # Fallback: if model doesn't exist, return a simple prediction
    if model is None:
        # Simple heuristic: average the avg_rating with a baseline
        return round((avg_rating + 3.5) / 2, 2)
    
    features = pd.DataFrame([[user_id, movie_id, num_ratings, avg_rating]],
                             columns=['user_id', 'movie_id', 'num_ratings', 'avg_rating'])
    prediction = model.predict(features)
    return round(prediction[0], 2)

if __name__ == "__main__":
    rating = predict_rating(
        user_id=1,
        movie_id=50,
        num_ratings=100,
        avg_rating=4.2
    )
    print(f"Predicted rating: {rating}")