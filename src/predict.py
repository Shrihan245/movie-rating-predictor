import pickle
import pandas as pd
import os

def load_model():
    model_path = 'model/movie_rating_model.pkl'
    
    # If model doesn't exist, train it automatically
    if not os.path.exists(model_path):
        print("Model not found. Training...")
        from src.train import train
        train()
    
    with open(model_path, 'rb') as f:
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