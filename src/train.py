import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import numpy as np

def train():
    # load processed data
    df = pd.read_csv('data/processed/processed_ratings.csv')
    
    # features and target
    X = df[['user_id', 'movie_id', 'num_ratings', 'avg_rating']]
    y = df['rating']
    
    # split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    print("Training model...")
    model.fit(X_train, y_train)
    
    # evaluate
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    
    # save model
    with open('model/movie_rating_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Model saved to model/movie_rating_model.pkl")

if __name__ == "__main__":
    train()