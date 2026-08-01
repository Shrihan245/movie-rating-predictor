import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import json
import numpy as np


def train():
    # load processed data
    df = pd.read_csv('data/processed/processed_ratings.csv')

    # everything except the target is a feature
    feature_cols = [c for c in df.columns if c != 'rating']
    X = df[feature_cols]
    y = df['rating']

    # split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # train model
    model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    print(f"Training model on {len(feature_cols)} features...")
    model.fit(X_train, y_train)

    # evaluate
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")

    # top features, useful to see if genre/year/demographics are pulling weight
    importances = sorted(zip(feature_cols, model.feature_importances_), key=lambda x: -x[1])
    print("\nTop 10 feature importances:")
    for name, imp in importances[:10]:
        print(f"  {name:20s} {imp:.4f}")

    # save model
    with open('model/movie_rating_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    # save the exact feature column order the model expects, so predict.py
    # can always build a matching row even as features evolve
    with open('model/feature_columns.json', 'w') as f:
        json.dump(feature_cols, f)

    print("\nModel saved to model/movie_rating_model.pkl")
    print("Feature columns saved to model/feature_columns.json")


if __name__ == "__main__":
    train()