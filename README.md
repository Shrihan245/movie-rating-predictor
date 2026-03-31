# 🎬 Movie Rating Predictor

A machine learning project that predicts movie ratings using the MovieLens 100K dataset — combining data science fundamentals with a deployable REST API.

![App Preview](movie-predictor-preview.png)

## What It Does

| Component | Description |
|-----------|-------------|
| Preprocessing | Cleans and engineers features from 100,000 real user ratings |
| Model | Random Forest Regressor trained to predict a user's movie rating |
| API | FastAPI endpoint that serves live predictions |
| Exploration | Jupyter notebook with data analysis and visualizations |

## Tech Stack
- **ML** — Python 3, scikit-learn, Pandas, NumPy
- **API** — FastAPI, Uvicorn
- **Data** — MovieLens 100K Dataset (GroupLens)
- **Dev Tools** — Git, GitHub, Jupyter, pip

## Model Performance
- **RMSE:** 1.0264 — predictions are on average ~1 star off
- **R2 Score:** 0.1659 — movie ratings are inherently subjective, making perfect prediction unrealistic

## Project Structure
```
movie-rating-predictor/
├── app.py                  # FastAPI app serving predictions
├── requirements.txt        # Dependencies
├── data/
│   ├── raw/                # MovieLens 100K dataset
│   └── processed/          # Cleaned and engineered features
├── notebooks/
│   └── exploration.ipynb   # EDA and visualizations
├── src/
│   ├── preprocess.py       # Data cleaning and feature engineering
│   ├── train.py            # Model training and evaluation
│   └── predict.py          # Load model and generate predictions
└── model/
    └── movie_rating_model.pkl  # Saved trained model (generated)
```

## Run Locally
```bash
# 1. Clone the repo
git clone https://github.com/Shrihan245/movie-rating-predictor.git
cd movie-rating-predictor

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download MovieLens 100K dataset into data/raw/
# https://grouplens.org/datasets/movielens/100k/

# 5. Preprocess and train
python3 src/preprocess.py
python3 src/train.py

# 6. Start the API
uvicorn app:app --reload
```

## API Usage

Send a POST request to `/predict`:
```json
{
  "user_id": 1,
  "movie_id": 50,
  "num_ratings": 100,
  "avg_rating": 4.2
}
```

Returns:
```json
{
  "predicted_rating": 4.63
}
```

## What I Learned
- Building an end-to-end ML pipeline from raw data to deployed API
- Feature engineering — deriving `num_ratings` and `avg_rating` per movie
- Training and evaluating a Random Forest Regressor with scikit-learn
- Serving ML models via a REST API with FastAPI
- Managing large files in git and proper `.gitignore` setup
- Evaluating model performance with RMSE and R2 score

## Roadmap
- Add more features (genre, release year, runtime)
- Try additional models (Gradient Boosting, SVR) and compare performance
- Build a simple frontend for non-technical users
- Deploy to Render or Railway (public URL)

## Author
Shrihan Bodapati — Built as a portfolio project to explore machine learning and API development.

[GitHub](https://github.com/Shrihan245) · [LinkedIn](https://www.linkedin.com/in/shrihan-bodapati)