from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import predict_rating

app = FastAPI()

class MovieInput(BaseModel):
    user_id: int
    movie_id: int
    num_ratings: int
    avg_rating: float

@app.get("/")
def home():
    return {"message": "Movie Rating Predictor API"}

@app.post("/predict")
def predict(input: MovieInput):
    rating = predict_rating(
        user_id=input.user_id,
        movie_id=input.movie_id,
        num_ratings=input.num_ratings,
        avg_rating=input.avg_rating
    )
    return {"predicted_rating": rating}