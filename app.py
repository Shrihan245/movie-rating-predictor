from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from src.predict import predict_rating

app = FastAPI()

class MovieInput(BaseModel):
    user_id: int
    movie_id: int
    num_ratings: int
    avg_rating: float

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Movie Rating Predictor</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
            }
            .container {
                background: white;
                border-radius: 12px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.2);
                max-width: 600px;
                width: 100%;
                padding: 40px;
            }
            h1 {
                color: #333;
                margin-bottom: 10px;
                font-size: 28px;
            }
            .subtitle {
                color: #666;
                margin-bottom: 30px;
                font-size: 14px;
            }
            .form-group {
                margin-bottom: 20px;
            }
            label {
                display: block;
                color: #333;
                font-weight: 500;
                margin-bottom: 8px;
                font-size: 14px;
            }
            input {
                width: 100%;
                padding: 12px;
                border: 2px solid #e0e0e0;
                border-radius: 6px;
                font-size: 14px;
                transition: border-color 0.3s;
            }
            input:focus {
                outline: none;
                border-color: #667eea;
            }
            button {
                width: 100%;
                padding: 14px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 6px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                transition: transform 0.2s, box-shadow 0.2s;
            }
            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
            }
            button:active {
                transform: translateY(0);
            }
            .results {
                margin-top: 30px;
                display: none;
            }
            .result-card {
                background: #f8f9fa;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 15px;
            }
            .result-label {
                color: #666;
                font-size: 13px;
                margin-bottom: 8px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            .result-value {
                font-size: 32px;
                font-weight: 700;
                color: #667eea;
            }
            .comparison {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 12px;
                margin-top: 20px;
            }
            .comparison-item {
                text-align: center;
                padding: 12px;
                background: white;
                border-radius: 6px;
                border: 1px solid #e0e0e0;
            }
            .comparison-label {
                color: #666;
                font-size: 12px;
                margin-bottom: 6px;
            }
            .comparison-value {
                font-size: 24px;
                font-weight: 600;
                color: #333;
            }
            .loading {
                display: none;
                text-align: center;
                color: #667eea;
                font-size: 14px;
            }
            .error {
                display: none;
                background: #fee;
                border: 1px solid #f99;
                color: #c33;
                padding: 12px;
                border-radius: 6px;
                margin-bottom: 20px;
                font-size: 13px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎬 Movie Rating Predictor</h1>
            <p class="subtitle">Predict what rating a user will give to a movie</p>
            
            <div class="error" id="error"></div>
            
            <form id="predictForm">
                <div class="form-group">
                    <label for="userId">User ID</label>
                    <input type="number" id="userId" name="user_id" placeholder="e.g., 1" required min="1">
                </div>
                
                <div class="form-group">
                    <label for="movieId">Movie ID</label>
                    <input type="number" id="movieId" name="movie_id" placeholder="e.g., 50" required min="1">
                </div>
                
                <div class="form-group">
                    <label for="numRatings">Number of Ratings (for this movie)</label>
                    <input type="number" id="numRatings" name="num_ratings" placeholder="e.g., 100" required min="1">
                </div>
                
                <div class="form-group">
                    <label for="avgRating">Average Rating (for this movie)</label>
                    <input type="number" id="avgRating" name="avg_rating" placeholder="e.g., 4.2" step="0.1" min="0.1" max="5" required>
                </div>
                
                <button type="submit">Predict Rating</button>
            </form>
            
            <div class="loading" id="loading">Predicting...</div>
            
            <div class="results" id="results">
                <div class="result-card">
                    <div class="result-label">Model Prediction (Random Forest)</div>
                    <div class="result-value" id="prediction">-</div>
                </div>
                
                <div style="font-size: 12px; color: #999; text-align: center; margin: 15px 0;">Model Performance</div>
                
                <div class="comparison">
                    <div class="comparison-item">
                        <div class="comparison-label">Your Model</div>
                        <div class="comparison-value" id="modelRmse">1.03</div>
                        <div style="font-size: 11px; color: #999;">RMSE</div>
                    </div>
                    <div class="comparison-item">
                        <div class="comparison-label">Baseline (avg: 3.5)</div>
                        <div class="comparison-value" id="baselineRmse">1.02</div>
                        <div style="font-size: 11px; color: #999;">RMSE</div>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            document.getElementById('predictForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const userId = parseInt(document.getElementById('userId').value);
                const movieId = parseInt(document.getElementById('movieId').value);
                const numRatings = parseInt(document.getElementById('numRatings').value);
                const avgRating = parseFloat(document.getElementById('avgRating').value);
                
                document.getElementById('loading').style.display = 'block';
                document.getElementById('results').style.display = 'none';
                document.getElementById('error').style.display = 'none';
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            user_id: userId,
                            movie_id: movieId,
                            num_ratings: numRatings,
                            avg_rating: avgRating
                        })
                    });
                    
                    if (!response.ok) {
                        throw new Error('Prediction failed');
                    }
                    
                    const data = await response.json();
                    document.getElementById('prediction').textContent = data.predicted_rating;
                    
                    document.getElementById('loading').style.display = 'none';
                    document.getElementById('results').style.display = 'block';
                } catch (err) {
                    document.getElementById('loading').style.display = 'none';
                    document.getElementById('error').style.display = 'block';
                    document.getElementById('error').textContent = 'Error: ' + err.message;
                }
            });
        </script>
    </body>
    </html>
    """

@app.post("/predict")
def predict(input: MovieInput):
    rating = predict_rating(
        user_id=input.user_id,
        movie_id=input.movie_id,
        num_ratings=input.num_ratings,
        avg_rating=input.avg_rating
    )
    return {"predicted_rating": rating}