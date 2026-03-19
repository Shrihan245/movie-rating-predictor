import pandas as pd

def load_data():
    ratings = pd.read_csv('data/raw/ml-100k/u.data',
                           sep='\t',
                           names=['user_id', 'movie_id', 'rating', 'timestamp'])
    
    movies = pd.read_csv('data/raw/ml-100k/u.item',
                          sep='|',
                          encoding='latin-1',
                          usecols=[0, 1],
                          names=['movie_id', 'title'])
    return ratings, movies

def preprocess(ratings, movies):
    df = ratings.merge(movies, on='movie_id')
    
    # engineer features
    movie_stats = df.groupby('movie_id')['rating'].agg(['count', 'mean']).reset_index()
    movie_stats.columns = ['movie_id', 'num_ratings', 'avg_rating']
    
    df = df.merge(movie_stats, on='movie_id')
    
    # drop timestamp, not useful for prediction
    df = df.drop(columns=['timestamp', 'title'])
    
    df.to_csv('data/processed/processed_ratings.csv', index=False)
    print("Saved to data/processed/processed_ratings.csv")
    return df

if __name__ == "__main__":
    ratings, movies = load_data()
    df = preprocess(ratings, movies)
    print(df.head())