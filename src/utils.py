import pandas as pd
import numpy as np
import os
import difflib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class DataHandler:
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)

    def load_data(self, movies_file: str, ratings_file: str) -> (pd.DataFrame, pd.DataFrame):
        movies_path = self.data_path / movies_file
        ratings_path = self.data_path / ratings_file

        movies = pd.read_csv(movies_path)
        ratings = pd.read_csv(ratings_path)
        return movies, ratings

    def preprocess_movies(self, movies: pd.DataFrame) -> pd.DataFrame:
        # Split genres into list
        movies['genres'] = movies['genres'].str.split('|')

        # Create genre flags
        all_genres = set(genre for sublist in movies['genres'] for genre in sublist)
        for genre in all_genres:
            movies[genre] = movies['genres'].apply(lambda x: int(genre in x))

        return movies

class ContentModel:
    def __init__(self, movies: pd.DataFrame):
        self.movies = movies
        self.tfidf = TfidfVectorizer(stop_words='english')

        # Create genre strings for TF-IDF
        genre_strings = movies['genres'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))
        self.tfidf_matrix = self.tfidf.fit_transform(genre_strings)

        # Precompute cosine similarity matrix if not exists
        if os.path.exists("cosine_sim.npy"):
            self.cosine_sim = np.load("cosine_sim.npy")
        else:
            self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
            np.save("cosine_sim.npy", self.cosine_sim)

        # Create title to index mapping
        self.indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

    def find_closest_title(self, input_title: str) -> str | None:
        """Find the closest matching title in the dataset"""
        titles = self.movies['title'].tolist()
        matches = difflib.get_close_matches(input_title, titles, n=1, cutoff=0.6)
        return matches[0] if matches else None

    def content_recommendations(self, user_ratings: dict, top_n=10) -> pd.DataFrame:
        """Get content-based recommendations using TF-IDF similarity"""
        print(f"Debug: Input ratings: {user_ratings}")

        # Find movies that match user input
        matched_movies = {}
        for title, rating in user_ratings.items():
            closest_match = self.find_closest_title(title)
            if closest_match:
                matched_movies[closest_match] = rating
                print(f"Debug: '{title}' matched to '{closest_match}'")
            else:
                print(f"Debug: No match found for '{title}'")

        if not matched_movies:
            print("Debug: No matched movies found!")
            return pd.DataFrame(columns=['title', 'genres'])

        # Calculate weighted similarity scores
        sim_scores = np.zeros(len(self.movies))
        total_weight = 0

        for movie_title, rating in matched_movies.items():
            if movie_title in self.indices:
                movie_idx = self.indices[movie_title]
                # Get similarity scores for this movie
                movie_similarities = self.cosine_sim[movie_idx]
                # Weight by user rating
                sim_scores += movie_similarities * rating
                total_weight += rating
                print(f"Debug: Added similarities for '{movie_title}' with weight {rating}")

        # Normalize by total weight
        if total_weight > 0:
            sim_scores = sim_scores / total_weight

        # Get movie indices sorted by similarity
        sim_scores_indexed = list(enumerate(sim_scores))
        sim_scores_indexed = sorted(sim_scores_indexed, key=lambda x: x[1], reverse=True)

        # Filter out movies the user already rated and get top N
        recommendations = []
        for movie_idx, score in sim_scores_indexed:
            movie_title = self.movies.iloc[movie_idx]['title']
            if movie_title not in matched_movies and len(recommendations) < top_n:
                recommendations.append(movie_idx)
                print(f"Debug: Added recommendation: {movie_title} (score: {score:.3f})")

        if not recommendations:
            print("Debug: No recommendations generated!")
            return pd.DataFrame(columns=['title', 'genres'])

        result_df = self.movies.iloc[recommendations][['title', 'genres']]
        print(f"Debug: Returning {len(result_df)} recommendations")
        return result_df
