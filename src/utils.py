import pandas as pd
import numpy as np
import os
import difflib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class DataHandler:
    """
    A class for handling movie and rating data loading and preprocessing.

    This class provides methods to load movie and rating datasets from CSV files
    and preprocess the movie data by splitting genres and creating genre flags.
    """

    def __init__(self, data_path: str):
        """
        Initialize the DataHandler with the path to data directory.
        """
        self.data_path = Path(data_path)

    def load_data(self, movies_file: str, ratings_file: str) -> (pd.DataFrame, pd.DataFrame):
        """
        Load movie and rating data from CSV files.

        This method reads the specified CSV files from the data directory
        and returns them as pandas DataFrames.
        """
        movies_path = self.data_path / movies_file
        ratings_path = self.data_path / ratings_file

        movies = pd.read_csv(movies_path)
        ratings = pd.read_csv(ratings_path)
        return movies, ratings

    def preprocess_movies(self, movies: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess movie data by splitting genres and creating genre flags.

        This method processes the 'genres' column by:
        1. Splitting genre strings on '|' delimiter into lists
        2. Creating binary flag columns for each unique genre
        3. Setting flag values to 1 if movie belongs to that genre, 0 otherwise
        """
        # Split genres into list
        movies['genres'] = movies['genres'].str.split('|')

        # Create genre flags
        all_genres = set(genre for sublist in movies['genres'] for genre in sublist)
        for genre in all_genres:
            movies[genre] = movies['genres'].apply(lambda x: int(genre in x))

        return movies

class ContentModel:
    """
    A content-based recommendation model using TF-IDF and cosine similarity.

    This class implements a content-based filtering approach for movie recommendations.
    It uses TF-IDF vectorization on movie genres and computes cosine similarity
    to find movies similar to those rated by users.
    """
    
    def __init__(self, movies: pd.DataFrame):
        """
        Initialize the ContentModel with movie data.

        This constructor sets up the TF-IDF vectorizer, computes the similarity matrix,
        and creates necessary mappings for efficient recommendation generation.
        """
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
        """
        Find the closest matching movie title in the dataset using fuzzy matching.

        This method uses difflib to find the most similar movie title to the user's
        input, helping to handle typos and variations in movie titles.
        """
        titles = self.movies['title'].tolist()
        matches = difflib.get_close_matches(input_title, titles, n=1, cutoff=0.6)
        return matches[0] if matches else None

    def content_recommendations(self, user_ratings: dict, top_n=10) -> pd.DataFrame:
        """
        Generate content-based movie recommendations using user ratings.

        This method implements a content-based filtering algorithm that:
        1. Matches user-provided movie titles to dataset titles using fuzzy matching
        2. Calculates weighted similarity scores based on user ratings
        3. Recommends movies most similar to highly-rated movies
        4. Filters out movies the user has already rated
        """
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
