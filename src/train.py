import pandas as pd
import numpy as np
import joblib
import faiss
import difflib
from scipy.sparse import csr_matrix
from utils import DataHandler, ContentModel

class HybridModel:
    """
    A hybrid recommendation system combining content-based and collaborative filtering.

    This class implements a hybrid approach that leverages both content-based filtering
    (using movie genres) and collaborative filtering (using user-item interactions).
    It uses FAISS for efficient similarity search in the collaborative filtering component
    and combines recommendations from both approaches.
    """
    
    def __init__(self, movies: pd.DataFrame, ratings: pd.DataFrame):
        """
        Initialize the hybrid recommendation model.

        Sets up the hybrid model by creating sparse matrices for collaborative filtering,
        training the FAISS index, and preparing all necessary mappings.
        """
        self.movies = movies
        self.ratings = ratings
        self.sparse_matrix, self.user_mapper, self.movie_mapper = self._create_sparse_matrix()
        self.model = self._train_model()

    def _create_sparse_matrix(self):
        """
        Create a sparse user-item rating matrix with ID mappings.

        This private method processes the ratings data to create:
        1. A sparse matrix representation of user-item interactions
        2. Mappings from user/movie IDs to matrix indices
        """
        # Create mappings
        user_ids = self.ratings['userId'].unique()
        movie_ids = self.ratings['movieId'].unique()

        user_mapper = {user: idx for idx, user in enumerate(user_ids)}
        movie_mapper = {movie: idx for idx, movie in enumerate(movie_ids)}

        # Convert to sparse matrix
        rows = [user_mapper[user] for user in self.ratings['userId']]
        cols = [movie_mapper[movie] for movie in self.ratings['movieId']]

        sparse_matrix = csr_matrix(
            (self.ratings['rating'], (rows, cols)),
            shape=(len(user_mapper), len(movie_mapper))
        )

        return sparse_matrix, user_mapper, movie_mapper

    def _train_model(self):
        """
        Train the FAISS index for collaborative filtering similarity search.

        This private method prepares the collaborative filtering component by:
        1. Converting sparse matrix to dense format
        2. Normalizing vectors using L2 normalization
        3. Creating and training a FAISS index for fast similarity search
        """
        dense_matrix = self.sparse_matrix.toarray().astype('float32')
        faiss.normalize_L2(dense_matrix)
        index = faiss.IndexFlatIP(dense_matrix.shape[1])
        index.add(dense_matrix)

        return index

    def find_closest_title(self, input_title: str) -> str | None:
        """
        Find the closest matching movie title using fuzzy string matching.

        This method handles typos and variations in movie titles by finding
        the most similar title in the movies database using sequence matching.
        """
        titles = self.movies['title'].tolist()
        matches = difflib.get_close_matches(input_title, titles, n=1, cutoff=0.6)
        return matches[0] if matches else None

    def hybrid_recommend(self, user_ratings: dict, content_weight=0.4, top_n=5) -> pd.DataFrame:
        """
        Generate hybrid recommendations combining content-based and collaborative filtering.

        This method implements a hybrid recommendation approach that:
        1. Generates content-based recommendations using movie genres
        2. Creates collaborative filtering recommendations using user similarity
        3. Combines both approaches to provide diverse, high-quality recommendations
        4. Handles fuzzy matching for movie titles to improve usability
        """
        print(f"Debug: Input user_ratings: {user_ratings}")

        # Content-based recommendations
        content_model = ContentModel(self.movies)
        content_recs = content_model.content_recommendations(user_ratings, top_n*2)
        print(f"Debug: Content recommendations: {content_recs['title'].tolist()}")

        # Collaborative filtering
        user_profile = np.zeros(self.sparse_matrix.shape[1], dtype='float32')
        matched_titles = {}

        for title, rating in user_ratings.items():
            matched_title = self.find_closest_title(title)
            if matched_title:
                matched_titles[title] = matched_title
                movie_id = self.movies[self.movies['title'] == matched_title]['movieId'].values[0]
                if movie_id in self.movie_mapper:
                    movie_idx = self.movie_mapper[movie_id]
                    user_profile[movie_idx] = rating
                    print(f"Debug: {title} → {matched_title} (movie_id: {movie_id}, idx: {movie_idx})")

        collab_movies = []
        if np.count_nonzero(user_profile) == 0:
            print("⚠️ Warning: No matched titles found in user input!")
        else:
            faiss.normalize_L2(user_profile.reshape(1, -1))
            _, indices = self.model.search(user_profile.reshape(1, -1), top_n * 3)
            print(f"Debug: Collaborative indices: {indices[0]}")

            # Get movie IDs from indices
            inv_movie_mapper = {v: k for k, v in self.movie_mapper.items()}
            for idx in indices[0]:
                movie_id = inv_movie_mapper.get(idx)
                if movie_id:
                    movie_row = self.movies[self.movies['movieId'] == movie_id]
                    if not movie_row.empty:
                        title = movie_row['title'].values[0]
                        # Don't recommend movies the user already rated
                        if title not in user_ratings.keys() and title not in matched_titles.values():
                            collab_movies.append(title)

        print(f"Debug: Collaborative movies: {collab_movies}")

        # Combine results - this was the main bug!
        all_recommendations = set()

        # Add content-based recommendations
        for title in content_recs['title'].tolist():
            if title not in user_ratings.keys() and title not in matched_titles.values():
                all_recommendations.add(title)

        # Add collaborative recommendations
        for title in collab_movies:
            all_recommendations.add(title)

        # Convert to list and get top N
        final_recommendations = list(all_recommendations)[:top_n]
        print(f"Debug: Final recommendations: {final_recommendations}")

        # Return properly formatted DataFrame
        if final_recommendations:
            result_df = self.movies[self.movies['title'].isin(final_recommendations)][['title', 'genres']].head(top_n)
            return result_df
        else:
            return pd.DataFrame(columns=['title', 'genres'])

def train_model():
    """
    Train and save the hybrid recommendation model.

    This function orchestrates the complete model training process:
    1. Loads movie and rating data from CSV files
    2. Applies data preprocessing and downsampling for performance
    3. Trains the hybrid model combining content-based and collaborative filtering
    4. Saves the trained model to disk using joblib

    The function performs data downsampling to improve training speed and memory usage
    by selecting the top 20,000 most active users and top 10,000 most rated movies.
    """
    # Initialize data handler
    data_handler = DataHandler("data/")

    # Load and preprocess data
    movies, ratings = data_handler.load_data("movies.csv", "ratings.csv")

    # downsample
    top_users = ratings['userId'].value_counts().head(20000).index
    top_movies = ratings['movieId'].value_counts().head(10000).index
    ratings = ratings[ratings['userId'].isin(top_users) & ratings['movieId'].isin(top_movies)]

    # Train hybrid model
    movies = data_handler.preprocess_movies(movies)
    model = HybridModel(movies, ratings)

    # Save model with proper module reference
    import __main__
    __main__.HybridModel = HybridModel
    joblib.dump(model, "hybrid_model.joblib")
    print("Hybrid model trained and saved!")

def load_hybrid_model():
    """
    Load a pre-trained hybrid recommendation model from disk.

    This helper function loads a previously saved hybrid model using joblib,
    ensuring proper class reference resolution for successful deserialization.
    """
    import __main__
    __main__.HybridModel = HybridModel
    return joblib.load("hybrid_model.joblib")

if __name__ == "__main__":
    train_model(
