import pandas as pd
import numpy as np
import joblib
import faiss
import difflib
from scipy.sparse import csr_matrix
from utils import DataHandler, ContentModel

class HybridModel:
    def __init__(self, movies: pd.DataFrame, ratings: pd.DataFrame):
        self.movies = movies
        self.ratings = ratings
        self.sparse_matrix, self.user_mapper, self.movie_mapper = self._create_sparse_matrix()
        self.model = self._train_model()

    def _create_sparse_matrix(self):
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
        dense_matrix = self.sparse_matrix.toarray().astype('float32')
        faiss.normalize_L2(dense_matrix)
        index = faiss.IndexFlatIP(dense_matrix.shape[1])
        index.add(dense_matrix)

        return index

    def find_closest_title(self, input_title: str) -> str | None:
        titles = self.movies['title'].tolist()
        matches = difflib.get_close_matches(input_title, titles, n=1, cutoff=0.6)
        return matches[0] if matches else None

    def hybrid_recommend(self, user_ratings: dict, content_weight=0.4, top_n=5) -> pd.DataFrame:
        """Hybrid content + collaborative filtering"""
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
    """Helper function to load the hybrid model with proper class reference"""
    import __main__
    __main__.HybridModel = HybridModel
    return joblib.load("hybrid_model.joblib")

if __name__ == "__main__":
    train_model()
