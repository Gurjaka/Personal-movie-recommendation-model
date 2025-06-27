import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

class Utils:
    def __init__(self, data_path:str) -> None:
       """
       Initialize the Utils class with a given data directory path.
       All CSV files will be read from this path.
       """
       self.data_path = Path(data_path)

    def read_merge_csv(self, movies_file:str, ratings_file:str) -> pd.DataFrame:
        """
        Reads movie and ratings CSVs, merges them on 'movieId',
        and returns a combined DataFrame.

        Parameters:
            movies_file: Filename of movies CSV.
            ratings_file: Filename of ratings CSV.

        Returns:
            Merged DataFrame containing movie and rating data.
        """
        movies_path = self.data_path / movies_file
        assert movies_path.exists(), f"File not found: {movies_path}"
        
        ratings_path = self.data_path / ratings_file 
        assert ratings_path.exists(), f"File not found: {ratings_path}"

        movies = pd.read_csv(movies_path)
        ratings = pd.read_csv(ratings_path)
        
        merged_data = pd.merge(ratings, movies, on="movieId")

        return merged_data

    def correlation_heatmap(self, data: pd.DataFrame) -> None:
        """
        Plots a correlation heatmap of all numerical features in the dataset.

        Parameters:
            data: The merged movie-rating DataFrame.
        """
        num_cols = data.select_dtypes(include=['float64', 'int64']).columns
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data[num_cols])
        scaled_df = pd.DataFrame(scaled_data, columns=num_cols)

        plt.figure(figsize=(10, 6))
        sns.heatmap(scaled_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Heatmap (Scaled Numerical Data)')
        plt.show()

    def user_movie_matrix(self, data: pd.DataFrame, min_ratings_per_user=50, min_ratings_per_movie=100) -> pd.DataFrame:
        """
        Creates a user-item matrix where each row is a user, each column is a movie,
        and each cell is the user's rating for that movie (0 if not rated).

        Parameters:
            data: The merged DataFrame of ratings and movie titles.

        Returns:
            A pivoted DataFrame (user-item matrix).
        """
        # filter users who rated at least `min_ratings_per_user` movies
        user_counts = data["userId"].value_counts()
        active_user_ids = user_counts[user_counts >= min_ratings_per_user].index
        data = data[data["userId"].isin(active_user_ids)]

        # filter movies that have received at least `min_ratings_per_movie` ratings
        movie_counts = data["title"].value_counts()
        popular_movie_titles = movie_counts[movie_counts >= min_ratings_per_movie].index
        data = data[data["title"].isin(popular_movie_titles)]

        pivot = data.pivot_table(index="userId", columns="title", values="rating")
        return pivot.fillna(0)

    def recommend_movies(self, user_id: int, user_item_matrix: pd.DataFrame, top_n=5) -> pd.Series:
        """
        Recommends top N movies to a user based on ratings from similar users
        using user-based collaborative filtering and cosine similarity.

        Parameters:
            user_id: The ID of the user to recommend for.
            user_item_matrix: The user-item matrix.
            top_n: Number of recommendations to return.

        Returns:
            A Series of top recommended movies with predicted scores.
        """
        # Compute similarity between target user and all users
        similarity = cosine_similarity([user_item_matrix.loc[user_id]], user_item_matrix)[0]

        # Get top-N most similar users (excluding the user itself at index 0)
        similar_users = user_item_matrix.iloc[similarity.argsort()[::-1][1:top_n+1]]

        # Average the ratings from these similar users
        mean_ratings = similar_users.mean(axis=0)

        # Get movies the target user hasn't rated yet
        user_ratings = user_item_matrix.loc[user_id]
        unseen_movies = user_ratings[user_ratings == 0].index

        # Recommend top-N highest-rated unseen movies
        recommendations = mean_ratings[unseen_movies].sort_values(ascending=False).head(top_n)

        return recommendations
