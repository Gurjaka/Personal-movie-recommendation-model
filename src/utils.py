import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class DataHandler:
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)

    def load_data(self, movies_file: str, ratings_file: str) -> pd.DataFrame:
        movies_path = self.data_path / movies_file
        ratings_path = self.data_path / ratings_file

        movies = pd.read_csv(movies_path)
        ratings = pd.read_csv(ratings_path)

        return pd.merge(ratings, movies, on="movieId")

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        # Remove duplicates
        data = data.drop_duplicates(subset=['userId', 'movieId'])

        # Handle missing values
        data = data.dropna(subset=['rating'])

        return data

    def create_matrix(self, data: pd.DataFrame) -> pd.DataFrame:
        # Filter active users and popular movies
        user_counts = data["userId"].value_counts()
        movie_counts = data["title"].value_counts()

        active_users = user_counts[user_counts >= 50].index
        popular_movies = movie_counts[movie_counts >= 100].index

        filtered = data[data["userId"].isin(active_users) & 
                       data["title"].isin(popular_movies)]

        return filtered.pivot_table(
            index="userId", 
            columns="title", 
            values="rating"
        ).fillna(0)

class Visualizer:
    @staticmethod
    def plot_heatmap(data: pd.DataFrame):
        plt.figure(figsize=(12, 8))
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
        plt.title('Rating Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png')
        plt.close()

    @staticmethod
    def plot_distribution(data: pd.DataFrame):
        plt.figure(figsize=(10, 6))
        sns.histplot(data['rating'], bins=5, kde=True)
        plt.title('Rating Distribution')
        plt.savefig('rating_distribution.png')
        plt.close()
