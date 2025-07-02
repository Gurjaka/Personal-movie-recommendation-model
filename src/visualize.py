import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from utils import DataHandler

# Load data
data_handler = DataHandler("data/")
movies, ratings = data_handler.load_data("movies.csv", "ratings.csv")
movies = data_handler.preprocess_movies(movies)
vis_dir = "visualizations"

# Check if visualizations directory exists, if not create one
if not os.path.exists(vis_dir):
    os.mkdir(vis_dir)

# Merge movies and ratings
df = pd.merge(movies, ratings, on='movieId')

class Visualizer:
    @staticmethod
    def plot_genre_distribution(movies: pd.DataFrame):
        """Plot distribution of movie genres"""
        # Get all unique genres
        all_genres = set()
        for genres_list in movies['genres']:
            if isinstance(genres_list, list):
                all_genres.update(genres_list)
        
        # Count genre occurrences
        genre_counts = {}
        for genre in all_genres:
            if genre != '(no genres listed)':
                genre_counts[genre] = movies[genre].sum()
        
        # Plot
        plt.figure(figsize=(12, 8))
        genres = list(genre_counts.keys())
        counts = list(genre_counts.values())
        
        plt.barh(genres, counts)
        plt.title('Movie Genre Distribution')
        plt.xlabel('Number of Movies')
        plt.tight_layout()
        plt.savefig(f'{vis_dir}/genre_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_rating_distribution(ratings: pd.DataFrame):
        """Plot distribution of ratings"""
        plt.figure(figsize=(10, 6))
        sns.histplot(ratings['rating'], bins=10, kde=True)
        plt.title('Rating Distribution')
        plt.xlabel('Rating')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(f'{vis_dir}/rating_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_movies_per_year(movies: pd.DataFrame):
        """Plot number of movies released per year"""
        # Extract year from title (assuming format like "Movie Title (Year)")
        movies['year'] = movies['title'].str.extract(r'\((\d{4})\)')
        movies['year'] = pd.to_numeric(movies['year'], errors='coerce')
        
        # Filter valid years
        movies_with_year = movies.dropna(subset=['year'])
        movies_with_year = movies_with_year[movies_with_year['year'] >= 1900]
        
        plt.figure(figsize=(12, 6))
        year_counts = movies_with_year['year'].value_counts().sort_index()
        plt.plot(year_counts.index, year_counts.values, linewidth=2)
        plt.title('Number of Movies Released Per Year')
        plt.xlabel('Year')
        plt.ylabel('Number of Movies')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{vis_dir}/movies_per_year.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_top_rated_movies(df: pd.DataFrame, min_ratings=50):
        """Plot top rated movies with minimum number of ratings"""
        # Calculate average rating and count for each movie
        movie_stats = df.groupby('title').agg({
            'rating': ['mean', 'count']
        }).reset_index()
        
        # Flatten column names
        movie_stats.columns = ['title', 'avg_rating', 'rating_count']
        
        # Filter movies with minimum ratings
        popular_movies = movie_stats[movie_stats['rating_count'] >= min_ratings]
        top_movies = popular_movies.nlargest(15, 'avg_rating')
        
        plt.figure(figsize=(10, 8))
        plt.barh(top_movies['title'], top_movies['avg_rating'])
        plt.title(f'Top 15 Highest Rated Movies (min {min_ratings} ratings)')
        plt.xlabel('Average Rating')
        plt.tight_layout()
        plt.savefig(f'{vis_dir}/top_rated_movies.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_rating_trends(df: pd.DataFrame):
        """Plot rating trends over time"""
        # Convert timestamp to datetime
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df['year'] = df['datetime'].dt.year
        
        # Calculate average rating per year
        yearly_ratings = df.groupby('year')['rating'].mean()
        
        plt.figure(figsize=(12, 6))
        plt.plot(yearly_ratings.index, yearly_ratings.values, marker='o', linewidth=2)
        plt.title('Average Rating Trends Over Time')
        plt.xlabel('Year')
        plt.ylabel('Average Rating')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{vis_dir}/rating_trends.png', dpi=300, bbox_inches='tight')
        plt.show()

def create_all_visualizations():
    """Create all visualizations"""
    print("Creating visualizations...")
    
    viz = Visualizer()
    
    print("1. Genre distribution...")
    viz.plot_genre_distribution(movies)
    
    print("2. Rating distribution...")
    viz.plot_rating_distribution(ratings)
    
    print("3. Movies per year...")
    viz.plot_movies_per_year(movies)
    
    print("4. Top rated movies...")
    viz.plot_top_rated_movies(df)
    
    print("5. Rating trends...")
    viz.plot_rating_trends(df)
    
    print("All visualizations saved!")

if __name__ == "__main__":
    create_all_visualizations()
