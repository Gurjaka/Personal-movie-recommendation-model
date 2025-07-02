from utils import DataHandler

def debug_data():
    print("Loading and analyzing data...")
    data_handler = DataHandler("data/")
    movies, ratings = data_handler.load_data("movies.csv", "ratings.csv")
    
    print(f"Raw movies shape: {movies.shape}")
    print(f"Raw ratings shape: {ratings.shape}")
    
    print("\nFirst 5 movies:")
    print(movies.head())
    
    print("\nMovie columns:")
    print(movies.columns.tolist())
    
    print("\nSample movie titles:")
    sample_titles = [
        "The Dark Knight",
        "Dark Knight, The",
        "The Dark Knight (2008)",
        "Inception",
        "Inception (2010)",
        "Toy Story",
        "Toy Story (1995)"
    ]
    
    for title in sample_titles:
        matches = movies[movies['title'].str.contains(title.split('(')[0].strip(), case=False, na=False)]
        if not matches.empty:
            print(f"'{title}' found matches:")
            for _, row in matches.head(3).iterrows():
                print(f"  - {row['title']}")
        else:
            print(f"'{title}' - NO MATCHES")
    
    print("\nPreprocessing movies...")
    movies_processed = data_handler.preprocess_movies(movies)
    
    print(f"Processed movies shape: {movies_processed.shape}")
    print(f"Processed movies columns: {movies_processed.columns.tolist()}")
    
    print("\nGenre analysis:")
    print("Unique genres found:")
    all_genres = set()
    for genres_list in movies_processed['genres']:
        if isinstance(genres_list, list):
            all_genres.update(genres_list)
    print(sorted(all_genres))
    
    print(f"\nTotal unique genres: {len(all_genres)}")

if __name__ == "__main__":
    debug_data()
