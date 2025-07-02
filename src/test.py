from utils import ContentModel, DataHandler

def test_recommendations():
    # Load data
    data_handler = DataHandler("data/")
    movies, _ = data_handler.load_data("movies.csv", "ratings.csv")
    movies = data_handler.preprocess_movies(movies)

    # Create test user preferences
    test_cases = [
        {"The Shawshank Redemption": 5, "The Godfather": 4.5},
        {"Toy Story": 5, "Finding Nemo": 4},
        {"Inception": 5, "The Matrix": 4.5}
    ]

    # Test content model
    content_model = ContentModel(movies)
    for i, preferences in enumerate(test_cases):
        print(f"\nTest Case {i+1}:")
        recs = content_model.content_recommendations(preferences)
        print(recs.head(5))

    # Test hybrid model if available
    try:
        import joblib
        hybrid_model = joblib.load("hybrid_model.joblib")
        for i, preferences in enumerate(test_cases):
            print(f"\nHybrid Test Case {i+1}:")
            recs = hybrid_model.hybrid_recommend(preferences)
            print(recs.head(5))
    except:
        print("\nHybrid model not available for testing")

if __name__ == "__main__":
    test_recommendations()
