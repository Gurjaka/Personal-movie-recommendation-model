import sys
import os

# Add the current directory to Python path so we can import train module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import DataHandler, ContentModel

def simple_test():
    print("Loading data...")
    data_handler = DataHandler("data/")
    movies, _ = data_handler.load_data("movies.csv", "ratings.csv")
    movies = data_handler.preprocess_movies(movies)
    print(f"Movies loaded: {len(movies)}")
    
    # Test different inputs
    test_cases = [
        {"The Dark Knight": 5.0, "Inception": 4.5},
        {"Toy Story": 5.0, "Finding Nemo": 4.0},
        {"The Shawshank Redemption": 5.0, "Pulp Fiction": 4.5},
        {"Titanic": 4.0, "Avatar": 4.5}
    ]
    
    print("\n" + "="*60)
    print("TESTING CONTENT-BASED MODEL")
    print("="*60)
    
    content_model = ContentModel(movies)
    for i, user_ratings in enumerate(test_cases):
        print(f"\nTest {i+1}: {user_ratings}")
        try:
            recs = content_model.content_recommendations(user_ratings, top_n=5)
            print("Content recommendations:")
            for _, row in recs.iterrows():
                print(f"  - {row['title']}")
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n" + "="*60)
    print("TESTING HYBRID MODEL")
    print("="*60)
    
    try:
        # Try to import and load hybrid model
        from train import load_hybrid_model
        hybrid_model = load_hybrid_model()
        print("Hybrid model loaded successfully")
        
        for i, user_ratings in enumerate(test_cases):
            print(f"\nTest {i+1}: {user_ratings}")
            try:
                recs = hybrid_model.hybrid_recommend(user_ratings, top_n=5)
                print("Hybrid recommendations:")
                if not recs.empty:
                    for _, row in recs.iterrows():
                        print(f"  - {row['title']}")
                else:
                    print("  No recommendations returned")
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
                
    except Exception as e:
        print(f"Failed to load hybrid model: {e}")

if __name__ == "__main__":
    simple_test()
