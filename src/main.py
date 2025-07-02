import gradio as gr
import pandas as pd
from utils import DataHandler, ContentModel

# Initialize components
data_handler = DataHandler("data/")
movies, _ = data_handler.load_data("movies.csv", "ratings.csv")
movies = data_handler.preprocess_movies(movies)

try:
    # Import the model class and loading function
    from train import load_hybrid_model
    hybrid_model = load_hybrid_model()
    model_loaded = True
    print("✅ Hybrid model loaded successfully")
except Exception as e:
    print(f"❌ Failed to load hybrid model: {e}")
    model_loaded = False

def format_recommendations_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "No recommendations found."
    
    lines = ["| Title | Genres |", "|-------|--------|"]
    for _, row in df.iterrows():
        title = row['title']
        genres = ', '.join(row['genres']) if isinstance(row['genres'], list) else row['genres']
        lines.append(f"| {title} | {genres} |")
    return '\n'.join(lines)

def format_recommendations(df: pd.DataFrame) -> str:
    if df.empty:
        return "No recommendations found."
    
    lines = []
    for _, row in df.iterrows():
        title = row['title']
        genres = ', '.join(row['genres']) if isinstance(row['genres'], list) else row['genres']
        lines.append(f"🎬 {title}\n   📂 Genres: {genres}")
    return '\n\n'.join(lines)

def recommend_movies(user_input: str):
    """Parse user input and recommend movies"""
    print(f"\n🔍 DEBUG: User input: '{user_input}'")
    
    # Parse input: "Movie Title, Movie Title, Movie Title"
    user_ratings = {}
    movie_titles = [title.strip() for title in user_input.split(',') if title.strip()]
    
    if not movie_titles:
        return "Please enter at least one movie title"
    
    print(f"🔍 DEBUG: Parsed movie titles: {movie_titles}")
    
    # Assign default rating of 4.0 to all movies (assuming user likes them)
    for title in movie_titles:
        user_ratings[title] = 4.0
    
    print(f"🔍 DEBUG: User ratings dict: {user_ratings}")
    print(f"🔍 DEBUG: Model loaded: {model_loaded}")

    # Use hybrid model if available, else content-based
    if model_loaded:
        print("🔍 DEBUG: Using hybrid model")
        recommendations = hybrid_model.hybrid_recommend(user_ratings)
        print(f"🔍 DEBUG: Hybrid recommendations shape: {recommendations.shape}")
        print(f"🔍 DEBUG: Hybrid recommendations:\n{recommendations}")
    else:
        print("🔍 DEBUG: Using content-based model")
        content_model = ContentModel(movies)
        recommendations = content_model.content_recommendations(user_ratings)
        print(f"🔍 DEBUG: Content recommendations shape: {recommendations.shape}")
        print(f"🔍 DEBUG: Content recommendations:\n{recommendations}")

    # Format output
    if recommendations.empty:
        return "No recommendations found. Please check if movie titles are correct."
    
    result = format_recommendations(recommendations)
    print(f"🔍 DEBUG: Final formatted result:\n{result}")
    return result

# Test function to run without Gradio
def test_recommendations():
    test_inputs = [
        "The Dark Knight, Inception, Interstellar",
        "Toy Story, Finding Nemo, Shrek", 
        "The Shawshank Redemption, Forrest Gump, Pulp Fiction"
    ]
    
    for i, test_input in enumerate(test_inputs):
        print(f"\n{'='*50}")
        print(f"TEST {i+1}: {test_input}")
        print(f"{'='*50}")
        result = recommend_movies(test_input)
        print(f"RESULT:\n{result}")

# Gradio interface
iface = gr.Interface(
    fn=recommend_movies,
    inputs=gr.Textbox(
        label="Movies You Like", 
        placeholder="The Shawshank Redemption, The Godfather, Inception",
        lines=3
    ),
    outputs=gr.Textbox(label="Recommended Movies"),
    concurrency_limit=1,
    title="Personal Movie Recommender",
    description="Enter movies you like separated by commas (we'll assume you rate them highly!)",
    examples=[
        ["The Dark Knight, Inception, Interstellar"],
        ["Toy Story, Finding Nemo, Shrek"],
        ["The Shawshank Redemption, Forrest Gump, Pulp Fiction"]
    ]
)

if __name__ == "__main__":
    # Then launch Gradio
    print("\nLaunching Gradio interface...")
    iface.launch(share=True)
