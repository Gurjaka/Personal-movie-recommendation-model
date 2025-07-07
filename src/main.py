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
    """
    Format movie recommendations as a Markdown table.

    This function converts a DataFrame of movie recommendations into a formatted
    Markdown table with movie titles and genres. Useful for displaying recommendations
    in Markdown-compatible interfaces or documentation.

    Example:
        ```
        >>> df = pd.DataFrame({
        ...     'title': ['The Matrix', 'Inception'],
        ...     'genres': [['Action', 'Sci-Fi'], ['Action', 'Thriller']]
        ... })
        >>> print(format_recommendations_markdown(df))
        | Title | Genres |
        |-------|--------|
        | The Matrix | Action, Sci-Fi |
        | Inception | Action, Thriller |
        ```
    """
    if df.empty:
        return "No recommendations found."

    lines = ["| Title | Genres |", "|-------|--------|"]
    for _, row in df.iterrows():
        title = row['title']
        genres = ', '.join(row['genres']) if isinstance(row['genres'], list) else row['genres']
        lines.append(f"| {title} | {genres} |")
    return '\n'.join(lines)

def format_recommendations(df: pd.DataFrame) -> str:
    """
    Format movie recommendations as a user-friendly text display.

    This function converts a DataFrame of movie recommendations into a visually
    appealing text format with emojis and clear formatting. Designed for display
    in the Gradio interface output.

    Example:
        ```
        >>> df = pd.DataFrame({
        ...     'title': ['The Matrix', 'Inception'],
        ...     'genres': [['Action', 'Sci-Fi'], ['Action', 'Thriller']]
        ... })
        >>> print(format_recommendations(df))
        🎬 The Matrix
           📂 Genres: Action, Sci-Fi

        🎬 Inception
           📂 Genres: Action, Thriller
        ```
    """
    if df.empty:
        return "No recommendations found."

    lines = []
    for _, row in df.iterrows():
        title = row['title']
        genres = ', '.join(row['genres']) if isinstance(row['genres'], list) else row['genres']
        lines.append(f"🎬 {title}\n   📂 Genres: {genres}")
    return '\n\n'.join(lines)

def recommend_movies(user_input: str):
    """
    Generate movie recommendations based on user input.

    This is the main function that powers the movie recommendation system.
    It parses user input, creates a rating profile, and generates recommendations
    using either the hybrid model (if available) or content-based filtering.
    """
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
    """
    Test the recommendation system with predefined movie inputs.
    
    This function runs automated tests on the recommendation system using
    different genres and movie combinations to verify functionality.
    Useful for debugging and ensuring the system works correctly before
    deployment.
    """
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
with gr.Blocks() as demo:
    # upload TBC-Logo
    gr.Image(
        value="assets/tbc-logo.png", 
        interactive=False, 
        show_label=False, 
        height=120, 
        width=120,
        show_download_button=False,
        show_fullscreen_button=False,
    )

    # model interface
    gr.Interface(
        fn=recommend_movies,
        inputs=gr.Textbox(
            label="Movies You Like", 
            placeholder="The Shawshank Redemption, The Godfather, Inception",
            lines=3
        ),
        outputs=gr.Textbox(label="Recommended Movies"),
        concurrency_limit=1,
        title="Personal Movie Recommender",
        description='<div align="center">Enter movies you like separated by commas (we\'ll assume you rate them highly!</div>',
        examples=[
            ["The Dark Knight, Inception, Interstellar"],
            ["Toy Story, Finding Nemo, Shrek"],
            ["The Shawshank Redemption, Forrest Gump, Pulp Fiction"]
        ]
    )

if __name__ == "__main__":
    """Main execution block for the Gradio movie recommendation application"""
    # Then launch Gradio
    print("\nLaunching Gradio interface...")
    demo.launch(share=True)
