import gradio as gr
from train import Model

def recommend_for_user(user_id: int):
    # Load trained model
    model = Model.load("movie_model.pkl")
    
    # Get recommendations
    recommendations = Model.recommend(user_id, model)
    
    # Format results
    results = [f"{movie} — {score:.2f}" for movie, score in recommendations.items()]
    return "\n".join(results)

# Gradio interface
iface = gr.Interface(
    fn=recommend_for_user,
    inputs=gr.Number(label="User ID"),
    outputs=gr.Textbox(label="Recommendations"),
    title="Movie Recommender",
    description="Enter a User ID to get personalized movie recommendations"
)

if __name__ == "__main__":
    iface.launch()
