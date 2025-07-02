import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from utils import DataHandler, Visualizer

class Model:
    @staticmethod
    def train(user_item_matrix: pd.DataFrame):
        # Compute user-user similarity
        similarity = cosine_similarity(user_item_matrix)
        return {
            'matrix': user_item_matrix,
            'similarity': similarity
        }

    @staticmethod
    def save(model, filename: str):
        with open(filename, 'wb') as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename: str):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def recommend(user_id: int, model: dict, top_n=5) -> dict:
        # Find similar users
        user_idx = model['matrix'].index.get_loc(user_id)
        sim_scores = list(enumerate(model['similarity'][user_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

        # Get recommendations
        similar_users = [i for i, _ in sim_scores]
        recommendations = model['matrix'].iloc[similar_users].mean(axis=0)
        recommendations = recommendations.sort_values(ascending=False).head(top_n)

        return recommendations.to_dict()

def train_model():
    # Initialize data handler
    data_handler = DataHandler("data/")

    # Load and clean data
    df = data_handler.load_data("movies.csv", "ratings.csv")
    cleaned = data_handler.clean_data(df)

    # Create user-item matrix
    matrix = data_handler.create_matrix(cleaned)

    # Train model
    model = Model.train(matrix)

    # Save model
    Model.save(model, "movie_model.pkl")

    # Visualize data
    Visualizer.plot_distribution(cleaned)
    Visualizer.plot_heatmap(cleaned.select_dtypes(include='number'))

    print("Model trained and saved successfully!")

if __name__ == "__main__":
    train_model()
