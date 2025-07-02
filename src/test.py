from train import Model
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class Evaluator:
    @staticmethod
    def evaluate(model: dict, test_size=0.2):
        # Prepare data
        X = model['matrix']
        X_train, X_test = train_test_split(X, test_size=test_size, random_state=42)
        
        # Re-train model
        sim_matrix = cosine_similarity(X_train)
        
        # Calculate RMSE
        predictions = []
        actuals = []
        
        for user_id in X_test.index:
            try:
                # Get actual ratings
                user_ratings = X_test.loc[user_id]
                rated_movies = user_ratings[user_ratings > 0].index
                
                # Predict ratings
                preds = Model.recommend(user_id, {'matrix': X_train, 'similarity': sim_matrix}, top_n=len(rated_movies))
                
                # Collect results
                for movie in rated_movies:
                    if movie in preds:
                        predictions.append(preds[movie])
                        actuals.append(user_ratings[movie])
            except KeyError:
                continue
        
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        return rmse
