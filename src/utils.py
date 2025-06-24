import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler

class Utils:
    def __init__(self, data_path:str) -> None:
        self.data_path = Path(data_path)

    def read_merge_csv(self, movies_file:str, ratings_file:str) -> pd.DataFrame:
        movies_path = self.data_path / movies_file
        assert movies_path.exists(), f"File not found: {movies_path}"
        
        ratings_path = self.data_path / ratings_file 
        assert ratings_path.exists(), f"File not found: {ratings_path}"

        movies = pd.read_csv(movies_path)
        ratings = pd.read_csv(ratings_path)
        
        merged_data = pd.merge(ratings, movies, on="movieId")

        return merged_data

    def correlation_heatmap(self, data: pd.DataFrame) -> None:
        num_cols = data.select_dtypes(include=['float64', 'int64']).columns
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data[num_cols])
        scaled_df = pd.DataFrame(scaled_data, columns=num_cols)

        plt.figure(figsize=(10, 6))
        sns.heatmap(scaled_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Heatmap (Scaled Numerical Data)')
        plt.show()
