from utils import *

if __name__ == "__main__":
    utils = Utils("data/")

    data_frame = utils.read_merge_csv("movies.csv", "ratings.csv")
    # print(data_frame.head())
    
    # utils.correlation_heatmap(data_frame)
    user_item = utils.user_movie_matrix(data_frame)
    user_id = int(input("User ID: "))
    recommendations = utils.recommend_movies(user_id, user_item, top_n=5)
    
    return_text = f"Top 5 movie recommendations for user {user_id}:"

    print("-" * len(return_text))
    print(return_text)
    print("-" * len(return_text))
    for title, score in recommendations.items():
        print(f"{title} — {score:.2f}")
    print("-" * len(return_text))
