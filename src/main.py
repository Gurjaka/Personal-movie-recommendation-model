from utils import *

if __name__ == "__main__":
    utils = Utils("data/")

    data_frame = utils.read_merge_csv("movies.csv", "ratings.csv")
    print(data_frame.head())
    
    utils.correlation_heatmap(data_frame)

