<div align="center">

<img alt="TBC-Logo" src="assets/tbc-logo.png" width="140px"/>

<h1>🎬 Personalized Movie Recommendation System</h1>
<strong>Final Project</strong> for the <strong>TBC x Geolab Bootcamp</strong><br>
A smart, personalized movie recommendation system that suggests films tailored to each user's unique taste.

<br><br>

<a href="https://www.python.org">
  <img src="https://img.shields.io/badge/python-3.8%2B-blue" alt="Python 3.8+"/>
</a>
<a href="LICENSE">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License"/>
</a>

</div>

---

## 📂 Project Structure

```
Personal-Movie-Recommendation-System/
├── data/               # Dataset CSV files (movies.csv, ratings.csv, etc.)
├── src/                # Source code
│   ├── main.py         # Main entry point
│   ├── train.py        # (Optional) Model training scripts
│   ├── utils.py        # Utility functions for data handling
│   └── test.py         # (Optional) Test scripts
├── .envrc              # Environment config (direnv)
├── .gitignore          # Git ignore rules
├── flake.nix           # Nix shell configuration
└── flake.lock          # Nix lock file
```

---

## 📥 Where to Get the Dataset

You can download the required datasets from Kaggle:
[Movie Recommendation System Dataset](https://www.kaggle.com/datasets/parasharmanas/movie-recommendation-system)

Make sure to download and place the files (`movies.csv`, `ratings.csv`, etc.) inside the `data/` folder

---

## ✨ Features

* 📊 Efficient loading and merging of movie metadata and user ratings
* 👤 Builds personalized user profiles from favorite movies, genres, and rating timestamps
* 🔍 Finds users with similar tastes for collaborative recommendations
* 🎯 Recommends highly rated movies from similar users that the target user hasn’t seen
* 🤝 Combines content-based filtering (genres) with collaborative filtering (user similarity) for hybrid recommendations

---

## 🚀 Getting Started
Check the [Usage Guide](usage.md) to get started.

## 📄 License
MIT Licensed – See [LICENSE](https://github.com/Gurjaka/Personal-movie-recommendation-model/blob/main/LICENSE) for details
