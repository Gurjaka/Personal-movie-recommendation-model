<div align="center">

<img alt="TBC Logo" src="docs/assets/tbc-logo.png" width="140" />

# 🎬 Personalized Movie Recommendation System

**Final Project** for the **TBC x Geolab Bootcamp**  
A smart, personalized movie recommendation system that suggests films tailored to each user's unique taste.

</div>

<p align="center">
<a href="https://gurjaka.github.io/Personal-movie-recommendation-model/" target="_blank">📚 View Documentation</a>
</p>

## 🙌 Team

This project was developed as part of the **TBC x Geolab Bootcamp**.

- 👨‍💻 **[Gurjaka](https://github.com/Gurjaka)** – Core development, design, documentation
- 🤝 **[Lol0kv28](https://github.com/Lol0kv28)** – Initial collaboration and feedback

---

## 📂 Project Structure

```

TBC-Final/
├── data/               # Dataset CSV files (movies.csv, ratings.csv, etc.)
├── src/                # Source code
│   ├── main.py         # Main entry point
│   ├── train.py        # Model training scripts
│   ├── utils.py        # Utility functions
│   └── test.py         # Unit and debugging tests
├── .envrc              # Environment config (direnv)
├── .gitignore          # Git ignore rules
├── flake.nix           # Nix shell configuration
└── flake.lock          # Nix lock file

````

---

## 📥 Dataset

Download the required datasets from Kaggle:  
[Movie Recommendation System Dataset](https://www.kaggle.com/datasets/parasharmanas/movie-recommendation-system)  

Place the files (`movies.csv`, `ratings.csv`, etc.) inside the `data/` folder.

---

## ✨ Features

- 📊 Efficient loading and merging of movie metadata and user ratings  
- 👤 Builds personalized user profiles based on favorite movies, genres, and rating timestamps  
- 🔍 Finds users with similar tastes for collaborative recommendations  
- 🎯 Recommends highly rated movies from similar users that the target user hasn’t seen  
- 🤝 Combines content-based (genres) and collaborative filtering (user similarity) for hybrid recommendations

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+  
- Key Python packages:  
  - pandas  
  - numpy  
  - scikit-learn (for similarity calculations)  
  - requests (optional, for TMDB API integration)  

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Gurjaka/Personal-movie-recommendation-model.git
   cd Personal-movie-recommendation-model
   ```

2. Install dependencies:

   Using pip:

   ```bash
   pip install -r requirements.txt
   ```

   Or if you use Nix flakes:

   ```bash
   nix develop
   ```

3. Add datasets (`movies.csv`, `ratings.csv`) to the `data/` directory.

---

### Usage

* Train the model and save as `hybrid_model.joblib`:

  ```bash
  python src/train.py
  ```

* Run the main application (Gradio interface):

  ```bash
  python src/main.py
  ```

  This will load data, build user profiles, and provide personalized movie recommendations.

* (Optional) Generate visualizations:

  ```bash
  python src/visualize.py
  ```

---

> **Note:** Additional debugging and test scripts are available in the `src/` directory.

---

## 📌 Branding Notice

This project was created as part of the **TBC x Geolab Bootcamp**.  
The **TBC logo** and related branding are owned by their respective entities and may not be reused, modified, or redistributed without permission.

Feel free to fork or use the code under the terms of the MIT license — but **do not use the TBC logo or project branding** in your own versions or hosted apps.

## 📄 License

MIT Licensed — see [LICENSE](LICENSE) for details.

