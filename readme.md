<div align="center">

<img alt="TBC-Logo" src="assets/tbc-logo.png" width="140px"/>

# <samp>🎬 Personalized Movie Recommendation System</samp>

<samp>**Final Project** for the **TBC x Geolab Bootcamp**
A smart, personalized movie recommendation system that suggests films tailored to each user's unique taste.</samp>

</div>

---

## 📂 Project Structure

```
TBC-Final/
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

### Prerequisites

* Python 3.8+
* Key packages:

  * pandas
  * numpy
  * scikit-learn (for similarity calculations)
  * requests (optional, for TMDB API integration)

### Installation

1. Clone this repo
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
   
   > or if you use nix flakes:
   ```bash
   nix develop
   ```

3. Place your datasets (`movies.csv`, `ratings.csv`) inside the `data/` directory.

### Usage

**Train the model, and dump as `hypbrid_model.joblib`:**
```bash
python src/train.py
```

**Run the main program (Gradio interface):**
```bash
python src/main.py
```

This will load data, build user profiles, and output personalized movie recommendations based on input preferences.

**Optional, generatie visualizations**
```bash
python src/visualize.py
```

> Note: there are some optional debugging test files that you can run in src/.

---

## 📄 License
MIT Licensed – See [LICENSE](LICENSE) for details
