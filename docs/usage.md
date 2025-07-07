## 🚀 Getting Started

### 📋 Prerequisites

---
> Note: this model creates sparse matrixes, and can be heavy for ram!
>
> Recommended ram for deployment system is at least 32GB!
---

Before running the Personal Movie Recommender, ensure you have the following installed:

* **💻 [WSL (Windows Subsystem for Linux)](https://learn.microsoft.com/en-us/windows/wsl/about)** this is necessary for [faiss](https://github.com/facebookresearch/faiss) library. (only works on linux & macos)
* **🐍 Python 3.8+** (Nix shell provides Python 3.13)
* **📦 Required packages:**

  * `pip` — Python package installer
  * `pandas` — for data manipulation and analysis
  * `matplotlib` — for plotting and visualizations
  * `seaborn` — statistical data visualization based on matplotlib
  * `scikit-learn` — for similarity calculations and machine learning
  * `gradio` — for the interactive web interface
  * `faiss` — efficient similarity search library

---

### 🔧 Installation

1. **📥 Clone the repository:**
   ```bash
   git clone https://github.com/Gurjaka/Personal-movie-recommendation-model.git
   cd Personal-movie-recommendation-model
   ```

2. **⚙️ Install dependencies:**
   
   Using pip:
   ```bash
   pip install -r requirements.txt
   ```
   
   Or if you use Nix flakes:
   ```bash
   nix develop
   ```

3. **📊 Prepare your data:**
   
   Download the required datasets and place them in the `data/` directory:
   - `movies.csv` - movie metadata (title, genres, year, etc.)
   - `ratings.csv` - user ratings data
   
   You can obtain these datasets from [Kaggle: Movie Recommendation System Dataset](https://www.kaggle.com/datasets/parasharmanas/movie-recommendation-system).

### 🎯 Usage

#### 1. 🧠 Train the Recommendation Model

First, train the hybrid recommendation model:

```bash
python src/train.py
```

This will:
- 📖 Load and preprocess the movie and ratings data
- 🔧 Build collaborative filtering and content-based models
- 💾 Save the trained model as `hybrid_model.joblib`

#### 2. 🛠️ Run the Data Debugger

```bash
python src/data_debug.py
```

This script will help you inspect and verify your dataset. It will:
- 📐 Print the shape (rows × columns) of the raw movies.csv and ratings.csv
- 📝 Show the first 5 entries from each dataset
- 🧾 List all columns in the movies.csv file
- 🔄 Preprocess and clean the movie data
- 📊 Display genre distribution and perform basic genre analysis

Use this to ensure your dataset is correctly formatted and loaded before training.

#### 3. ✅ Run a Basic Test Before Deployment

```bash
python src/simple_test.py
```

This script performs a sanity check to confirm the model is functioning. It will:
- 🧠 Load the model and required similarity data
- ⚙️ Generate the cosine_sim.npy file (if it doesn't already exist) — this precomputed similarity matrix helps reduce RAM usage and speed up recommendations
- 🎬 Run the model on a few predefined test cases
- 🖨️ Print out sample recommendations to the terminal

Run this after training to confirm everything works as expected before launching the interface.

#### 4. 🌐 Launch the Web Interface

Start the Gradio web application:

```bash
python src/main.py
```

This will:
- 🚀 Load the trained model
- 🖥️ Launch an interactive web interface
- 🎭 Allow you to input preferences and get personalized recommendations

The interface will be available at `http://localhost:7860` by default.

#### 5. 📈 Generate Visualizations (Optional)

Create data visualizations and analysis charts:

```bash
python src/visualize.py
```

This generates various plots showing:
- 📊 Rating distributions
- 🎬 Genre popularity
- 👥 User behavior patterns
- 🎯 Model performance metrics

### 🛠️ Development & Testing

The `src/` directory contains additional utility scripts for development:

- `test.py` - 🧪 unit tests for core functionality
- `utils.py` - 🔧 helper functions for data processing
- Various debugging scripts for testing specific components

To run tests:
```bash
python src/test.py
```

### 🔧 Troubleshooting

- **📁 Missing data files**: Ensure `movies.csv` and `ratings.csv` are in the `data/` directory
- **🐏 Memory issues**: For large datasets, consider using data sampling or increasing system memory
- **🔌 Port conflicts**: If port 7860 is occupied, Gradio will automatically use the next available port

### 🎯 Next Steps

- 📚 Explore the API documentation for programmatic access
- 📊 Check out the visualization outputs to understand your data better
- 🔬 Modify the model parameters in `train.py` to experiment with different approaches
