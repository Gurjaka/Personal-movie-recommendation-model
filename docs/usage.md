## 🚀 Getting Started

### 📋 Prerequisites

Before running the Personal Movie Recommender, ensure you have the following installed:

* **🐍 Python 3.8+** (Your nix shell provides Python 3)
* **📦 Required packages:**

  * `pip` — Python package installer
  * `pandas` — for data manipulation and analysis
  * `matplotlib` — for plotting and visualizations
  * `seaborn` — statistical data visualization based on matplotlib
  * `scikit-learn` — for similarity calculations and machine learning
  * `gradio` — for the interactive web interface
  * `faiss` — efficient similarity search library
  * `mkdocs` — for building documentation website
  * `mkdocs-material` — Material Design theme for MkDocs
  * `mkdocstrings-python` — for generating API documentation from Python docstrings

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
   
   You can obtain these datasets from [Kaggle: Movie Recommendation System Dataset](https://www.kaggle.com/datasets/example).

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

#### 2. 🌐 Launch the Web Interface

Start the Gradio web application:

```bash
python src/main.py
```

This will:
- 🚀 Load the trained model
- 🖥️ Launch an interactive web interface
- 🎭 Allow you to input preferences and get personalized recommendations

The interface will be available at `http://localhost:7860` by default.

#### 3. 📈 Generate Visualizations (Optional)

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
