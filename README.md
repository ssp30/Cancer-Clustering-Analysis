# 🔬 Cancer Clustering Analysis with Machine Learning

Unsupervised machine learning project that applies **K-Means** and **Agglomerative Hierarchical Clustering** to a breast cancer dataset to discover natural groupings in tumor characteristics — without using diagnosis labels.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Algorithms Used](#algorithms-used)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)

---

## Overview

This project explores whether unsupervised clustering algorithms can recover meaningful groupings in cancer data based solely on tumor measurement features (e.g., radius, texture). The diagnosis label is intentionally dropped before clustering to simulate a real-world scenario where ground truth may be unknown.

---

## Dataset

- **Source:** Breast Cancer clustering dataset (CSV)
- **Features used:** Tumor measurements including `radius_mean`, `texture_mean`, and additional numeric features
- **Preprocessing steps:**
  - Dropped non-informative columns (`id`, `diagnosis`)
  - Checked for and confirmed no missing values
  - Applied **MinMaxScaler** normalization for visualisation and distance-based algorithms

---

## Project Workflow

```
Raw Data → EDA → Drop Labels → Clustering → Evaluation
```

1. **Exploratory Data Analysis (EDA)**
   - Shape, summary statistics, null checks
   - Pairplot of key features (`radius_mean` vs `texture_mean`)

2. **K-Means Clustering**
   - Initial run with `k=3`
   - Elbow Method (WCSS) to determine optimal `k`
   - Final model with `k=2`

3. **Agglomerative Hierarchical Clustering**
   - Dendrogram with Ward linkage to visualise cluster hierarchy
   - Final model with `n_clusters=2`, Ward linkage

4. **Evaluation**
   - Silhouette Score computed for both algorithms
   - Scatter plots of cluster assignments on normalised feature space

---

## Algorithms Used

| Algorithm | Key Parameters | Purpose |
|---|---|---|
| K-Means | `n_clusters=2`, Elbow Method | Partition-based clustering |
| Agglomerative Clustering | `n_clusters=2`, `linkage='ward'` | Hierarchical clustering |
| MinMaxScaler | — | Feature normalisation |
| Silhouette Score | — | Cluster quality evaluation |

---

## Results

- The **Elbow Method** on WCSS indicated **k=2** as the optimal number of clusters, aligning with the binary nature of the diagnosis (Malignant / Benign).
- Both K-Means and Agglomerative Clustering converged on 2 clusters.
- **Silhouette Scores** were computed for both models to compare cluster cohesion and separation.
- Scatter plots of `radius_mean` vs `texture_mean` visually confirmed well-separated clusters.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/cancer-clustering-analysis.git
cd cancer-clustering-analysis

# (Optional) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

1. Place the dataset CSV in a `data/` folder (or update the file path in the notebook).
2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook Machine_Learning_Cancer_Clustering_Analysis.ipynb
   ```
3. Run all cells sequentially.

---

## Dependencies

```
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
jupyter
```

Install all at once:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy jupyter
```

---

## 📁 Project Structure

```
cancer-clustering-analysis/
│
├── data/
│   └── ML Project - Clustering Cancer Database.csv
│
├── Machine_Learning_Cancer_Clustering_Analysis.ipynb
├── requirements.txt
└── README.md
```

---

## License

This project is for educational purposes. Feel free to fork and build upon it.
