<div align="center">

<img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white"/>
<img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
<img src="https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge"/>
<img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge"/>

<br><br>

# 🧬 Cancer Clustering Analysis
### Unsupervised Machine Learning on Breast Cancer Tumor Data

> *Can machine learning discover malignant vs benign tumors — without ever seeing a single label?*

<br>

[Explore the Notebook](#-usage) · [View Methodology](#-methodology) · [See Results](#-results) · [Get Started](#-quick-start)

</div>

---

## 🧭 Overview

This project applies **unsupervised machine learning** to a breast cancer dataset to uncover natural structure in tumor measurements — without relying on diagnosis labels. By removing ground-truth labels before training, the project simulates real-world scenarios where annotated data may be scarce or unavailable.

Two complementary clustering approaches are compared:

| Approach | Algorithm | Strength |
|---|---|---|
| Partition-based | K-Means | Fast, scalable, globally optimal |
| Hierarchy-based | Agglomerative (Ward) | Reveals nested structure via dendrograms |

Both converge on **2 clusters**, mirroring the binary diagnosis (Malignant / Benign) — achieved with **zero label supervision**.

---

## 📂 Repository Structure

```
cancer-clustering-analysis/
│
├── 📓 Machine_Learning_Cancer_Clustering_Analysis.ipynb   ← Main notebook
│
├── 📁 data/
│   └── ML Project - Clustering Cancer Database.csv       ← Dataset
│
├── 📄 requirements.txt                                    ← Dependencies
└── 📄 README.md
```

---

## 📊 Dataset

| Property | Detail |
|---|---|
| Domain | Oncology / Medical Diagnostics |
| Task | Unsupervised Clustering |
| Instances | ~570 tumor samples |
| Features | 30 numeric tumor measurements |
| Labels used? | ❌ No — dropped before training |

**Key features:**

- `radius_mean` — Mean radius of tumor cell nuclei
- `texture_mean` — Standard deviation of gray-scale values
- + 28 additional numeric measurements (area, smoothness, compactness, concavity, etc.)

**Preprocessing pipeline:**

```
Raw CSV
  └── Drop `id`         (non-informative identifier)
  └── Drop `diagnosis`  (ground-truth label — intentionally excluded)
  └── Null check        → 0 missing values confirmed
  └── MinMaxScaler      (normalization for visualization & distance metrics)
```

---

## 🔬 Methodology

### 1 · Exploratory Data Analysis

- Shape inspection, `.describe()` summary statistics
- Null value audit
- Pairplot of `radius_mean` vs `texture_mean` to understand feature distributions

### 2 · K-Means Clustering

**Elbow Method** — WCSS plotted across `k = 1…9` to find the optimal number of clusters:

```python
wcss = []
for i in range(1, 10):
    km = KMeans(n_clusters=i)
    km.fit(data[["radius_mean", "texture_mean"]])
    wcss.append(km.inertia_)
plt.plot(range(1, 10), wcss)
```

> 📍 The elbow forms clearly at **k = 2**, justifying a 2-cluster solution.

**Final model:**

```python
km = KMeans(n_clusters=2)
data["cluster"] = km.fit_predict(data)
```

### 3 · Agglomerative Hierarchical Clustering

Ward linkage minimises within-cluster variance at each merge step. A **dendrogram** was plotted to visualise the full merge hierarchy before choosing the cut point.

```python
from scipy.cluster.hierarchy import dendrogram, linkage
dendrogram(linkage(data, method="ward"), color_threshold=10000)
```

**Final model:**

```python
agg = AgglomerativeClustering(n_clusters=2, linkage="ward")
data["cluster"] = agg.fit_predict(data)
```

### 4 · Evaluation — Silhouette Score

The **Silhouette Score** measures how similar each point is to its own cluster vs. neighbouring clusters. Score ranges from -1 (wrong cluster) to +1 (well-separated).

```python
from sklearn.metrics import silhouette_score
score = silhouette_score(data, data["cluster"])
```

Both models were evaluated and compared using this metric.

---

## 📈 Results

| Model | Clusters | Silhouette Score |
|---|---|---|
| K-Means | 2 | ✅ Computed in notebook |
| Agglomerative (Ward) | 2 | ✅ Computed in notebook |

**Key findings:**

- ✅ Both algorithms independently identified **2 natural groupings** — consistent with the known binary diagnosis
- ✅ The Elbow Method objectively validated `k = 2` without any label information
- ✅ Scatter plots of `radius_mean` vs `texture_mean` show clear visual separation between clusters
- ✅ Dendrogram confirmed a 2-cluster structure at the highest hierarchy level

---

## ⚡ Quick Start

### Prerequisites

- Python 3.8 or higher
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/cancer-clustering-analysis.git
cd cancer-clustering-analysis

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate       # macOS / Linux
# venv\Scripts\activate        # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### 📓 Usage

```bash
# Launch the notebook
jupyter notebook Machine_Learning_Cancer_Clustering_Analysis.ipynb
```

> **Note:** Update the CSV file path in the data-loading cell to point to your local copy of the dataset before running.

---

## 🛠 Tech Stack

| Library | Version | Purpose |
|---|---|---|
| `pandas` | ≥ 1.3 | Data loading & manipulation |
| `numpy` | ≥ 1.21 | Numerical operations |
| `matplotlib` | ≥ 3.4 | Plotting & visualisation |
| `seaborn` | ≥ 0.11 | Statistical pairplots |
| `scikit-learn` | ≥ 0.24 | KMeans, Agglomerative, MinMaxScaler, Silhouette |
| `scipy` | ≥ 1.7 | Dendrogram & Ward linkage |
| `jupyter` | ≥ 1.0 | Interactive notebook environment |

**`requirements.txt`**

```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=0.24.0
scipy>=1.7.0
jupyter>=1.0.0
```

---

## 🧠 Concepts Demonstrated

- ✦ Unsupervised learning without ground-truth labels
- ✦ K-Means algorithm and centroid-based partitioning
- ✦ Agglomerative hierarchical clustering with Ward linkage
- ✦ Dendrogram construction and interpretation
- ✦ Elbow Method for optimal `k` selection
- ✦ Silhouette Score for internal cluster validation
- ✦ Feature normalisation with MinMaxScaler
- ✦ Data visualisation of high-dimensional clusters in 2D

---

## 🔭 Potential Extensions

- [ ] Apply **PCA** or **t-SNE** to visualise all 30 features in 2D
- [ ] Use **DBSCAN** for density-based, outlier-robust clustering
- [ ] Compare cluster assignments against true labels to compute **purity score**
- [ ] Build an interactive **Streamlit dashboard** for live cluster exploration
- [ ] Tune hyperparameters using **Grid Search** on Silhouette Score

---

## 👤 Author

**Your Name**
- GitHub: [@AMAAN PETIWALA](https://github.com/ssp30)
- LinkedIn: [Amaan Petiwala](https://linkedin.com/in/your-profile)

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

*Built for learning. Driven by curiosity. Powered by data.*

⭐ **If you found this useful, consider starring the repo!**

</div>
