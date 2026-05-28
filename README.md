# 🌸 Task 3: Exploratory Data Analysis (EDA) on Iris Dataset

**Edutech Solution — Data Science Internship**

---

## 📌 Objective
Perform Exploratory Data Analysis (EDA) on the Iris dataset to understand data patterns, visualize distributions, detect correlations, and identify trends.

---

## 🛠️ Tools & Libraries Used
- Python 3.x
- Pandas — data manipulation
- NumPy — numerical operations
- Matplotlib — plotting
- Seaborn — statistical visualizations
- Scikit-learn — loading the Iris dataset

---

## 📂 Dataset
**Iris Dataset** (built into scikit-learn)
- 150 samples, 4 features, 3 species
- Features: Sepal Length, Sepal Width, Petal Length, Petal Width
- Species: Setosa, Versicolor, Virginica

---

## 🔍 Steps Performed

### 1. Summary Statistics
- Loaded dataset and converted to DataFrame
- Checked shape, data types, and missing values
- Generated `.describe()` for mean, std, min, max

### 2. Visualize Distributions
- Histograms for all 4 features
- Boxplots comparing features across species

### 3. Correlation Heatmap
- Calculated correlation matrix
- Visualized with annotated heatmap (coolwarm colormap)

### 4. Detect Patterns
- Pairplot showing relationships between all feature pairs
- Color-coded by species to spot clusters

### 5. Identify Trends
- Scatter plot: Petal Length vs Petal Width
- Clear linear trend visible within each species

---

## 📊 Key Insights

| Finding | Detail |
|---------|--------|
| Most separable species | Setosa (distinct from others) |
| Highest correlation | Petal Length ↔ Petal Width (0.96) |
| Best features for classification | Petal Length & Petal Width |
| Least informative feature | Sepal Width |

---

## 🖼️ Output Plots
- `plot1_distributions.png` — Histograms
- `plot2_boxplots.png` — Boxplots by species
- `plot3_heatmap.png` — Correlation heatmap
- `plot4_pairplot.png` — Pairplot (patterns)
- `plot5_scatter.png` — Petal trend scatter plot

---

## ▶️ How to Run

```bash
# Install dependencies
pip install pandas matplotlib seaborn scikit-learn

# Run the script
python iris_eda.py
```

---

## 💡 Interview Q&A

**Q: What is EDA?**  
EDA (Exploratory Data Analysis) is the process of analyzing a dataset to summarize its main characteristics, often using visual methods. It helps understand patterns, spot anomalies, test hypotheses, and check assumptions before applying machine learning models.

**Q: Why is visualization important?**  
Visualization makes complex data easy to understand. It helps identify patterns, trends, correlations, and outliers that would be hard to detect in raw numbers. It also makes findings easy to communicate to non-technical stakeholders.

---


