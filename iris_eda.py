# ============================================================
# Task 3: Exploratory Data Analysis (EDA) on Iris Dataset
# Edutech Solution - Data Science Internship
# Tools: Python, Matplotlib, Seaborn
# ============================================================

# --- STEP 1: Import Libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Set a nice style for all plots
sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams['figure.figsize'] = (10, 6)

# ============================================================
# STEP 2: Load the Dataset
# ============================================================
iris_data = load_iris()

# Convert to a DataFrame (table format)
df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
df['species'] = pd.Categorical.from_codes(iris_data.target, iris_data.target_names)

print("=" * 60)
print("  IRIS DATASET - EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# ============================================================
# STEP 3: Basic Info About the Dataset
# ============================================================
print("\n📌 First 5 rows of the dataset:")
print(df.head())

print("\n📌 Shape of dataset (rows, columns):", df.shape)

print("\n📌 Dataset Info:")
print(df.info())

print("\n📌 Species distribution (how many flowers per type):")
print(df['species'].value_counts())

# ============================================================
# STEP 4: Summary Statistics (Mean, Min, Max, Std, etc.)
# ============================================================
print("\n📊 Summary Statistics:")
print(df.describe().round(2))

# Check for missing values
print("\n🔍 Missing values in each column:")
print(df.isnull().sum())

# ============================================================
# STEP 5: Visualize Distributions
# ============================================================

# --- Plot 1: Histogram for each feature ---
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Distribution of Each Feature', fontsize=16, fontweight='bold')

features = iris_data.feature_names
colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']

for i, (ax, feature, color) in enumerate(zip(axes.flatten(), features, colors)):
    ax.hist(df[feature], bins=20, color=color, edgecolor='white', alpha=0.85)
    ax.set_title(feature.replace(' (cm)', '').title(), fontsize=12)
    ax.set_xlabel('Value (cm)')
    ax.set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('plot1_distributions.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: plot1_distributions.png")

# --- Plot 2: Boxplot - Compare features across species ---
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Feature Distribution by Species (Boxplot)', fontsize=16, fontweight='bold')

for ax, feature in zip(axes.flatten(), features):
    sns.boxplot(data=df, x='species', y=feature, ax=ax, palette='Set2')
    ax.set_title(feature.replace(' (cm)', '').title(), fontsize=12)
    ax.set_xlabel('Species')
    ax.set_ylabel('Value (cm)')

plt.tight_layout()
plt.savefig('plot2_boxplots.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: plot2_boxplots.png")

# ============================================================
# STEP 6: Correlation Heatmap
# ============================================================
plt.figure(figsize=(9, 7))
corr_matrix = df.drop('species', axis=1).corr().round(2)

sns.heatmap(
    corr_matrix,
    annot=True,          # Show numbers inside boxes
    fmt='.2f',           # Format to 2 decimal places
    cmap='coolwarm',     # Red = high correlation, Blue = low
    linewidths=0.5,
    square=True,
    annot_kws={"size": 12}
)
plt.title('Correlation Heatmap of Iris Features', fontsize=15, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('plot3_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: plot3_heatmap.png")

# ============================================================
# STEP 7: Detect Patterns - Pairplot
# ============================================================
# This shows all features plotted against each other
# Great for seeing which features separate the species

pair = sns.pairplot(df, hue='species', palette='Set2', diag_kind='kde', height=2.5)
pair.fig.suptitle('Pairplot: Patterns Between Features & Species',
                  fontsize=14, fontweight='bold', y=1.02)
plt.savefig('plot4_pairplot.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: plot4_pairplot.png")

# ============================================================
# STEP 8: Identify Trends - Scatter Plot (Petal features)
# ============================================================
plt.figure(figsize=(9, 6))
sns.scatterplot(
    data=df,
    x='petal length (cm)',
    y='petal width (cm)',
    hue='species',
    palette='Set2',
    s=100,              # Dot size
    edgecolor='white'
)
plt.title('Petal Length vs Petal Width by Species', fontsize=14, fontweight='bold')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.legend(title='Species')
plt.tight_layout()
plt.savefig('plot5_scatter.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: plot5_scatter.png")

# ============================================================
# STEP 9: Print Key Insights
# ============================================================
print("\n" + "=" * 60)
print("  KEY INSIGHTS FROM EDA")
print("=" * 60)
print("""
1. 📏 Setosa has the SMALLEST petal length and width.
   Virginica has the LARGEST.

2. 🔗 Petal length and petal width are HIGHLY CORRELATED (0.96).
   This means bigger flowers have both longer and wider petals.

3. 🌸 Setosa is easily separable from the other two species.
   Versicolor and Virginica overlap slightly.

4. 📊 Sepal width has the LOWEST correlation with other features.
   It is less useful for distinguishing species.

5. 🎯 Petal features are MORE USEFUL than sepal features
   for classifying iris species.
""")

print("🎉 EDA Complete! All 5 plots saved as PNG files.")
