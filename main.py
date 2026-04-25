# =========================
# 1. Import Libraries
# =========================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =========================
# 2. Load Dataset (iris.data)
# =========================
df = pd.read_csv("iris.data", header=None)

# Add column names
df.columns = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "species"
]

# Remove any empty rows (important)
df = df.dropna()

# =========================
# 3. Data Exploration
# =========================
print("First 5 rows:\n", df.head())
print("\nInfo:\n")
print(df.info())

print("\nMissing Values:\n", df.isnull().sum())

print("\nStatistics:\n", df.describe())

# =========================
# 4. Data Preprocessing
# =========================

# Encode target (species → numbers)
le = LabelEncoder()
df["species"] = le.fit_transform(df["species"])

# Features & Target
X = df.drop("species", axis=1)
y = df["species"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 5. Visualization
# =========================

# Pairplot
sns.pairplot(df, hue="species")
plt.show()

# Heatmap
plt.figure(figsize=(6,4))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.show()

# Boxplot
plt.figure(figsize=(8,5))
sns.boxplot(data=df)
plt.show()

# Scatter
plt.figure(figsize=(6,4))
sns.scatterplot(
    x="petal_length",
    y="petal_width",
    hue="species",
    data=df
)
plt.show()

# =========================
# 6. Models
# =========================

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

# =========================
# 7. Evaluation
# =========================

def evaluate_model(name, y_test, y_pred):
    print(f"\n=== {name} ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

evaluate_model("Logistic Regression", y_test, lr_pred)
evaluate_model("KNN", y_test, knn_pred)
evaluate_model("Decision Tree", y_test, dt_pred)

# =========================
# 8. Compare Results
# =========================
print("\nModel Comparison:")
print("LR Accuracy:", accuracy_score(y_test, lr_pred))
print("KNN Accuracy:", accuracy_score(y_test, knn_pred))
print("DT Accuracy:", accuracy_score(y_test, dt_pred))