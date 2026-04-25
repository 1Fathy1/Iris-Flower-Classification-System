# 🌸 Iris Flower Classification using Machine Learning

## 📌 Project Overview

This project focuses on analyzing and classifying iris flowers into three species using Machine Learning techniques.
The dataset contains measurements of iris flowers, and the goal is to build a model that can accurately predict the species based on these features.

---

## 📊 Dataset Information

* Dataset Name: **Iris Dataset**
* Number of Samples: 150
* Features:

  * Sepal Length
  * Sepal Width
  * Petal Length
  * Petal Width
* Target:

  * Species (Setosa, Versicolor, Virginica)

---

## 🧹 Data Preprocessing

* Loaded dataset from `.data` file
* Added column names manually
* Removed missing values (if any)
* Encoded categorical target variable using Label Encoding

---

## 📈 Data Visualization

The following visualizations were used:

* Pairplot (to visualize relationships between features)
* Heatmap (to check feature correlations)
* Boxplot (to detect outliers)
* Scatter Plot (to visualize class separation)

---

## 🤖 Machine Learning Models

The following models were implemented:

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Decision Tree

---

## ⚙️ Requirements

Install the required libraries using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## 🚀 How to Run

1. Place the dataset file (`iris.data`) in the project folder
2. Run the Python script:

```bash
python main.py
```

---

## 📊 Output

The project provides:

* Model Accuracy
* Confusion Matrix
* Classification Report (Precision, Recall, F1-score)

```
First 5 rows:
    sepal_length  sepal_width  petal_length  petal_width      species
0           5.1          3.5           1.4          0.2  Iris-setosa
1           4.9          3.0           1.4          0.2  Iris-setosa
2           4.7          3.2           1.3          0.2  Iris-setosa
3           4.6          3.1           1.5          0.2  Iris-setosa
4           5.0          3.6           1.4          0.2  Iris-setosa

Info:

<class 'pandas.DataFrame'>
RangeIndex: 150 entries, 0 to 149
Data columns (total 5 columns):
 #   Column        Non-Null Count  Dtype  
---  ------        --------------  -----  
 0   sepal_length  150 non-null    float64
 1   sepal_width   150 non-null    float64
 2   petal_length  150 non-null    float64
 3   petal_width   150 non-null    float64
 4   species       150 non-null    str    
dtypes: float64(4), str(1)
memory usage: 6.0 KB
None

Missing Values:
 sepal_length    0
sepal_width     0
petal_length    0
petal_width     0
species         0
dtype: int64

Statistics:
        sepal_length  sepal_width  petal_length  petal_width
count    150.000000   150.000000    150.000000   150.000000
mean       5.843333     3.054000      3.758667     1.198667
std        0.828066     0.433594      1.764420     0.763161
min        4.300000     2.000000      1.000000     0.100000
25%        5.100000     2.800000      1.600000     0.300000
50%        5.800000     3.000000      4.350000     1.300000
75%        6.400000     3.300000      5.100000     1.800000
max        7.900000     4.400000      6.900000     2.500000

=== Logistic Regression ===
Accuracy: 1.0
Confusion Matrix:
 [[10  0  0]
 [ 0  9  0]
 [ 0  0 11]]
Classification Report:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00        10
           1       1.00      1.00      1.00         9
           2       1.00      1.00      1.00        11

    accuracy                           1.00        30
   macro avg       1.00      1.00      1.00        30
weighted avg       1.00      1.00      1.00        30


=== KNN ===
Accuracy: 1.0
Confusion Matrix:
 [[10  0  0]
 [ 0  9  0]
 [ 0  0 11]]
Classification Report:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00        10
           1       1.00      1.00      1.00         9
           2       1.00      1.00      1.00        11

    accuracy                           1.00        30
   macro avg       1.00      1.00      1.00        30
weighted avg       1.00      1.00      1.00        30


=== Decision Tree ===
Accuracy: 1.0
Confusion Matrix:
 [[10  0  0]
 [ 0  9  0]
 [ 0  0 11]]
Classification Report:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00        10
           1       1.00      1.00      1.00         9
           2       1.00      1.00      1.00        11

    accuracy                           1.00        30
   macro avg       1.00      1.00      1.00        30
weighted avg       1.00      1.00      1.00        30


Model Comparison:
LR Accuracy: 1.0
KNN Accuracy: 1.0
DT Accuracy: 1.0
```

---

## 🧠 Key Insights

* Petal length and petal width are the most important features
* The dataset is highly separable
* KNN achieved the highest accuracy in most cases

---

## 🎯 Conclusion

This project demonstrates how simple Machine Learning models can achieve high accuracy on clean and well-structured datasets like Iris.
It also highlights the importance of data visualization in understanding patterns before modeling.

---

## 👨‍💻 Author

[Fathy Moamen](https://github.com/1Fathy1) & [Nader Mohamed](https://github.com/nader035/)

---
