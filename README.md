# Credit Card Fraud Detection: Clustering & Classification

This project aims to detect fraudulent credit card transactions using both unsupervised and supervised machine learning techniques. It addresses the significant challenge of "class imbalance" through data resampling and provides a comparative analysis of K-Means Clustering and Logistic Regression.

## 📌 Project Overview

The dataset contains transactions made by credit cards, where most transactions are normal (Class 0) and a very small percentage are fraudulent (Class 1). To achieve high performance, the project follows a rigorous data science pipeline:

1.  **Exploratory Data Analysis (EDA):** Loading and inspecting the `creditcard.csv` dataset.
2.  **Data Preparation:** * Feature scaling using `StandardScaler`.
    * Data splitting into training and testing sets.
    * **Class Balancing:** Handling imbalanced data by oversampling the minority class (fraud cases) to ensure the model doesn't ignore fraud patterns.
3.  **Unsupervised Learning:** Using **K-Means Clustering** to see if fraud can be detected without labels.
4.  **Supervised Learning:** Using **Logistic Regression** to build a robust classifier.
5.  **Evaluation:** Using advanced metrics like Precision, Recall, F1-Score, and Confusion Matrices.

## 🚀 Key Features

* **Handling Imbalance:** Implementation of a manual oversampling strategy to balance the training set.
* **Dual Approach:** Comparison between Clustering (K-Means) and Classification (Logistic Regression).
* **Comprehensive Evaluation:** Goes beyond simple "Accuracy" to look at **Recall** (crucial for fraud detection) and **F1-Score**.
* **Visualization:** * Cluster analysis plots.
    * Confusion Matrix heatmaps using `Seaborn`.

## 🛠️ Technologies Used

* **Python**
* **Pandas & NumPy:** For data manipulation and preprocessing.
* **Scikit-Learn:** For scaling, model training, and metrics.
* **Matplotlib & Seaborn:** For data visualization.

## 📊 Results & Performance

The project evaluates models on:
* **Actual Dataset:** Running K-Means on the original skewed data.
* **Prepared Dataset:** Running K-Means and Logistic Regression on the balanced/scaled data.

> **Note:** In fraud detection, **Recall** is often more important than Accuracy because it measures the system's ability to catch as many fraudulent cases as possible.

## 💻 How to Run

1.  Place the `creditcard.csv` file in the project directory.
2.  Ensure you have the required libraries installed:
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn
    ```
3.  Execute the script:
    ```bash
    python fraud_detection.py
    ```

## 📉 Visualizations

The script generates:
- **Cluster Plots:** Visual representation of how data points are grouped.
- **Confusion Matrix:** A heatmap showing True Positives, False Positives, True Negatives, and False Negatives.

---
Developed as part of a Machine Learning exploration project.
