# 🧠 Automated ML Pipeline

[![Python](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/) 
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.2.0-orange)](https://scikit-learn.org/) 
[![MLflow](https://img.shields.io/badge/MLflow-2.9.1-green)](https://mlflow.org/)  

An **end-to-end machine learning pipeline** that automates preprocessing, feature engineering, model selection, hyperparameter tuning, evaluation, and model tracking using MLflow. Perfect for binary classification and regression tasks.


## 🚀 Features & Program Capabilities

- **Preprocessing Pipeline**
  - Handles missing values automatically
  - Normalization and scaling of numerical features
  - Encoding of categorical features
- **Feature Engineering**
  - Add or remove features dynamically
  - Support for creating new features from existing data
- **Model Selection**
  - Test multiple models automatically (Random Forest, Logistic Regression, SVM, Decision Tree, etc.)
  - Automatically chooses the best model using cross-validation
- **Hyperparameter Tuning**
  - GridSearchCV and RandomizedSearchCV
  - Optimizes model parameters for best performance
- **Evaluation**
  - Classification: Accuracy, Precision, Recall, F1-score, Confusion Matrix, ROC curve
  - Regression: MAE, MSE, RMSE, R²
  - Generates plots to visualize results and learning curves
- **Save & Load Pipeline**
  - Save fully trained pipelines with preprocessing and model using `joblib` or `pickle`

## 🛠 Program Capabilities

- Supports **binary classification** and **regression tasks**
- Can automatically **select the best model** among multiple candidates
- Performs **hyperparameter tuning** for optimized performance
- Generates **evaluation metrics and visualizations** automatically
- Supports **customizable preprocessing pipelines** for different datasets
- Easy to extend with new **models** or **feature engineering steps**