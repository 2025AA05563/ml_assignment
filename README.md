# ml_assignment - Classification models

## Problem Statement
The objective of this assignment is to implement, evaluate, and compare multiple machine learning classification models on a real-world dataset. The models are integrated into an interactive Streamlit web application and deployed using Streamlit Community Cloud to demonstrate end-to-end ML deployment.
---
## Dataset Description
- **Dataset Name:** Adult (Salary / Income prediction)
- **Source:** UCI Machine Learning Repository
- **Problem Type:** Binary Classification
- **Number of Instances:** 32561
- **Number of Features:** 14 (excluding target variable)
- **Target Variable:** income

The dataset was preprocessed by handling label encoding (if required) and feature scaling with Min-max standardization to ensure fair comparison across models.

## Models Used
The following six classification models were implemented and evaluated on the same dataset
- Logistic Regression
- Decision Tree
- K-Nearest Neighbor (KNN )
- Naive Bayes (Gaussian)
- Random Forest
- XGBoost

## Model Performance Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|--------------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression |  |  |  |  |  |  |
| Decision Tree |  |  |  |  |  |  |
| KNN |  |  |  |  |  |  |
| Naive Bayes |  |  |  |  |  |  |
| Random Forest (Ensemble) |  |  |  |  |  |  |
| XGBoost (Ensemble) |  |  |  |  |  |  |


## Observations on Model Performance

| ML Model Name | Observation |
|--------------|-------------|
| Logistic Regression | Performs well on linearly separable data but may underperform on complex patterns. |
| Decision Tree | Captures non-linear relationships but is prone to overfitting. |
| KNN | Sensitive to feature scaling and choice of k value. |
| Naive Bayes | Computationally efficient but assumes feature independence. |
| Random Forest (Ensemble) | Provides robust performance by reducing overfitting through ensemble learning. |
| XGBoost (Ensemble) | Achieves the best overall per

## Streamlit Web Application Features
The deployed Streamlit application includes the following functionalities:

- CSV dataset upload option (test data only)
- Model selection dropdown
- Display of evaluation metrics
- Confusion matrix visualization

---

## Deployment
The application is deployed on **Streamlit Community Cloud** and is accessible via
a public URL. The deployment uses the same GitHub repository and requirements file.
