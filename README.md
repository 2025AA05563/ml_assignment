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
| Logistic Regression |0.7992  |0.7920  |0.8171 |0.2136  | 0.3387 |0.3494  |
| Decision Tree | 0.7935 |0.5789  |0.8780  |0.1652  |0.2780  |0.3246  |
| KNN |0.7832  |0.7447  |0.5689  |0.4107  |0.4770  |0.3520  |
| Naive Bayes |0.2407  |0.5  |0.2407  |1.00  |0.3881  |0.00  |
| Random Forest (Ensemble) |0.7998  |0.7825  |0.9889  |0.1703  |0.2905  |0.3639  |
| XGBoost (Ensemble) |0.8142  |0.8334  | 0.9761 |0.2341  |0.3776  |0.4257  |


## Observations on Model Performance

| ML Model Name | Observation |
|--------------|-------------|
| Logistic Regression | Achieved overall good accuracy and AUC, which shows stable performance but Recall is very low which suggests model struggles to identify positive income cases. |
| Decision Tree | Model is performed with high precision but poor recall and AUC, indicating that overfitting and weak generalization. Missing most positive cases |
| KNN | Provided a more balanced precision–recall tradeoff compared to Logistic and Decision tree models. Moderate F1 and MCC indicate consistent but not optimal classification performance |
| Naive Bayes | Extremely low precision, accuracy, and MCC indicate severe class imbalance bias and poor overall reliability. |
| Random Forest (Ensemble) | Delivered very high precision and solid accuracy, confirming strong ensemble learning capability. |
| XGBoost (Ensemble) | Achieved the best overall performance with highest accuracy, AUC, and MCC. Excellent precision with improved recall, making it the most reliable and well-balanced model.|

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
