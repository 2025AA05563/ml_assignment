# Importing Streamlit (used to create web UI)
import streamlit as st

# Import core libraries
import pandas as pd
import matplotlib.pyplot as plt

# Import sklearn utilities for evaluation
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Import our own helper functions
from model.data_creation import data_load_and_preprocess
from model.metrics import evaluate_model

# Import training functions for all models
from model.logistic_regression import train_model as logistic_model
from model.decision_tree import train_model as decision_tree_model
from model.knn import train_model as knn_model
from model.naive_bayes import train_model as naive_bayes_model
from model.random_forest import train_model as random_forest_model
from model.xgboost_model import train_model as xgboost_model

# --------------------------------------------------------
# Mapping model names (UI) to training functions
# --------------------------------------------------------
MODEL_MAP = {
    "Logistic Regression": logistic_model,
    "Decision Tree": decision_tree_model,
    "K-Nearest Neighbors": knn_model,
    "Naive Bayes": naive_bayes_model,
    "Random Forest (Ensemble)": random_forest_model,
    "XGBoost (Ensemble)": xgboost_model
}

# --------------------------------------------------------
# Streamlit Page Title
# --------------------------------------------------------
st.title("Machine Learning Assignment – Classification Models")

st.write(
    """
    This application allows you to upload a dataset, select a machine learning classification model and evaluate its performance using multiple metrics.
    """
)

# --------------------------------------------------------
# Dataset Upload Section
# --------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Dataset (Test Data Only)",
    type=["csv"]
)

# Target column input
target_column = st.text_input(
    "Enter the Target Column Name (exactly as in CSV)"
)

# Model selection dropdown
model_name = st.selectbox(
    "Select Classification Model",
    list(MODEL_MAP.keys())
)

# --------------------------------------------------------
# Main Logic: Execute only when inputs are provided
# --------------------------------------------------------
if uploaded_file is not None and target_column != "":

    # Load and display dataset
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Dataset Preview")
    st.dataframe(df.head())

    # ----------------------------------------------------
    # Data Preprocessing
    # ----------------------------------------------------
    try:
        X_train, X_test, y_train, y_test = load_and_preprocess(
            uploaded_file,
            target_column
        )
    except Exception as e:
        st.error("Error in preprocessing. Check target column name.")
        st.stop()

    # ----------------------------------------------------
    # Model Training
    # ----------------------------------------------------
    st.subheader("Model Training")

    model_train_function = MODEL_MAP[model_name]
    model = model_train_function(X_train, y_train)

    st.success(f"{model_name} trained successfully!")

    # ----------------------------------------------------
    # Prediction
    # ----------------------------------------------------
    y_pred = model.predict(X_test)

    # Check if model supports probability prediction
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)
    else:
        y_prob = None

    # ----------------------------------------------------
    # Evaluation Metrics
    # ----------------------------------------------------
    st.subheader("Evaluation Metrics")

    metrics = evaluate_model(y_test, y_pred, y_prob)

    # Display metrics neatly
    for metric_name, metric_value in metrics.items():
        st.write(f"**{metric_name}:** {metric_value}")

    # ----------------------------------------------------
    # Confusion Matrix
    # ----------------------------------------------------
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(cm).plot(ax=ax)
    st.pyplot(fig)

else:
    st.info("Please upload a CSV file and enter the target column name.")
