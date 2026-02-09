# ============================================================
# Streamlit App for ML Assignment
# ------------------------------------------------------------
# This app:
# 1. Loads dataset from UCI url link
# 2. Loads trained models (.pkl)
# 3. Applies saved preprocessing (scaler, label encoder)
# 4. Evaluates selected model
# 5. Displays all required metrics
# ============================================================

import streamlit as st
import pandas as pd
import pickle
import os
import requests
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import seaborn as sns

PKL_DIR = "./model/pkl_files"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl",
}

# Column names for Adult dataset
cols = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="ML Classification Models",
    layout="wide"
)
st.title("📈 Machine Learning Assignment")
st.caption("An interactive Streamlit app for model analysis and visualization")
#st.write("Assignment: Model Training, Evaluation & Deployment")

st.markdown(
    """
    <style>
    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: #1F2937;   /* Dark blue-grey */
        padding: 22px;
    }

    /* Default sidebar text */
    [data-testid="stSidebar"] * {
        color: #E5E7EB;              /* Soft white */
        font-size: 14.5px;
    }

    /* Sidebar headers */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #60A5FA;              /* Light blue */
        font-weight: 600;
    }

    /* Labels (input text, slider text) */
    [data-testid="stSidebar"] label {
        color: #CBD5E1;              /* Light gray */
    }

    /* Buttons */
    [data-testid="stSidebar"] button {
        background-color: #2563EB;  /* Blue */
        color: #FFFFFF;
        border-radius: 6px;
        border: none;
    }

    [data-testid="stSidebar"] button:hover {
        background-color: #1D4ED8;
    }

    /* Selectbox / Text input */
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] select,
    [data-testid="stSidebar"] textarea {
        background-color: #111827;  /* Very dark */
        color: #F9FAFB;
        border-radius: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# SIDEBAR — USER INPUTS / CONTROLS
# --------------------------------------------------
with st.sidebar:
    st.header("⚙️ User Inputs ")

    # Dataset URL
    st.subheader("🔗 Dataset Source")
    dataset_url = st.text_input(
        "Enter Dataset URL (CSV)",
        value="https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    )

    download_btn = st.button("⬇️ Download Dataset")

    # Data Load section
    st.subheader("📂 Data Load")
    load_data_btn = st.button("📥 Load Dataset")

    # Model selection
    st.subheader("🤖 Model Selection")
    model_name = st.selectbox(
        "Select Classification Model",
        [
            "Logistic Regression",
            "Decision Tree",
            "KNN",
            "Naive Bayes",
            "Random Forest",
            "XGBoost"
        ]
    )

   # Train/Test split
    st.subheader("⚖️ Train/Test Split")
    test_size = st.slider(
        "Test Size (%)",
        min_value=10,
        max_value=40,
        value=20,
        step=5
    )

    # Evaluate
    evaluate_btn = st.button("🚀 Evaluate Metrics")

# --------------------------------------------------
# MAIN WINDOW — LAYOUT CONTAINERS (NO OVERLAP)
# --------------------------------------------------
download_container = st.container()
data_container = st.container()
metrics_container = st.container()
cm_container = st.container()

# --------------------------------------------------
# MAIN WINDOW — OUTPUT VISUALIZATION
# --------------------------------------------------
DATA_PATH = "dataset.csv"

# -------------------------
# DOWNLOAD DATASET
# -------------------------
with download_container:
    if download_btn:
        try:
            response = requests.get(dataset_url)
            response.raise_for_status()
            with open(DATA_PATH, "wb") as f:
                f.write(response.content)
            st.success("✅ Dataset downloaded successfully")
        except Exception as e:
            st.error(f"❌ Failed to download dataset: {e}")

# -------------------------
# LOAD DATASET
# -------------------------
with download_container:
    if load_data_btn:

        if not os.path.exists(DATA_PATH):
            st.error("❌ Dataset not found. Please download it first.")
            st.stop()

        df = pd.read_csv(DATA_PATH, names=cols, sep=",", skipinitialspace=True)  #pd.read_csv(DATA_PATH, sep=";")
        st.subheader("📄 Dataset Loaded Successfully")

        col1, col2 = st.columns(2)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])

        st.subheader("🔍 Dataset Preview")
        st.dataframe(df.head(20), use_container_width=True)



# ------------------------------------------------------------
# LOAD PKL FILES
# ------------------------------------------------------------
@st.cache_data
def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)
# -------------------------
# MODEL EVALUATION
# -------------------------
with metrics_container:
    if evaluate_btn:

        if not os.path.exists(DATA_PATH):
            st.error("❌ Please download and load dataset first.")
            st.stop()

        #df = pd.read_csv(DATA_PATH, sep=";")
        df = pd.read_csv(DATA_PATH, names=cols, sep=",", skipinitialspace=True)

        #copying Data set content
        df_eval = df.copy()

        # Debug view (helps avoid column name issues)
        with st.expander("🔍 View train Dataset Columns"):
            st.write(df.columns.tolist())

        # Target column (Wine Quality example)
        TARGET_COLUMN = "income"

        # Remove whitespace
        df_eval["income"] = df_eval["income"].str.strip()

        # Remove trailing period (.)
        df_eval["income"] = df_eval["income"].str.replace(".", "", regex=False)

        # Load artifacts
        model = load_pickle(os.path.join(PKL_DIR, MODEL_FILES[model_name]))
        scaler = load_pickle(os.path.join(PKL_DIR, "scaler.pkl"))
        target_encoder = load_pickle(os.path.join(PKL_DIR, "target_encoder.pkl"))
        feature_columns = load_pickle(os.path.join(PKL_DIR, "feature_columns.pkl"))

        # 1️⃣ Replace invalid categorical values
        for col in df_eval.columns:
            if df_eval[col].dtype == "object":
                mode_val = df_eval[col].replace(" ?", pd.NA).mode()[0]
                df_eval[col] = df_eval[col].replace(" ?", mode_val)

        # 2️⃣ Label encode categorical columns (same as training)
        categorical_cols = [
            "workclass",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
            "education"
        ]

        # 3️⃣ Separate X and y
        X = df_eval.drop(["income", "education-num"], axis=1)
        #st.write("Encoder type:", type(target_encoder))
        y = target_encoder.transform(df_eval["income"]) 

        # 4️⃣ MinMax scale numerical columns (same list)
        numerical_columns = [
            "age",
            "fnlwgt",
            "capital-gain",
            "capital-loss",
            "hours-per-week"
        ]   
        X[numerical_columns] = scaler.transform(X[numerical_columns])

        # 5️⃣ One-hot encode categorical columns
        X = pd.get_dummies(X,  columns=categorical_cols)

        # 6️⃣ ALIGN COLUMNS WITH TRAINING
        X = X.reindex(columns=feature_columns, fill_value=0)


        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size / 100,
            random_state=42,
            stratify=y
        )

    # Model selection
    #if model_name == "Logistic Regression":
    #    model = LogisticRegression(max_iter=1000)
    #elif model_name == "Decision Tree":
    #    model = DecisionTreeClassifier()
    #elif model_name == "KNN":
    #    model = KNeighborsClassifier()
    #elif model_name == "Naive Bayes":
    #    model = GaussianNB()
    #elif model_name == "Random Forest":
    #    model = RandomForestClassifier()

    #model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob) if y_prob is not None else "N/A"

        st.subheader(f"📈 Evaluation Metrics — {model_name}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{accuracy:.4f}")
        c1.metric("Precision", f"{precision:.4f}")
        c2.metric("Recall", f"{recall:.4f}")
        c2.metric("F1 Score", f"{f1:.4f}")
        c3.metric("MCC", f"{mcc:.4f}")
        c3.metric("AUC", auc if isinstance(auc, str) else f"{auc:.4f}")

        with cm_container:
            # Confusion Matrix
            st.subheader("🔢 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)

            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

            st.success("✅ Model evaluation completed successfully")
