# ============================================================
# Streamlit App for ML Assignment
# ------------------------------------------------------------
# This app:
# 1. Loads dataset from GitHub
# 2. Loads trained models (.pkl)
# 3. Applies saved preprocessing (scaler, label encoder)
# 4. Evaluates selected model
# 5. Displays all required metrics
# ============================================================

import streamlit as st
import pandas as pd
import pickle
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score
)

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="ML Classification Models",
    layout="wide"
)

st.title("📊 Machine Learning Classification Models")
st.write("Assignment: Model Training, Evaluation & Deployment")

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
DATA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/winequality-white.csv"
TARGET_COLUMN = "Salary"
PKL_DIR = "./model/pkl_files"
csv_file = "./model/ML_Assignment_2.csv"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl",
}

# ------------------------------------------------------------
# LOAD DATASET
# ------------------------------------------------------------
@st.cache_data
def load_dataset():
    df = pd.read_csv(csv_file)
    return df

# ------------------------------------------------------------
# LOAD PKL FILES
# ------------------------------------------------------------
def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

# ------------------------------------------------------------
# MAIN APP
# ------------------------------------------------------------
try:
    df = load_dataset()
    st.success("✅ Dataset loaded successfully")

    # Debug view (helps avoid column name issues)
    with st.expander("🔍 View Dataset Columns"):
        st.write(df.columns.tolist())

    if TARGET_COLUMN not in df.columns:
        st.error(
            f"❌ Target column '{TARGET_COLUMN}' not found in dataset."
        )
        st.stop()

except Exception as e:
    st.error(f"❌ Error loading dataset: {e}")
    st.stop()

# ------------------------------------------------------------
# MODEL SELECTION
# ------------------------------------------------------------
st.subheader("🔽 Select Model")

model_name = st.selectbox(
    "Choose a classification model:",
    list(MODEL_FILES.keys())
)

# ------------------------------------------------------------
# RUN EVALUATION
# ------------------------------------------------------------
if st.button("🚀 Evaluate Model"):

    try:
        # Load artifacts
        model = load_pickle(os.path.join(PKL_DIR, MODEL_FILES[model_name]))
        scaler = load_pickle(os.path.join(PKL_DIR, "scaler.pkl"))
        target_encoder = load_pickle(os.path.join(PKL_DIR, "target_encoder.pkl"))
        feature_columns = load_pickle(os.path.join(PKL_DIR, "feature_columns.pkl"))

        #copying Data set content
        df_eval = df.copy()

        # 1️⃣ Replace invalid categorical values
        for col in df_eval.columns:
            if df_eval[col].dtype == "object":
                mode_val = df_eval[col].replace(" ?", pd.NA).mode()[0]
                df_eval[col] = df_eval[col].replace(" ?", mode_val)

        # 2️⃣ Label encode categorical columns (same as training)
        categorical_cols = [
            "WorkClass",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
            "education"

        ]
        #for col in categorical_cols:
            #df_eval[col] = label_encoder.transform(df_eval[col])

        # 3️⃣ Separate X and y
        X = df_eval.drop(["Salary", "education-num"], axis=1)

        st.write("Encoder type:", type(target_encoder))
        y = target_encoder.transform(df_eval["Salary"])  

        # 🚨 SAFETY FIX: force y to 1D if encoded incorrectly
        if len(y.shape) > 1:
            y = y.argmax(axis=1)

        st.write("y shape:", y.shape)
        st.write("y sample:", y[:10])  
        
        # 4️⃣ MinMax scale numerical columns (same list)
        numerical_columns = [
            "Age",
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

        # Debug view (helps avoid column name issues)
        with st.expander("🔍 View train Dataset Columns"):
            st.write(X.columns.tolist())

        #st.subheader("Target column (y) - raw values")
        #st.write(y)

        y_pred = model.predict(X)

        st.subheader("DEBUG: y inspection")
        st.write("Type of y:", type(y))
        st.write("y shape:", y.shape)
    
        # If y is numpy array, show first row
        if hasattr(y, "ndim"):
            st.write("y ndim:", y.ndim)
            st.write("First 5 rows of y:", y[:5])

        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        mcc = matthews_corrcoef(y, y_pred)
        
        # Probabilities (for AUC)
        #if hasattr(model, "predict_proba"):
        #    y_prob = model.predict_proba(X)
        #    auc = roc_auc_score(
        #        y,
        #        y_prob,
        #        multi_class="ovr"
        #    )
        #else:
        #    auc = "Not Supported"

        # ----------------------------------------------------
        # DISPLAY RESULTS
        # ----------------------------------------------------
        st.subheader(f"📈 Evaluation Metrics — {model_name}")

        col1, col2, col3 = st.columns(3)

        col1.metric("Accuracy", f"{accuracy:.4f}")
        col1.metric("Precision", f"{precision:.4f}")

        col2.metric("Recall", f"{recall:.4f}")
        col2.metric("F1 Score", f"{f1:.4f}")

        col3.metric("MCC Score", f"{mcc:.4f}")
        #col3.metric("AUC Score", auc if isinstance(auc, str) else f"{auc:.4f}")

        st.success("✅ Model evaluation completed successfully")

    except FileNotFoundError:
        st.error(
            "❌ Model or preprocessing files not found. "
            "Please ensure `.pkl` files exist in `pkl_files/`."
        )

    except Exception as e:
        st.error(f"❌ Error during evaluation: {e}")
