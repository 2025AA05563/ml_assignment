# Importing Streamlit (used to create web UI)
import streamlit as st    # Streamlit UI framework

# Import core libraries
import pandas as pd
import matplotlib.pyplot as plt

# Import sklearn utilities for evaluation
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Import our own helper functions
from model.data_creation import data_load_preprocess
from model.model_evaluation import evaluate_model
from model.models_training import train_model


# ------------------------------------------------------------
# 🧠 3. MODEL SELECTION MAPPING
# ------------------------------------------------------------
# Maps dropdown option → corresponding training function

MODEL_MAP = {
    "Logistic Regression": logistic_model,
    "Decision Tree": decision_tree_model,
    "K-Nearest Neighbors (KNN)": knn_model,
    "Naive Bayes": naive_bayes_model,
    "Random Forest (Ensemble)": random_forest_model,
    "XGBoost (Ensemble)": xgboost_model
}


# ------------------------------------------------------------
# 🖥️ 4. STREAMLIT PAGE HEADER
# ------------------------------------------------------------

st.set_page_config(
    page_title="📊 Machine Learning Assignment – Classification Models",
    layout="centered"
)
#st.title("📊 Machine Learning Assignment – Classification Models")

st.markdown(
    """
    🔹 Upload a CSV dataset  
    🔹 Select a machine learning model  
    🔹 Evaluate performance using multiple metrics  
    🔹 Visualize results using a confusion matrix  
    """
)


# ------------------------------------------------------------
# 📂 5. USER INPUT SECTION
# ------------------------------------------------------------

# 📁 Dataset upload
uploaded_file = st.file_uploader(
    "📁 Upload CSV Dataset (Test Data Only)",
    type=["csv"]
)

# 🎯 Target column name input
target_column = st.text_input(
    "🎯 Enter Target Column Name (exactly as in CSV)"
)

# 🤖 Model selection dropdown
model_name = st.selectbox(
    "🤖 Select Classification Model",
    list(MODEL_MAP.keys())
)


# ------------------------------------------------------------
# ▶️ 6. MAIN APPLICATION LOGIC
# ------------------------------------------------------------

# Proceed only if dataset and target column are provided
if uploaded_file is not None and target_column != "":

    # --------------------------------------------------------
    # 👀 Dataset Preview
    # --------------------------------------------------------
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    # --------------------------------------------------------
    # ⚙️ Data Preprocessing
    # --------------------------------------------------------
    try:
        X_train, X_test, y_train, y_test = load_and_preprocess(
            uploaded_file,
            target_column
        )
    except Exception:
        st.error("❌ Error during preprocessing. Please check target column name.")
        st.stop()

    # --------------------------------------------------------
    # 🏋️ Model Training
    # --------------------------------------------------------
    st.subheader("🏋️ Model Training")

    # Get the selected model function
    train_function = MODEL_MAP[model_name]

    # Train model
    model = train_function(X_train, y_train)

    st.success(f"✅ {model_name} trained successfully!")

    # --------------------------------------------------------
    # 🔮 Model Prediction
    # --------------------------------------------------------
    y_pred = model.predict(X_test)

    # Check for probability prediction (needed for AUC)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)
    else:
        y_prob = None

    # --------------------------------------------------------
    # 📊 Evaluation Metrics
    # --------------------------------------------------------
    st.subheader("📊 Evaluation Metrics")

    metrics = evaluate_model(y_test, y_pred, y_prob)

    # Display metrics clearly
    for metric, value in metrics.items():
        st.write(f"🔹 **{metric}:** {value}")

    # --------------------------------------------------------
    # 🔢 Confusion Matrix Visualization
    # --------------------------------------------------------
    st.subheader("🔢 Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(cm).plot(ax=ax)
    st.pyplot(fig)

else:
    # --------------------------------------------------------
    # ℹ️ User Guidance Message
    # --------------------------------------------------------
    st.info("⬆️ Please upload a CSV file and enter the target column name to continue.")
