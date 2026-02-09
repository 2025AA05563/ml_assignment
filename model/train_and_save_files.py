# ============================================================
# 📌 Script: train_and_save_models.py
# ------------------------------------------------------------
# This script:
# 1. Loads dataset from GitHub
# 2. Preprocesses data
# 3. Trains all 6 ML models
# 4. Evaluates models
# 5. Saves models and artifacts as .pkl files
# ============================================================

import os
import pickle

# Import project modules
from data_creation import data_load_preprocess
from models_training import train_model
from model_evaluation import evaluate_model


# ------------------------------------------------------------
# 📌 Dataset configuration
# ------------------------------------------------------------
print("Loading dataset...")
csv_file = "ML_Assignment_2.csv"
X_train, X_test, y_train, y_test = data_load_preprocess(csv_file)

print("Preparing data...")
#X_train, X_test, y_train, y_test, scaler, label_encoder = data_load_preprocess(
#    df,
#    TARGET_COLUMN
#)



# ------------------------------------------------------------
# 📌 Models to train
# ------------------------------------------------------------
MODEL_NAMES = [
    "Logistic Regression",
    "Decision Tree",
    "KNN",
    "Naive Bayes",
    "Random Forest",
    "XGBoost"
]

all_metrics = {}


# ------------------------------------------------------------
# 📌 Train, evaluate, and save each model
# ------------------------------------------------------------
for model_name in MODEL_NAMES:
    print(f"\nTraining model: {model_name}")

    # Train model
    model = train_model(model_name, X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    # Evaluation
    metrics = evaluate_model(y_test, y_pred, y_prob)
    all_metrics[model_name] = metrics

    # Save trained model
    model_file = os.path.join(
        ARTIFACTS_DIR,
        f"{model_name.replace(' ', '_').lower()}.pkl"
    )

    with open(model_file, "wb") as f:
        pickle.dump(model, f)

    print(f"Saved model → {model_file}")


# ------------------------------------------------------------
# 📌 Save preprocessing objects and metrics
# ------------------------------------------------------------
with open(os.path.join(ARTIFACTS_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

with open(os.path.join(ARTIFACTS_DIR, "label_encoder.pkl"), "wb") as f:
    pickle.dump(label_encoder, f)

with open(os.path.join(ARTIFACTS_DIR, "metrics.pkl"), "wb") as f:
    pickle.dump(all_metrics, f)

print("\n✅ All models and artifacts saved successfully")
