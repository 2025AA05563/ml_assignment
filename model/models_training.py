from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def train_model(model_name, X_train, y_train):
    """
    Trains and returns a machine learning model based on model_name.

    Parameters:
    ----------
    model_name : str
        Name of the model selected by user
    X_train : array with Training features
    y_train : array with Training target

    Returns:
    -------
    model : trained ML model
    """

    # --------------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------------
    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=10000)

    # --------------------------------------------------------
    # Decision Tree Classifier
    # --------------------------------------------------------
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)

    # --------------------------------------------------------
    # K-Nearest Neighbors
    # --------------------------------------------------------
    elif model_name == "KNN":
        model = KNeighborsClassifier(n_neighbors=5)

    # --------------------------------------------------------
    # Naive Bayes (Gaussian)
    # --------------------------------------------------------
    elif model_name == "Naive Bayes":
        model = GaussianNB()

    # --------------------------------------------------------
    # Random Forest (Ensemble)
    # --------------------------------------------------------
    elif model_name == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

    # --------------------------------------------------------
    # XGBoost (Ensemble)
    # --------------------------------------------------------
    elif model_name == "XGBoost":
        model = XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss"
        )

    # --------------------------------------------------------
    # Invalid Model Name
    # --------------------------------------------------------
    else:
        raise ValueError("Invalid model name selected")

    # --------------------------------------------------------
    # Train the selected model
    # --------------------------------------------------------
    model.fit(X_train, y_train)

    return model
