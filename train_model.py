import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

DATASET_PATH = Path("Telco-Customer-Churn.csv")
MODEL_PATH = Path("Model.sav")

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "Dependents",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "Contract",
    "PaperlessBilling",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = "Churn"


def build_category_maps(df):
    category_maps = {}
    for feature in CATEGORICAL_FEATURES:
        values = sorted(df[feature].astype(str).str.strip().fillna("Unknown").unique())
        category_maps[feature] = {value: idx for idx, value in enumerate(values)}
    return category_maps


def build_model():
    df = pd.read_csv(DATASET_PATH)
    df = df.copy()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")

    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])
    df["MonthlyCharges"] = df["MonthlyCharges"].fillna(0)
    df["tenure"] = df["tenure"].fillna(0)

    for feature in CATEGORICAL_FEATURES:
        df[feature] = df[feature].fillna("Unknown")

    target_map = {"Yes": 1, "No": 0}
    df[TARGET] = df[TARGET].map(target_map)

    X = df[FEATURES].copy()
    y = df[TARGET]

    category_maps = build_category_maps(df)
    for feature in CATEGORICAL_FEATURES:
        X[feature] = X[feature].astype(str).str.strip().fillna("Unknown")
        X[feature] = X[feature].map(category_maps[feature]).fillna(-1)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    artifact = {"model": model, "category_maps": category_maps, "features": FEATURES}
    with open(MODEL_PATH, "wb") as handle:
        pickle.dump(artifact, handle)

    print(f"Model trained and saved to {MODEL_PATH}")
    print(f"Accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    build_model()
