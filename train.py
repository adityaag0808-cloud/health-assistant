import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, f1_score

RANDOM_STATE = 42

def clean(symptom: str) -> str:
    return str(symptom).strip().lower().replace(" ", "_").replace("__", "_")

def load_dataset(path="data/dataset.csv"):
    if not os.path.exists(path):
        train_path = "data/Training.csv"
        if os.path.exists(train_path):
            df = pd.read_csv(train_path)
            cols = [c.strip() for c in df.columns if c.strip() != "prognosis" and not c.startswith("Unnamed")]
            vocabulary = sorted([clean(c) for c in cols])
            X = df[cols].values
            y = df["prognosis"].str.strip().values
            return X, y, vocabulary

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    symptom_cols = [c for c in df.columns if c.lower().startswith("symptom")]

    vocabulary = sorted({clean(v) for v in df[symptom_cols].values.ravel() if pd.notna(v)})
    index = {s: i for i, s in enumerate(vocabulary)}

    X = np.zeros((len(df), len(vocabulary)), dtype=np.int8)
    for row, values in enumerate(df[symptom_cols].values):
        for value in values:
            if pd.notna(value):
                clean_val = clean(value)
                if clean_val in index:
                    X[row, index[clean_val]] = 1

    y = df["Disease"].str.strip().values
    return X, y, vocabulary

def main():
    os.makedirs("models", exist_ok=True)
    X, y, vocabulary = load_dataset()
    print(f"Matrix: {X.shape[0]} records x {X.shape[1]} symptoms, {len(set(y))} diseases")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    candidates = {
        "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "NaiveBayes": MultinomialNB(),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}

    for name, model in candidates.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        results[name] = {
            "cv_mean": float(scores.mean()),
            "cv_std": float(scores.std()),
            "test_accuracy": float(accuracy_score(y_test, predictions)),
            "test_f1_macro": float(f1_score(y_test, predictions, average="macro")),
        }
        print(f"{name:14s} CV={scores.mean():.4f} (+/-{scores.std():.4f}) test acc={results[name]['test_accuracy']:.4f}")

    PREFERENCE = ["RandomForest", "NaiveBayes", "DecisionTree"]
    best_name = min(results, key=lambda k: (-results[k]["test_f1_macro"], -results[k]["cv_mean"], PREFERENCE.index(k)))
    best = candidates[best_name]
    print(f"\nSelected Best Model: {best_name}")

    joblib.dump({
        "model": best,
        "model_name": best_name,
        "vocabulary": vocabulary,
        "classes": list(best.classes_),
    }, "models/disease_model.joblib")

    with open("models/metrics.json", "w") as fh:
        json.dump(results, fh, indent=2)

    print("Saved models/disease_model.joblib and models/metrics.json")

if __name__ == "__main__":
    main()
