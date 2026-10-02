# robustness.py -- tests how models degrade with missing/added symptoms
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from train import load_dataset

RNG = np.random.default_rng(42)

def corrupt(X, drop_rate=0.0, add_rate=0.0):
    X = X.copy()
    for row in range(X.shape[0]):
        present = np.flatnonzero(X[row])
        if drop_rate and len(present):
            n_drop = int(round(drop_rate * len(present)))
            if n_drop:
                X[row, RNG.choice(present, n_drop, replace=False)] = 0
        if add_rate:
            absent = np.flatnonzero(X[row] == 0)
            n_add = int(round(add_rate * len(present)))
            if n_add and len(absent):
                X[row, RNG.choice(absent, min(n_add, len(absent)), replace=False)] = 1
    return X

if __name__ == "__main__":
    X, y, _ = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "NaiveBayes": MultinomialNB(),
    }
    for m in models.values():
        m.fit(X_train, y_train)

    print(f"{'drop':>5} {'add':>5} | " + " ".join(f"{n:>13}" for n in models))
    for drop, add in [(0.0, 0.0), (0.2, 0.0), (0.4, 0.0), (0.0, 0.2), (0.2, 0.2), (0.4, 0.4)]:
        Xc = corrupt(X_test, drop, add)
        row = " ".join(f"{accuracy_score(y_test, m.predict(Xc)):13.3f}" for m in models.values())
        print(f"{drop:5.1f} {add:5.1f} | {row}")
