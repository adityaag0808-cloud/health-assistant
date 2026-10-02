import json
import pandas as pd
from pathlib import Path
import redflags

VOCABULARY = []
CLASSES = []

# Load training data directly to build the symptom vocabulary & dictionary
try:
    df_train = pd.read_csv("data/Training.csv")
    VOCABULARY = [c.strip() for c in df_train.columns if c.strip() != "prognosis"]
    CLASSES = sorted(df_train["prognosis"].unique().tolist())
except Exception:
    VOCABULARY = ["fever", "headache", "cough", "fatigue", "vomiting", "skin_rash", "chills", "joint_pain"]
    CLASSES = ["Fungal infection", "Allergy", "GERD", "Chronic cholestasis", "Migraine", "Common Cold", "Pneumonia"]

INDEX = {s: i for i, s in enumerate(VOCABULARY)}

def _table(path, key):
    try:
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        return {str(r[key]).strip(): r for _, r in df.iterrows()}
    except Exception:
        return {}

SEVERITY = {}
try:
    _sev = pd.read_csv("data/Symptom-severity.csv")
    _sev.columns = [c.strip() for c in _sev.columns]
    for _, r in _sev.iterrows():
        SEVERITY[str(r["Symptom"]).strip().lower().replace(" ", "_")] = int(r["weight"])
except Exception:
    pass

DESCRIPTIONS = _table("data/symptom_Description.csv", "Disease")
PRECAUTIONS = _table("data/symptom_precaution.csv", "Disease")

def severity_score(symptoms):
    total = sum(SEVERITY.get(s.strip().lower().replace(" ", "_"), 3) for s in symptoms)
    return min(100, round(100 * total / (7 * max(len(symptoms), 1))))

def predict(symptoms, top_k=3):
    level, message = redflags.check(symptoms)

    clean_syms = [s.strip().lower().replace(" ", "_") for s in symptoms]
    known = [s for s in clean_syms if s in INDEX]
    unknown = [s for s in symptoms if s.strip().lower().replace(" ", "_") not in INDEX]

    if not known:
        return {
            "status": "no_known_symptoms",
            "unknown": unknown,
            "redflag_level": level,
            "redflag_message": message
        }

    scores = {}
    try:
        df_train = pd.read_csv("data/Training.csv")
        for disease, group in df_train.groupby("prognosis"):
            match_count = 0
            for s in known:
                if s in group.columns and group[s].iloc[0] == 1:
                    match_count += 1
            if match_count > 0:
                scores[disease] = match_count
    except Exception:
        scores = {CLASSES[0]: 3, CLASSES[1]: 2}

    if not scores:
        sorted_diseases = [(CLASSES[0], 1)]
    else:
        sorted_diseases = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    total_score = sum(s[1] for s in sorted_diseases) or 1
    results = []
    for disease, sc in sorted_diseases:
        prob = round((sc / total_score) * 100, 1)
        row = PRECAUTIONS.get(disease)
        precautions = (
            [str(row[c]).strip() for c in row.index if c.lower().startswith("precaution") and pd.notna(row[c])]
            if row is not None else []
        )
        desc_row = DESCRIPTIONS.get(disease)
        results.append({
            "disease": disease,
            "probability": prob,
            "description": (str(desc_row["Description"]).strip() if desc_row is not None else "Consult your physician for diagnosis."),
            "precautions": precautions
        })

    load = severity_score(symptoms)
    if level == "emergency":
        advice = "Seek emergency medical care immediately."
    elif level == "urgent" or load >= 60:
        advice = "Consult a doctor within 24 hours."
    elif load >= 30:
        advice = "Book an appointment with a doctor in the next few days."
    else:
        advice = "Monitor your symptoms. See a doctor if they worsen or persist beyond three days."

    return {
        "status": "ok",
        "predictions": results,
        "symptom_load": load,
        "advice": advice,
        "redflag_level": level,
        "redflag_message": message,
        "unknown": unknown,
        "confident": results[0]["probability"] >= 40 if results else False
    }
