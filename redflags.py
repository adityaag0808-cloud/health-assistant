# redflags.py -- deterministic safety layer, always evaluated first
EMERGENCY_SETS = [
    ({"chest_pain", "breathlessness"},
     "Chest pain with difficulty breathing can indicate a cardiac emergency."),
    ({"chest_pain", "sweating"},
     "Chest pain with sweating can indicate a heart attack."),
    ({"slurred_speech"},
     "Sudden slurred speech can indicate a stroke."),
    ({"altered_sensorium"},
     "Confusion or altered consciousness needs immediate assessment."),
    ({"breathlessness", "bluish_lips"},
     "Breathlessness with bluish lips indicates low oxygen."),
    ({"severe_bleeding"},
     "Uncontrolled bleeding needs emergency care."),
    ({"stiff_neck", "high_fever"},
     "High fever with a stiff neck can indicate meningitis."),
]

URGENT_SINGLE = {
    "coma", "unconsciousness", "seizures", "fainting",
    "blood_in_sputum", "vomiting_blood",
}

def check(symptoms):
    """Return (level, message) where level is 'emergency', 'urgent' or 'normal'."""
    selected = {s.strip().lower().replace(" ", "_") for s in symptoms}

    for trigger, reason in EMERGENCY_SETS:
        if trigger.issubset(selected):
            return "emergency", reason

    hit = selected & URGENT_SINGLE
    if hit:
        return "urgent", f"'{sorted(hit)[0].replace('_', ' ')}' requires prompt medical assessment."

    return "normal", ""
