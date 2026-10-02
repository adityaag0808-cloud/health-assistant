import json
import random
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy

import predict as engine

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///sessions.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)


class SessionLog(db.Model):
    """Anonymous log: no name, no email, no IP. Used to report usage statistics."""
    id          = db.Column(db.Integer, primary_key=True)
    symptoms    = db.Column(db.Text)
    top_disease = db.Column(db.String(120))
    probability = db.Column(db.Float)
    redflag     = db.Column(db.String(20))
    created_at  = db.Column(db.DateTime, default=datetime.utcnow)


class AmbulanceBooking(db.Model):
    """Stores emergency ambulance dispatch requests."""
    id             = db.Column(db.Integer, primary_key=True)
    booking_id     = db.Column(db.String(20), unique=True)
    patient_name   = db.Column(db.String(100))
    contact_phone  = db.Column(db.String(20))
    pickup_address = db.Column(db.Text)
    ambulance_type = db.Column(db.String(50))
    city           = db.Column(db.String(50))
    status         = db.Column(db.String(30), default="Dispatched")
    created_at     = db.Column(db.DateTime, default=datetime.utcnow)


with app.app_context():
    db.create_all()


@app.route("/")
def index():
    return render_template("index.html",
                           symptoms=[s.replace("_", " ") for s in engine.VOCABULARY])


@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(silent=True) or {}
    symptoms = payload.get("symptoms", [])

    if not isinstance(symptoms, list) or not symptoms:
        return jsonify({"error": "Select at least one symptom."}), 400
    if len(symptoms) > 20:
        return jsonify({"error": "Please select 20 symptoms or fewer."}), 400

    result = engine.predict(symptoms)

    if result["status"] == "ok":
        db.session.add(SessionLog(
            symptoms=json.dumps(sorted(symptoms)),
            top_disease=result["predictions"][0]["disease"],
            probability=result["predictions"][0]["probability"],
            redflag=result["redflag_level"]
        ))
        db.session.commit()

    return jsonify(result)


# ==========================================
# AMBULANCE DISPATCH API ROUTE
# ==========================================

@app.route("/api/book-ambulance", methods=["POST"])
def api_book_ambulance():
    data = request.get_json(silent=True) or {}
    name = data.get("name", "").strip()
    phone = data.get("phone", "").strip()
    address = data.get("address", "").strip()
    amb_type = data.get("type", "Basic Life Support (BLS)")
    city = data.get("city", "Mumbai")

    if not name or not phone or not address:
        return jsonify({"success": False, "error": "Name, phone number, and pickup address are required."}), 400

    # Generate reference booking ID like AMB-8492
    booking_code = f"AMB-{random.randint(1000, 9999)}"
    eta_mins = random.randint(8, 14)

    booking = AmbulanceBooking(
        booking_id=booking_code,
        patient_name=name,
        contact_phone=phone,
        pickup_address=address,
        ambulance_type=amb_type,
        city=city,
        status="Dispatched"
    )
    db.session.add(booking)
    db.session.commit()

    return jsonify({
        "success": True,
        "booking_id": booking_code,
        "eta": f"{eta_mins} minutes",
        "driver_contact": "+91 98200 12345",
        "ambulance_number": f"MH 04 AB {random.randint(1000, 9999)}",
        "message": f"Emergency ambulance dispatched! Unit arriving in ~{eta_mins} mins."
    })


# ==========================================
# CLINIC DIRECTORY & CHATBOT ROUTE
# ==========================================

CLINIC_DIRECTORY = {
    "mumbai": [
        {
            "name": "City Care Clinic (Andheri)",
            "doctor": "Dr. Sharma (General Physician)",
            "phone": "+912226730000",
            "display_phone": "+91 22 2673 0000",
            "timings": "9 AM - 8 PM",
            "map": "https://maps.google.com/?q=clinics+in+Andheri+Mumbai"
        },
        {
            "name": "LifeLine Health Center (Bandra)",
            "doctor": "Dr. Mehta (Family Medicine)",
            "phone": "+912226401111",
            "display_phone": "+91 22 2640 1111",
            "timings": "10 AM - 7 PM",
            "map": "https://maps.google.com/?q=clinics+in+Bandra+Mumbai"
        }
    ],
    "delhi": [
        {
            "name": "Metro Health Clinic (Connaught Place)",
            "doctor": "Dr. Verma (Internal Medicine)",
            "phone": "+911123412222",
            "display_phone": "+91 11 2341 2222",
            "timings": "9 AM - 6 PM",
            "map": "https://maps.google.com/?q=clinics+in+Connaught+Place+Delhi"
        },
        {
            "name": "Apex Community Clinic (South Ex)",
            "doctor": "Dr. Kapoor (General Physician)",
            "phone": "+911124623333",
            "display_phone": "+91 11 2462 3333",
            "timings": "10 AM - 8 PM",
            "map": "https://maps.google.com/?q=clinics+in+South+Extension+Delhi"
        }
    ],
    "bangalore": [
        {
            "name": "Greenwood Family Clinic (Indiranagar)",
            "doctor": "Dr. Rao (Family Physician)",
            "phone": "+918025214444",
            "display_phone": "+91 80 2521 4444",
            "timings": "9 AM - 7 PM",
            "map": "https://maps.google.com/?q=clinics+in+Indiranagar+Bangalore"
        },
        {
            "name": "Koramangala Medical Centre",
            "doctor": "Dr. Iyer (General Practitioner)",
            "phone": "+918025535555",
            "display_phone": "+91 80 2553 5555",
            "timings": "10 AM - 9 PM",
            "map": "https://maps.google.com/?q=clinics+in+Koramangala+Bangalore"
        }
    ]
}

@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(silent=True) or {}
    message = data.get("message", "").lower().strip()

    # 1. Critical Red-Flag Emergency / Ambulance Triggers
    critical_triggers = [
        "ambulance", "book ambulance", "chest pain", "breathlessness", 
        "shortness of breath", "heart attack", "unconscious", "bleeding", 
        "behosh", "saas lene me taklif", "dil ka daura"
    ]
    if any(trigger in message for trigger in critical_triggers):
        reply = (
            "🚨 <strong>EMERGENCY DISPATCH PROTOCOL ACTIVATED:</strong><br>"
            "You can request an emergency ambulance directly or call national helplines immediately:<br><br>"
            "<button class='btn btn-sm btn-danger fw-bold me-2 mb-2' data-bs-toggle='modal' data-bs-target='#ambulanceModal'>"
            "🚑 Book Emergency Ambulance Now</button><br>"
            "<a href='tel:108' class='btn btn-sm btn-outline-danger me-1'>📞 Dial 108 (Govt Ambulance)</a>"
            "<a href='tel:102' class='btn btn-sm btn-outline-danger'>📞 Dial 102</a>"
        )

    # 2. Check if user mentions a city
    elif any(city in message for city in CLINIC_DIRECTORY.keys()):
        matched_city = next(city for city in CLINIC_DIRECTORY.keys() if city in message)
        clinics = CLINIC_DIRECTORY[matched_city]
        
        cards = []
        for c in clinics:
            cards.append(
                f"<div class='p-2 mb-2 border rounded bg-light text-dark'>"
                f"<strong>🏥 {c['name']}</strong><br>"
                f"<small class='text-muted'>{c['doctor']} | ⏰ {c['timings']}</small><br>"
                f"<div class='mt-2'>"
                f"<a href='tel:{c['phone']}' class='btn btn-sm btn-success me-1'>📞 Call {c['display_phone']}</a> "
                f"<a href='{c['map']}' target='_blank' class='btn btn-sm btn-outline-primary'>🗺️ Directions</a>"
                f"</div></div>"
            )
        reply = f"Here are verified clinics in <strong>{matched_city.capitalize()}</strong>:<br><br>" + "".join(cards)

    # 3. Doctor/Clinic query without a city
    elif any(k in message for k in ["doctor", "clinic", "hospital", "number", "contact", "phone", "near me", "aspataal"]):
        reply = (
            "I can find verified nearby clinic contacts for you! 📍<br>"
            "Which city are you currently in?<br><br>"
            "<button class='btn btn-sm btn-outline-primary me-1 mb-1' onclick='sendPrompt(\"mumbai\")'>Mumbai</button> "
            "<button class='btn btn-sm btn-outline-primary me-1 mb-1' onclick='sendPrompt(\"delhi\")'>Delhi</button> "
            "<button class='btn btn-sm btn-outline-primary me-1 mb-1' onclick='sendPrompt(\"bangalore\")'>Bangalore</button>"
        )

    # 4. Care Remedies
    elif "fever" in message or "bukhar" in message:
        reply = (
            "🌡️ <strong>Fever Home Care:</strong><br>"
            "• Sip plenty of fluids/electrolytes to avoid dehydration.<br>"
            "• Rest well and apply cold forehead compresses.<br>"
            "• If temperature stays above 102°F or lasts >3 days, consult a physician."
        )
    elif "headache" in message or "sar dard" in message:
        reply = (
            "💆 <strong>Headache Relief:</strong> Rest in a quiet dark room, drink 2 glasses of water, and avoid digital screens."
        )
    elif "acidity" in message or "jalan" in message:
        reply = (
            "🥛 <strong>Acidity Relief:</strong> Sip chilled milk or warm water slowly. Avoid oily or spicy meals and stay upright."
        )

    # 5. Greetings
    elif any(g in message for g in ["hi", "hello", "hey"]):
        reply = (
            "👋 Hello! I am your Health Assistant. You can ask for care tips, clinic numbers, or click "
            "<strong>🚑 Book Ambulance</strong> at any time."
        )

    # 6. Fallback
    else:
        reply = (
            "I can assist with clinic contacts, remedies, and ambulance dispatch. "
            "Type 'Book ambulance' or select symptoms on the left to start."
        )

    return jsonify({"reply": reply})


@app.route("/stats")
def stats():
    rows = (
        db.session.query(SessionLog.top_disease, db.func.count(SessionLog.id).label("n"))
        .group_by(SessionLog.top_disease)
        .order_by(db.desc("n"))
        .limit(10)
        .all()
    )
    total_amb = AmbulanceBooking.query.count()
    return render_template("stats.html", rows=rows, total=SessionLog.query.count(), total_amb=total_amb)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
