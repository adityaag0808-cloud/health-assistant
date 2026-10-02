from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class SessionLog(db.Model):
    __tablename__ = "session_logs"
    id          = db.Column(db.Integer, primary_key=True)
    symptoms    = db.Column(db.Text, nullable=False)
    top_disease = db.Column(db.String(120), nullable=True)
    probability = db.Column(db.Float, nullable=True)
    redflag     = db.Column(db.String(20), default="normal")
    created_at  = db.Column(db.DateTime, default=datetime.utcnow)
