from flask_sqlalchemy import SQLAlchemy

from flask_login import UserMixin

from datetime import datetime

db = SQLAlchemy()

# =====================================================
# USER MODEL
# =====================================================

class User(UserMixin, db.Model):

    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)

    username = db.Column(
        db.String(100),
        nullable=False
    )

    email = db.Column(
        db.String(150),
        unique=True,
        nullable=False
    )

    password = db.Column(
        db.String(255),
        nullable=False
    )
    is_admin = db.Column(
    db.Boolean,
    default=False
    )

    # Relationship with JobAnalysis
    analyses = db.relationship(
        "JobAnalysis",
        backref="user",
        lazy=True
    )

    def __repr__(self):

        return f"<User {self.username}>"

# =====================================================
# JOB ANALYSIS MODEL
# =====================================================

class JobAnalysis(db.Model):

    __tablename__ = "job_analyses"

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id"),
        nullable=False
    )

    job_title = db.Column(
        db.String(255),
        nullable=False
    )

    prediction = db.Column(
        db.String(100),
        nullable=False
    )

    confidence = db.Column(
        db.Float,
        nullable=False
    )

    risk_level = db.Column(
        db.String(100),
        nullable=False
    )

    trust_score = db.Column(
        db.Integer,
        nullable=False
    )

    verification_status = db.Column(
        db.String(100),
        nullable=False
    )

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )

    def __repr__(self):

        return f"<JobAnalysis {self.job_title}>"