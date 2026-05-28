import os

class Config:

    # ==========================================
    # SECRET KEY
    # ==========================================

    SECRET_KEY = os.getenv(
        "SECRET_KEY",
        "jobshield_secret_key"
    )

    # ==========================================
    # DATABASE CONFIGURATION
    # ==========================================

    SQLALCHEMY_DATABASE_URI = os.getenv(
        "DATABASE_URL",
        "sqlite:///jobshield.db"
    )

    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # ==========================================
    # SESSION & COOKIE SECURITY
    # ==========================================

    SESSION_COOKIE_SECURE = True

    REMEMBER_COOKIE_SECURE = True

    SESSION_COOKIE_HTTPONLY = True

    REMEMBER_COOKIE_HTTPONLY = True

    SESSION_COOKIE_SAMESITE = "Lax"

    REMEMBER_COOKIE_DURATION = 86400