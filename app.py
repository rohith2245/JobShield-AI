from flask import Flask, render_template, request, redirect, url_for, flash
from flask import send_file
from flask_login import (
    LoginManager,
    login_user,
    logout_user,
    login_required,
    current_user
)

import joblib
import pandas as pd

from config import Config

from auth.models import db, User, JobAnalysis

from auth.forms import validate_user_email

from utils.security import (
    hash_password,
    verify_password
)
from verification.company_verifier import verify_company
from utils.pdf_generator import generate_pdf_report

# =====================================================
# APP CONFIGURATION
# =====================================================

app = Flask(__name__)

app.config.from_object(Config)

# =====================================================
# DATABASE INITIALIZATION
# =====================================================

db.init_app(app)

with app.app_context():
    db.create_all()

# =====================================================
# LOGIN MANAGER
# =====================================================

login_manager = LoginManager()

login_manager.init_app(app)

login_manager.login_view = "login"

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# =====================================================
# LOAD ML MODEL
# =====================================================

model = joblib.load("model/jobshield_model.pkl")

# =====================================================
# MODEL METRICS
# =====================================================

rf_metrics = {
    "accuracy": 0.9687,
    "precision": 0.68,
    "recall": 0.66,
    "f1": 0.67,
    "roc_auc": 0.9819
}

lr_metrics = {
    "accuracy": 0.9726,
    "precision": 0.99,
    "recall": 0.44,
    "f1": 0.61,
    "roc_auc": 0.9819
}

# =====================================================
# LANDING PAGE
# =====================================================

@app.route("/")
def landing():
    return render_template("landing.html")


# =====================================================
# REGISTER
# =====================================================

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        username = request.form.get("username")

        email = request.form.get("email")

        password = request.form.get("password")

        # ==========================================
        # VALIDATE EMAIL
        # ==========================================

        valid_email = validate_user_email(email)

        if not valid_email:

            flash(
                "Please enter a valid email address.",
                "danger"
            )

            return redirect(
                url_for("register")
            )

        # ==========================================
        # CHECK EXISTING USER
        # ==========================================

        existing_user = User.query.filter_by(
            email=email
        ).first()

        if existing_user:

            flash(
                "An account with this email already exists.",
                "warning"
            )

            return redirect(
                url_for("register")
            )

        # ==========================================
        # HASH PASSWORD
        # ==========================================

        hashed_password = hash_password(password)

        # ==========================================
        # CREATE USER
        # ==========================================

        new_user = User(

            username=username,

            email=email,

            password=hashed_password
        )

        db.session.add(new_user)

        db.session.commit()

        # ==========================================
        # SUCCESS MESSAGE
        # ==========================================

        flash(
            "Registration successful. Please login.",
            "success"
        )

        return redirect(
            url_for("login")
        )

    return render_template("register.html")

# =====================================================
# LOGIN
# =====================================================

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email = request.form.get("email")

        password = request.form.get("password")

        # ==========================================
        # FIND USER
        # ==========================================

        user = User.query.filter_by(
            email=email
        ).first()

        # ==========================================
        # USER NOT FOUND
        # ==========================================

        if not user:

            flash(
                "User does not exist.",
                "danger"
            )

            return redirect(
                url_for("login")
            )

        # ==========================================
        # WRONG PASSWORD
        # ==========================================

        if not verify_password(
            user.password,
            password
        ):

            flash(
                "Incorrect password.",
                "danger"
            )

            return redirect(
                url_for("login")
            )

        # ==========================================
        # LOGIN SUCCESS
        # ==========================================

        login_user(user)

        flash(
            "Login successful.",
            "success"
        )

        return redirect(
            url_for("analyze")
        )

    return render_template("login.html")

# =====================================================
# LOGOUT
# =====================================================

@app.route("/logout")
@login_required
def logout():

    logout_user()

    flash("Logged out successfully.", "info")

    return redirect(url_for("landing"))

# =====================================================
# ANALYZE
# =====================================================

@app.route("/analyze", methods=["GET", "POST"])
@login_required
def analyze():

    if request.method == "POST":

        title = request.form.get("title", "")

        company_profile = request.form.get("company_profile", "")

        description = request.form.get("description", "")

        requirements = request.form.get("requirements", "")

        benefits = request.form.get("benefits", "")

        salary = request.form.get("salary", "")
        recruiter_email = request.form.get("recruiter_email", "")
        company_website = request.form.get("company_website", "")
        linkedin_url = request.form.get("linkedin_url", "")

        combined_text = " ".join([
            title,
            company_profile,
            description,
            requirements,
            benefits
        ])

        desc_length = len(description)

        company_profile_length = len(company_profile)

        has_salary = 1 if salary.strip() != "" else 0

        input_df = pd.DataFrame([{
            "combined_text": combined_text,
            "desc_length": desc_length,
            "company_profile_length": company_profile_length,
            "has_salary": has_salary
        }])

        # ==========================================
        # PREDICTION
        # ==========================================

        fake_prob = model.predict_proba(input_df)[0][1]

        genuine_prob = 1 - fake_prob

        confidence = round(genuine_prob * 100, 2)

        # ==========================================
        # RISK CLASSIFICATION
        # ==========================================

        if fake_prob < 0.15:

            result = "GENUINE JOB POSTING"

            risk_level = "Low Risk"

        elif fake_prob < 0.40:

            result = "MODERATE RISK JOB POSTING"

            risk_level = "Medium Risk"

        else:

            result = "LIKELY FAKE JOB POSTING"

            risk_level = "High Risk"

        # ==========================================
        # SIGNALS
        # ==========================================

        signals = []

        if desc_length < 200:
            signals.append("Short job description detected")
        else:
            signals.append("Detailed job description")

        if company_profile_length < 100:
            signals.append("Limited company information")
        else:
            signals.append("Structured company profile")

        if has_salary:
            signals.append("Salary information provided")
        else:
            signals.append("Salary details missing")

        # ==========================================
        # COMPANY VERIFICATION
        # ==========================================

        verification_data = verify_company(
        recruiter_email,
        company_website,
        linkedin_url,
        has_salary,
        desc_length
        )

        trust_score = verification_data["trust_score"]
        verification_status = verification_data["verification_status"]
        verification_signals = verification_data["verification_signals"]

        # ==========================================
        # INTERPRETATION
        # ==========================================

        if confidence > 85:

            interpretation = "Model is highly confident in this classification."

            recommendation = (
                "This posting appears legitimate based on structural and textual patterns."
            )

        elif confidence >= 60:

            interpretation = "Model shows moderate confidence."

            recommendation = (
                "Verify company website and official contact channels before proceeding."
            )

        else:

            interpretation = "Model shows strong fraud indicators."

            recommendation = (
                "Avoid sharing personal or financial information."
            )

        # ==========================================
        # FEATURE VISUALIZATION
        # ==========================================

        feature_names = [
            "Description Length",
            "Company Profile Length",
            "Salary Presence"
        ]

        feature_values = [
            desc_length,
            company_profile_length,
            has_salary * 100
        ]
        # ==========================================
        # SAVE ANALYSIS HISTORY
        # ==========================================

        analysis = JobAnalysis(
            user_id=current_user.id,
            job_title=title,
            prediction=result,
            confidence=confidence,
            risk_level=risk_level,
            trust_score=trust_score,
            verification_status=verification_status
        )
        db.session.add(analysis)
        db.session.commit()
        return render_template(
            "result.html",
            result=result,
            confidence=confidence,
            risk_level=risk_level,
            interpretation=interpretation,
            recommendation=recommendation,
            signals=signals,
            feature_names=feature_names,
            feature_values=feature_values,
            trust_score=trust_score,
            verification_status=verification_status,
            verification_signals=verification_signals,
            company_website=company_website,
            linkedin_url=linkedin_url,
            rf_metrics=rf_metrics,
            lr_metrics=lr_metrics
        )

    return render_template("analyze.html")
        # =====================================================
# PDF REPORT DOWNLOAD
# =====================================================

@app.route("/download-report")
@login_required
def download_report():

    filepath = "jobshield_report.pdf"

    # ==========================================
    # SAMPLE DATA
    # ==========================================

    result = request.args.get("result")

    confidence = request.args.get("confidence")

    risk_level = request.args.get("risk_level")

    trust_score = request.args.get("trust_score")

    verification_status = request.args.get(
        "verification_status"
    )

    recommendation = request.args.get(
        "recommendation"
    )

    signals = request.args.getlist("signals")

    verification_signals = request.args.getlist(
        "verification_signals"
    )

    # ==========================================
    # GENERATE PDF
    # ==========================================

    generate_pdf_report(

        filepath,

        result,

        confidence,

        risk_level,

        trust_score,

        verification_status,

        recommendation,

        signals,

        verification_signals
    )

    return send_file(
        filepath,
        as_attachment=True
    )


    # =====================================================
# PROFILE PAGE
# =====================================================

@app.route("/profile")
@login_required
def profile():

    # ==========================================
    # FETCH USER ANALYSES
    # ==========================================

    analyses = JobAnalysis.query.filter_by(
        user_id=current_user.id
    ).all()

    # ==========================================
    # TOTAL ANALYSES
    # ==========================================

    total_analyses = len(analyses)

    # ==========================================
    # GENUINE JOB COUNT
    # ==========================================

    genuine_count = len([
        a for a in analyses
        if "GENUINE" in a.prediction
    ])

    # ==========================================
    # FAKE JOB COUNT
    # ==========================================

    fake_count = len([
        a for a in analyses
        if "FAKE" in a.prediction
    ])

    # ==========================================
    # AVERAGE TRUST SCORE
    # ==========================================

    if total_analyses > 0:

        average_trust_score = round(

            sum(a.trust_score for a in analyses)

            / total_analyses,

            2
        )

    else:

        average_trust_score = 0

    # ==========================================
    # RECENT ANALYSES
    # ==========================================

    recent_analyses = JobAnalysis.query.filter_by(
        user_id=current_user.id
    ).order_by(
        JobAnalysis.created_at.desc()
    ).limit(5).all()

    # ==========================================
    # RENDER PROFILE
    # ==========================================

    return render_template(

        "profile.html",

        user=current_user,

        total_analyses=total_analyses,

        genuine_count=genuine_count,

        fake_count=fake_count,

        average_trust_score=average_trust_score,

        recent_analyses=recent_analyses
    )
# =====================================================
# HISTORY PAGE
# =====================================================

@app.route("/history")
@login_required
def history():

    analyses = JobAnalysis.query.filter_by(
        user_id=current_user.id
    ).order_by(
        JobAnalysis.created_at.desc()
    ).all()

    return render_template(
        "history.html",
        analyses=analyses
    )
# =====================================================
# ADMIN DASHBOARD
# =====================================================

@app.route("/admin")
@login_required
def admin_dashboard():

    # ==========================================
    # ADMIN CHECK
    # ==========================================

    if not current_user.is_admin:

        flash(
            "Access denied. Admins only.",
            "danger"
        )

        return redirect(
            url_for("landing")
        )

    # ==========================================
    # TOTAL USERS
    # ==========================================

    total_users = User.query.count()

    # ==========================================
    # TOTAL ANALYSES
    # ==========================================

    total_analyses = JobAnalysis.query.count()

    # ==========================================
    # FAKE JOB COUNT
    # ==========================================

    fake_jobs = JobAnalysis.query.filter(
        JobAnalysis.prediction.like("%FAKE%")
    ).count()

    # ==========================================
    # HIGH RISK JOB COUNT
    # ==========================================

    high_risk_jobs = JobAnalysis.query.filter_by(
        verification_status="HIGH RISK"
    ).count()

    # ==========================================
    # RECENT ANALYSES
    # ==========================================

    recent_analyses = JobAnalysis.query.order_by(
        JobAnalysis.created_at.desc()
    ).limit(10).all()

    # ==========================================
    # RENDER DASHBOARD
    # ==========================================

    return render_template(

        "admin_dashboard.html",

        total_users=total_users,

        total_analyses=total_analyses,

        fake_jobs=fake_jobs,

        high_risk_jobs=high_risk_jobs,

        recent_analyses=recent_analyses
    )

# =====================================================
# RUN APP
# =====================================================

if __name__ == "__main__":

    app.run(host="0.0.0.0", port=10000, debug=True)