from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model (Random Forest pipeline)
model = joblib.load("model/jobshield_model.pkl")

# ---- Model comparison metrics (from your evaluation) ----
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


# ================= Landing =================
@app.route("/")
def landing():
    return render_template("landing.html")


# ================= Analyze =================
@app.route("/analyze", methods=["GET", "POST"])
def analyze():

    if request.method == "POST":

        title = request.form.get("title", "")
        company_profile = request.form.get("company_profile", "")
        description = request.form.get("description", "")
        requirements = request.form.get("requirements", "")
        benefits = request.form.get("benefits", "")
        salary = request.form.get("salary", "")

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

        # Prediction
        fake_prob = model.predict_proba(input_df)[0][1]
        genuine_prob = 1 - fake_prob
        confidence = round(genuine_prob * 100, 2)

        # Risk classification
        if fake_prob < 0.15:
            result = "GENUINE JOB POSTING"
            risk_level = "Low Risk"
        elif fake_prob < 0.40:
            result = "MODERATE RISK JOB POSTING"
            risk_level = "Medium Risk"
        else:
            result = "LIKELY FAKE JOB POSTING"
            risk_level = "High Risk"

        # Signals
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

        # Interpretation
        if confidence > 85:
            interpretation = "Model is highly confident in this classification."
            recommendation = "This posting appears legitimate based on structural and textual patterns."
        elif confidence >= 60:
            interpretation = "Model shows moderate confidence."
            recommendation = "Verify company website and official contact channels before proceeding."
        else:
            interpretation = "Model shows strong fraud indicators."
            recommendation = "Avoid sharing personal or financial information."

        # Metadata feature visualization
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
            rf_metrics=rf_metrics,
            lr_metrics=lr_metrics
        )

    return render_template("analyze.html")


if __name__ == "__main__":
    app.run(debug=True)