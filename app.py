from flask import Flask, render_template, request
import joblib
import pandas as pd
from model.preprocess import build_text_features, add_meta_features

app = Flask(__name__)

# Load trained ML pipeline
model = joblib.load("model/jobshield_model.pkl")


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":

        # Collect form data
        data = {
            "title": request.form["title"],
            "company_profile": request.form["company_profile"],
            "description": request.form["description"],
            "requirements": request.form["requirements"],
            "benefits": request.form["benefits"],
            "salary_range": request.form["salary"]
        }

        # Convert to DataFrame
        df = pd.DataFrame([data])

        # Apply same preprocessing used during training
        df = build_text_features(df)
        df = add_meta_features(df)

        # Select required features
        X = df[['combined_text', 'desc_length', 'company_profile_length', 'has_salary']]

        # Predict
        prediction = model.predict(X)[0]
        confidence = max(model.predict_proba(X)[0]) * 100

        # Confidence-based decision logic
        if prediction == 0 and confidence >= 85:
            result = "✅ Genuine Job Posting"
            warning = "This job appears legitimate based on strong indicators."
            color = "#27ae60"
        elif prediction == 0 and confidence >= 60:
            result = "⚠️ Moderately Safe Job Posting"
            warning = "This job looks legitimate, but users are advised to verify company details before proceeding."
            color = "#f39c12"
        else:
            result = "❌ Likely Fake Job Posting"
            warning = "This job shows multiple risk indicators. Avoid sharing personal or financial information."
            color = "#e74c3c"

        return render_template(
            "result.html",
            result=result,
            confidence=round(confidence, 2),
            warning=warning,
            color=color
        )

    return render_template("index.html")


@app.route("/performance")
def performance():
    return render_template("performance.html")


if __name__ == "__main__":
    app.run(debug=True)
