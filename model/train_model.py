import matplotlib
matplotlib.use('Agg')

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from preprocess import build_text_features, add_meta_features

# Load dataset
df = pd.read_csv("data/raw/fake_job_postings.csv")

df = build_text_features(df)
df = add_meta_features(df)

X = df[['combined_text', 'desc_length', 'company_profile_length', 'has_salary']]
y = df['fraudulent']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Text vectorizer
text_vectorizer = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 2),
    min_df=3
)

preprocessor = ColumnTransformer([
    ('text', text_vectorizer, 'combined_text'),
    ('num', StandardScaler(), ['desc_length', 'company_profile_length', 'has_salary'])
])

# ----------------------------
# RANDOM FOREST MODEL
# ----------------------------
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=30,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf_pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('model', rf_model)
])

rf_pipeline.fit(X_train, y_train)
rf_pred = rf_pipeline.predict(X_test)

print("=== RANDOM FOREST RESULTS ===")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, rf_pred)
plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Random Forest - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("static/confusion_matrix.png")
plt.close()

# ----------------------------
# LOGISTIC REGRESSION MODEL
# ----------------------------
lr_model = LogisticRegression(max_iter=1000)

lr_pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('model', lr_model)
])

lr_pipeline.fit(X_train, y_train)
lr_pred = lr_pipeline.predict(X_test)

print("\n=== LOGISTIC REGRESSION RESULTS ===")
print("Accuracy:", accuracy_score(y_test, lr_pred))
print(classification_report(y_test, lr_pred))

# ----------------------------
# ROC CURVE
# ----------------------------
rf_probs = rf_pipeline.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, rf_probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'Random Forest (AUC = {roc_auc:.2f})')
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig("static/roc_curve.png")
plt.close()

print("ROC AUC Score:", roc_auc)

# ----------------------------
# FEATURE IMPORTANCE
# ----------------------------
feature_names = (
    rf_pipeline.named_steps['preprocess']
    .transformers_[0][1]
    .get_feature_names_out()
)

importances = rf_model.feature_importances_[:len(feature_names)]

top_indices = np.argsort(importances)[-20:]

plt.figure(figsize=(8,6))
plt.barh(range(len(top_indices)), importances[top_indices])
plt.yticks(range(len(top_indices)), feature_names[top_indices])
plt.title("Top 20 Important Text Features")
plt.tight_layout()
plt.savefig("static/feature_importance.png")
plt.close()

# Save best model (Random Forest)
joblib.dump(rf_pipeline, "model/jobshield_model.pkl")

print("Model saved successfully")
