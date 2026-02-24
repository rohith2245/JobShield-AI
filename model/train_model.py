import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
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

from imblearn.over_sampling import SMOTE

from preprocess import build_text_features, add_meta_features


# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv("data/raw/fake_job_postings.csv")

df = build_text_features(df)
df = add_meta_features(df)

X = df[['combined_text', 'desc_length', 'company_profile_length', 'has_salary']]
y = df['fraudulent']

# ===============================
# TRAIN TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===============================
# PREPROCESSING
# ===============================
text_vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    min_df=3
)

preprocessor = ColumnTransformer([
    ('text', text_vectorizer, 'combined_text'),
    ('num', StandardScaler(), ['desc_length', 'company_profile_length', 'has_salary'])
])

# Fit transform training data
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# ===============================
# SMOTE FOR IMBALANCE
# ===============================
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_transformed, y_train)

print("After SMOTE class distribution:")
print(pd.Series(y_resampled).value_counts())

# ===============================
# RANDOM FOREST MODEL
# ===============================
rf_model = RandomForestClassifier(
    n_estimators=400,
    max_depth=35,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_resampled, y_resampled)

rf_pred = rf_model.predict(X_test_transformed)

print("\n=== RANDOM FOREST RESULTS (IMPROVED) ===")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

# ===============================
# CONFUSION MATRIX
# ===============================
cm = confusion_matrix(y_test, rf_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Random Forest - Confusion Matrix (Improved)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("static/confusion_matrix.png")
plt.close()

# ===============================
# ROC CURVE
# ===============================
rf_probs = rf_model.predict_proba(X_test_transformed)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, rf_probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'Random Forest (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], '--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig("static/roc_curve.png")
plt.close()

print("ROC AUC Score:", roc_auc)

# ===============================
# FEATURE IMPORTANCE (FIXED)
# ===============================
# Get feature names from fitted preprocessor
feature_names = preprocessor.named_transformers_['text'].get_feature_names_out()

importances = rf_model.feature_importances_[:len(feature_names)]

top_indices = np.argsort(importances)[-20:]

plt.figure(figsize=(8, 6))
plt.barh(range(len(top_indices)), importances[top_indices])
plt.yticks(range(len(top_indices)), feature_names[top_indices])
plt.title("Top 20 Important Text Features")
plt.tight_layout()
plt.savefig("static/feature_importance.png")
plt.close()

# ===============================
# SAVE PIPELINE
# ===============================
pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('model', rf_model)
])

joblib.dump(pipeline, "model/jobshield_model.pkl")

print("\nModel saved successfully (Improved Version)")