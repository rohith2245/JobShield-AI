# JobShield AI – Fake Job Posting Detection System

An intelligent NLP-based Machine Learning system that detects fraudulent job postings and provides confidence-based risk warnings.

---

## 🔍 Overview

JobShield AI analyzes job posting content using Natural Language Processing (NLP) and classification algorithms to determine whether a job listing is:

- ✅ Genuine  
- ⚠️ Moderately Safe  
- ❌ Likely Fake  

The system provides probability-based confidence scores to improve decision reliability and user safety.

---

## 🚀 Key Features

- Text preprocessing (cleaning, stopword removal, lemmatization)
- TF-IDF vectorization (uni-grams & bi-grams)
- Metadata feature engineering (description length, salary presence)
- Model comparison:
  - Random Forest
  - Logistic Regression
- Performance evaluation using:
  - Confusion Matrix
  - Precision, Recall, F1-Score
  - ROC-AUC (0.98)
- Confidence-based risk classification
- Flask-based web deployment

---

## 🛠 Tech Stack

**Programming:** Python  
**Machine Learning:** Scikit-learn  
**NLP:** TF-IDF, Text Preprocessing  
**Data Handling:** Pandas, NumPy  
**Visualization:** Matplotlib, Seaborn  
**Deployment:** Flask  

---

## 📊 Model Performance

- Random Forest Accuracy: ~97%  
- ROC-AUC Score: 0.98  
- Strong legitimate job detection  
- Balanced fraud detection capability  

---

## 📂 Dataset

Kaggle Fake Job Postings Dataset:  
https://www.kaggle.com/datasets/josereimondez/fake-jobs-posting-detection

---

## ▶️ How to Run Locally

### 1️⃣ Install Dependencies
## 📈 System Workflow  

User Input  
→ Text Preprocessing  
→ TF-IDF Feature Extraction  
→ Metadata Feature Integration  
→ Random Forest Prediction  
→ Confidence Thresholding  
→ Risk Classification Output  

---

## 🔮 Future Enhancements  

- Integration of Transformer models (BERT)  
- Handling class imbalance using SMOTE  
- Real-time job portal scraping  
- Company verification integration  
- Deployment to cloud platform  

---

## 👨‍💻 Author  

Rohith Veeramalla  
B.Tech – Artificial Intelligence & Machine Learning  
CVR College of Engineering