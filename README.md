# JobShield AI
An intelligent system to detect fake job postings using NLP and Machine Learning.

Tech Stack:
- Python
- Flask
- Scikit-learn
- NLP
# JobShield AI – Fake Job Posting Detection System

## Overview
JobShield AI is an NLP-based machine learning system designed to detect fraudulent job postings.

## Features
- Text preprocessing using TF-IDF (uni & bi-grams)
- Metadata feature engineering
- Random Forest & Logistic Regression comparison
- ROC-AUC evaluation (0.98)
- Confidence-based risk classification
- Flask web deployment

## Tech Stack
- Python
- Scikit-learn
- Pandas, NumPy
- Flask
- Matplotlib, Seaborn

## Dataset
Kaggle Fake Job Postings Dataset:
https://www.kaggle.com/datasets/josereimondez/fake-jobs-posting-detection

## How to Run
1. Install requirements:
   pip install -r requirements.txt

2. Train model:
   python model/train_model.py

3. Run application:
   python app.py