import re
import string
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if pd.isna(text):
        return ""

    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

def build_text_features(df):
    df['combined_text'] = (
        df['title'].fillna('') + ' ' +
        df['company_profile'].fillna('') + ' ' +
        df['description'].fillna('') + ' ' +
        df['requirements'].fillna('') + ' ' +
        df['benefits'].fillna('')
    )

    df['combined_text'] = df['combined_text'].apply(clean_text)
    return df

def add_meta_features(df):
    df['desc_length'] = df['description'].fillna('').apply(len)
    df['company_profile_length'] = df['company_profile'].fillna('').apply(len)
    df['has_salary'] = df['salary_range'].notna().astype(int)
    return df
