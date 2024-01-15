import pandas as pd
from catboost import Pool
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.preprocessing import OneHotEncoder


def load_data(data_path):
    try:
        small_df = pd.read_csv(data_path)
        small_df.columns = small_df.columns.str.strip().str.lower().str.replace(' ', '_')
        data = small_df[['product_title', 'merchant_id', 'cluster_label', 'category_id']]
        data.columns = data.columns.str.strip()

        missing_values = sum(data.isnull().sum())
        if missing_values > 0:
            raise ValueError('Есть пропущенные значения')
        else:
            return data
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        return None
    

def load_model(path_model):
    try:
        with open(path_model, 'rb') as file:
            loaded_model = pickle.load(file)
        return loaded_model
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
        return None


def catboost_prediction(data, model):
    try:
        text_features = ['product_title', 'cluster_label']
        new_data_pool = Pool(data=data, text_features=text_features)
        predictions = model.predict(new_data_pool)
        # predictions.to_csv(path_itog)
        return predictions
    except Exception as e:
        print(f"Ошибка при предсказании с помощью CatBoost: {e}")
        return None


def logreg_prediction(data, model):
    try:
        nltk.download('punkt')
        with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
            loaded_tfidf_vectorizer = pickle.load(vectorizer_file)

        def preprocess_text(text):
            text = text.lower()
            text = re.sub(r'\b\d+\b', '', text)
            return text

        data_processed = data.copy()

        data_processed['product_title'] = data_processed['product_title'].apply(preprocess_text)
        data_processed['cluster_label'] = data_processed['cluster_label'].apply(preprocess_text)

        text_features = ['product_title', 'cluster_label']
        data_processed['text_features'] = data_processed[text_features].agg(' '.join, axis=1)
        data_product = loaded_tfidf_vectorizer.transform(data_processed['text_features']).toarray()

        predictions = model.predict(data_product)
        return predictions
    except Exception as e:
        print(f"Ошибка при предсказании с помощью Logistic Regression: {e}")
        return None

def svm_prediction(data, model, loaded_tfidf_vectorizer):
    try:
        nltk.download('punkt')

        def preprocess_text(text):
            text = text.lower()
            text = re.sub(r'\b\d+\b', '', text)
            return text

        data_processed = data.copy()

        data_processed['product_title'] = data_processed['product_title'].apply(preprocess_text)
        data_processed['cluster_label'] = data_processed['cluster_label'].apply(preprocess_text)

        text_features = ['product_title', 'cluster_label']
        data_processed['text_features'] = data_processed[text_features].agg(' '.join, axis=1)
        data_product = loaded_tfidf_vectorizer.transform(data_processed['text_features']).toarray()

        predictions = model.predict(data_product)
        return predictions
    except Exception as e:
        print(f"Ошибка при предсказании с помощью SVM: {e}")
        return None