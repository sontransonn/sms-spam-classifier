import pandas as pd
import numpy as np
import os, pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model():
    test_df = pd.read_csv('data/processed/test.csv')
    X_test_clean_messages = test_df['clean_message']
    y_test = test_df['label'].values

    with open('models/sms_spam_classifier_model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
    with open("models/tfidf_vectorizer.pkl", 'rb') as f:
        loaded_vectorizer = pickle.load(f)

    X_test_vectorized = loaded_vectorizer.transform(X_test_clean_messages).toarray()

    y_pred = loaded_model.predict(X_test_vectorized)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"[*] Accuracy: {accuracy:.4f}")
    print(f"[*] Precision: {precision:.4f}")
    print(f"[*] Recall: {recall:.4f}")
    print(f"[*] F1-Score: {f1:.4f}\n")

if __name__ == "__main__":
    evaluate_model()