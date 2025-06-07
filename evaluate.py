import pandas as pd
import numpy as np
import os, pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

PROCESSED_DATA_DIR = 'data/processed/'
TEST_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'test.csv')

MODEL_ARTIFACTS_DIR = 'models/'
CLASSIFIER_MODEL_PATH = os.path.join(MODEL_ARTIFACTS_DIR, 'sms_spam_classifier_model_v1.pkl')
TFIDF_VECTORIZER_PATH = os.path.join(MODEL_ARTIFACTS_DIR, 'tfidf_vectorizer.pkl')

def evaluate_model():
    test_df = pd.read_csv(TEST_DATA_PATH)
    X_test_clean_messages = test_df['clean_message']
    y_test = test_df['label'].values

    with open(CLASSIFIER_MODEL_PATH, 'rb') as f:
        loaded_model = pickle.load(f)
    with open(TFIDF_VECTORIZER_PATH, 'rb') as f:
        loaded_vectorizer = pickle.load(f)

    X_test_vectorized = loaded_vectorizer.transform(X_test_clean_messages).toarray()

    y_pred = loaded_model.predict(X_test_vectorized)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"[*] Accuracy (Độ chính xác tổng thể): {accuracy:.4f}")
    print(f"[*] Precision (Tỷ lệ dự đoán đúng SPAM): {precision:.4f}")
    print(f"[*] Recall (Tỷ lệ SPAM thực tế được nhận diện): {recall:.4f}")
    print(f"[*] F1-Score (Trung bình điều hòa của Precision và Recall): {f1:.4f}\n")

if __name__ == "__main__":
    evaluate_model()