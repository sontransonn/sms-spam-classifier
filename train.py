import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from src.logistic_regression import LogisticRegression

PROCESSED_DATA_DIR = 'data/processed/' 
TRAIN_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'train.csv')
TEST_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'test.csv') 

MODEL_ARTIFACTS_DIR = 'models/'
CLASSIFIER_MODEL_PATH = os.path.join(MODEL_ARTIFACTS_DIR, 'sms_spam_classifier_model_v1.pkl')
TFIDF_VECTORIZER_PATH = os.path.join(MODEL_ARTIFACTS_DIR, 'tfidf_vectorizer.pkl')

LEARNING_RATE = 0.5
NUM_ITERATIONS = 15000
MAX_FEATURES = 3000

def train_model():
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)
        
    X_train_clean_messages = train_df['clean_message']
    y_train = train_df['label'].values
    X_test_clean_messages = test_df['clean_message']

    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
    X_train_vectorized = vectorizer.fit_transform(X_train_clean_messages).toarray()
    _ = vectorizer.transform(X_test_clean_messages).toarray() 
    
    model = LogisticRegression(lr=LEARNING_RATE, num_iter=NUM_ITERATIONS)
    model.fit(X_train_vectorized, y_train)

    os.makedirs(MODEL_ARTIFACTS_DIR, exist_ok=True) 
    with open(CLASSIFIER_MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(TFIDF_VECTORIZER_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)

if __name__ == "__main__":
    train_model()