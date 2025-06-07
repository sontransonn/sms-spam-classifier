import pandas as pd
import os, pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from src.logistic_regression import LogisticRegression

def train_model():
    train_df = pd.read_csv('data/processed/train.csv')
    test_df = pd.read_csv('data/processed/test.csv')
        
    X_train_clean_messages = train_df['clean_message']
    y_train = train_df['label'].values
    X_test_clean_messages = test_df['clean_message']

    vectorizer = TfidfVectorizer(max_features=3000)
    X_train_vectorized = vectorizer.fit_transform(X_train_clean_messages).toarray()
    _ = vectorizer.transform(X_test_clean_messages).toarray() 
    
    model = LogisticRegression(lr=0.5, num_iter=15000)
    model.fit(X_train_vectorized, y_train)

    os.makedirs('models/', exist_ok=True) 
    with open('models/sms_spam_classifier_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('models/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

if __name__ == "__main__":
    train_model()