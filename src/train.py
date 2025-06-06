import pandas as pd
import numpy as np
import os, pickle
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import TfidfVectorizer

from data_preprocessor import DataPreprocessor
from logistic_regression import LogisticRegression

input_path = "data/raw/spam.csv"
output_path = "data/processed/sms_spam_processed.csv"
preprocessor = DataPreprocessor()
preprocessor.preprocess(input_path, output_path)

data = pd.read_csv(output_path)

X = data["clean_text"]
y = data['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}\n")

vectorizer = TfidfVectorizer(max_features=3000)
X_train = vectorizer.fit_transform(X_train).toarray()
X_test = vectorizer.transform(X_test).toarray()

print(f"X_train after vectorization shape: {X_train.shape}")
print(f"X_test after vectorization shape: {X_test.shape}\n")

model = LogisticRegression(lr=0.5, num_iter=15000)
model.fit(X_train, y_train)