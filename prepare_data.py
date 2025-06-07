import os
from src.data_preprocessor import DataPreprocessor

def prepare_data():
    preprocessor = DataPreprocessor(test_size=0.2, random_state=42)
    preprocessor.preprocess('data/raw/spam.csv', 'data/processed/')

if __name__ == "__main__":
    prepare_data()