import os
from src.data_preprocessor import DataPreprocessor

INPUT_DATA = 'data/raw/spam.csv' 
OUTPUT_DIR = 'data/processed/' 

TEST_SIZE = 0.2
RANDOM_STATE = 42

def prepare_data():
    preprocessor = DataPreprocessor(test_size=TEST_SIZE, random_state=RANDOM_STATE)
    preprocessor.preprocess(INPUT_DATA, OUTPUT_DIR)

if __name__ == "__main__":
    prepare_data()