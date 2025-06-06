import pandas as pd
import re
import os

class DataPreprocessor:
    def __init__(self):
        pass

    def _clean_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def preprocess(self, input_path, output_path):
        df = pd.read_csv(input_path, encoding='latin-1')[['v1', 'v2']]
        df.columns = ['label', 'text']
        
        df['text'] = df['text'].fillna('')
        df['label'] = df['label'].map({'ham': 0, 'spam': 1})

        df['clean_text'] = df['text'].apply(self._clean_text)

        df = df[df['clean_text'] != '']

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df[['label', 'clean_text']].to_csv(output_path, index=False)