import pandas as pd
import re, os
import string
from sklearn.model_selection import train_test_split
class DataPreprocessor:
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state

    def _clean_message(self, message):
        if not isinstance(message, str):
            return ""
        message = message.lower()
        message = message.translate(str.maketrans('', '', string.punctuation))
        message = re.sub(r'\d+', '', message)
        message = re.sub(r'[^a-z\s]', '', message)
        message = re.sub(r'\s+', ' ', message).strip()
        return message

    def preprocess(self, input_data, output_dir):
        df = pd.read_csv(input_data, encoding='latin-1')[['v1', 'v2']]
        df.columns = ['label', 'message']
        
        df['message'] = df['message'].astype(str).fillna('')
        df['label'] = df['label'].map({'ham': 0, 'spam': 1})

        df['clean_message'] = df['message'].apply(self._clean_message)

        df = df[df['clean_message'] != ''].copy()
        df['clean_message'] = df['clean_message'].astype(str)

        X = df["clean_message"]
        y = df['label'] 

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state, stratify=y)

        train_df = pd.DataFrame({'label': y_train.reset_index(drop=True), 'clean_message': X_train.reset_index(drop=True)})
        test_df = pd.DataFrame({'label': y_test.reset_index(drop=True), 'clean_message': X_test.reset_index(drop=True)})

        os.makedirs(output_dir, exist_ok=True) 
        train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
        test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
