import os, pickle
from src.data_preprocessor import DataPreprocessor
from src.logistic_regression import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer 

MODEL_ARTIFACTS_DIR = 'models/'
CLASSIFIER_MODEL_PATH = os.path.join(MODEL_ARTIFACTS_DIR, 'sms_spam_classifier_model_v1.pkl')
TFIDF_VECTORIZER_PATH = os.path.join(MODEL_ARTIFACTS_DIR, 'tfidf_vectorizer.pkl')

def predict_message(message: str) -> str:
    with open(CLASSIFIER_MODEL_PATH, 'rb') as f:
        loaded_model: LogisticRegression = pickle.load(f) 
    with open(TFIDF_VECTORIZER_PATH, 'rb') as f:
        loaded_vectorizer: TfidfVectorizer = pickle.load(f)

    temp_preprocessor = DataPreprocessor() 
    clean_message = temp_preprocessor._clean_message(message)
    
    print(f"[*] Tin nhắn sau tiền xử lý: '{clean_message}'")

    message_vectorized = loaded_vectorizer.transform([clean_message]).toarray()

    prediction_proba = loaded_model.predict_proba(message_vectorized)[0][1]
    prediction = loaded_model.predict(message_vectorized)[0]

    result = "SPAM" if prediction == 1 else "HAM"
    print(f"[*] Dự đoán: {result}")
    print(f"[*] Xác suất là SPAM: {prediction_proba:.4f}\n")
    return result

if __name__ == "__main__":
    print(f"\n===== Test Message 1 =====")
    predict_message("Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075")

    print(f"\n===== Test Message 2 =====")
    predict_message("Hey, how are you doing today? Let's catch up later.")

    print(f"\n===== Test Message 3 =====")
    predict_message("WINNER! You have won a FREE iPhone! Reply to this message now!")

    print(f"\n===== Test Message 4 =====")
    predict_message("Hi, can you meet me for lunch tomorrow?")
