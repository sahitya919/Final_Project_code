# ===================== IMPORTS =====================
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from cryptography.fernet import Fernet
import os

# ===================== FIXED ENCRYPTION KEY =====================
FIXED_KEY = b'zIp5X89QOdvEfit4zsnMvUuJ09zHckenE8q38LJL46Q='

# ===================== ENCRYPTION MODULE =====================
class EncryptionModule:
    def __init__(self, key):
        self.cipher = Fernet(key)

    def encrypt(self, text):
        return self.cipher.encrypt(text.encode()).decode()

    def decrypt(self, token):
        return self.cipher.decrypt(token.encode()).decode()

    def try_decrypt(self, text):
        try:
            return self.decrypt(text)
        except Exception:
            return text  # Plain text fallback

# ===================== LOAD DATASET =====================
dataset_file = "CEAS_08.csv"
script_dir = os.path.dirname(os.path.abspath(__file__))
abs_dataset_path = os.path.join(script_dir, dataset_file)

if not os.path.exists(abs_dataset_path):
    print(f"❌ Error: Dataset {dataset_file} not found at {abs_dataset_path}.")
    print("Please ensure the CSV file is in the same directory as the script.")
    exit(1)

dataset_file = abs_dataset_path

# Initial diagnostic load as requested by user
print("--- Loading dataset for diagnostics ---")
df_diag = pd.read_csv(dataset_file, engine='python', on_bad_lines='skip')
print("Columns:", df_diag.columns)
if 'label' in df_diag.columns:
    print(df_diag['label'].value_counts())
    print(df_diag['label'].value_counts(normalize=True) * 100)
print("--- Diagnostics complete ---\n")

# Main processing
df = df_diag.copy()
df["email_text"] = df["subject"].fillna("") + " " + df["body"].fillna("")
y = df["label"]

# ===================== TEXT CLEANING =====================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+', ' URL ', text)
    text = re.sub(r'[^a-zA-Z0-9 ]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

df["email_text"] = df["email_text"].apply(clean_text)

# ===================== VECTORIZATION =====================
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X = vectorizer.fit_transform(df["email_text"])

# ===================== TRAIN-TEST SPLIT =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ===================== TRAIN MODEL =====================
model = MultinomialNB(alpha=1.0)
model.fit(X_train, y_train)

# ===================== EVALUATION =====================
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ===================== HEURISTIC PHISHING RULES =====================
def heuristic_email_score(email_text):
    score = 0
    if re.search(r'https?://', email_text):
        score += 2
    if re.search(r'\b\d+\s?(hours?|days?)\b', email_text.lower()):
        score += 2
    if re.search(r'suspend|verify|unusual activity|account', email_text.lower()):
        score += 1
    if email_text.count("!") > 2:
        score += 1
    return score

# ===================== FINAL EMAIL PREDICTION =====================
def predict_email(user_input, encryption_module):
    email_text = encryption_module.try_decrypt(user_input)
    print("\nEmail used for prediction:\n", email_text)

    cleaned = clean_text(email_text)
    vector = vectorizer.transform([cleaned])
    ml_pred = model.predict(vector)[0]

    heuristic_score = heuristic_email_score(email_text)

    if heuristic_score >= 3:
        return "⚠️ Phishing Email (Heuristic-Based)"
    elif ml_pred == 1:
        return "⚠️ Phishing Email (ML-Based)"
    else:
        return "✅ Legitimate Email"

# ===================== INITIALIZE ENCRYPTION =====================
encryption_module = EncryptionModule(FIXED_KEY)

def detect_email(email_text):
    """Wrapper for main.py"""
    result = predict_email(email_text, encryption_module)
    print(f"Prediction: {result}")
    return result

if __name__ == "__main__":
    # ===================== USER INPUT =====================
    print("\nEnter EMAIL (plain OR encrypted). Type 'END' on a new line when finished:")
    try:
        lines = []
        while True:
            line = input()
            if line.strip().upper() == "END":
                break
            lines.append(line)
        user_input = "\n".join(lines).strip()
        
        # ===================== RESULT =====================
        result = predict_email(user_input, encryption_module)
        print("\nFinal Prediction:", result)

        # ===================== OPTIONAL: ENCRYPT INPUT =====================
        if not user_input.startswith("gAAAA"):
            encrypted_email = encryption_module.encrypt(user_input)
            print("\nEncrypted Email (save this):")
            print(encrypted_email)
    except EOFError:
        print("No input provided.")
