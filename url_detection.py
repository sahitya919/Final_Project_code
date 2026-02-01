# ===================== IMPORTS =====================
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from urllib.parse import urlparse
import re
from cryptography.fernet import Fernet
import os

# ===================== FIXED ENCRYPTION KEY =====================
FIXED_KEY = b'zIp5X89QOdvEfit4zsnMvUuJ09zHckenE8q38LJL46Q='

# ===================== ENCRYPTION MODULE =====================
class EncryptionModule:
    def __init__(self, key):
        if isinstance(key, str):
            key = key.encode()
        self.key = key
        self.cipher = Fernet(self.key)

    def get_key(self):
        return self.key

    def encrypt(self, data):
        if isinstance(data, str):
            data = data.encode("utf-8")
        return self.cipher.encrypt(data)

    def decrypt(self, encrypted_data):
        if isinstance(encrypted_data, str):
            encrypted_data = encrypted_data.encode("utf-8")
        return self.cipher.decrypt(encrypted_data).decode("utf-8")

    def try_decrypt(self, data):
        try:
            return self.decrypt(data)
        except Exception:
            raise ValueError("❌ Decryption failed — KEY MISMATCH or INVALID TOKEN")

# ===================== LOAD DATASET =====================
dataset_file = "phishing_reduced_dataset.csv"
if not os.path.exists(dataset_file):
    print(f"Dataset {dataset_file} not found. Please run prepare_dataset.py first.")
    exit()

df = pd.read_csv(dataset_file)
print("Dataset shape:", df.shape)

X = df.drop(columns=["url", "label"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===================== TRAIN MODEL =====================
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# ===================== FEATURE EXTRACTION =====================
def extract_features_from_url(url):
    url = str(url)

    url_length = len(url)
    dot_count = url.count(".")
    hyphen_count = url.count("-")
    digit_count = sum(c.isdigit() for c in url)
    equal_count = url.count("=")
    question_count = url.count("?")

    domain = urlparse(url).netloc
    has_ip = 1 if re.fullmatch(r"\d+\.\d+\.\d+\.\d+", domain) else 0
    is_https = 1 if url.lower().startswith("https") else 0

    return np.array([
        url_length,
        dot_count,
        hyphen_count,
        digit_count,
        equal_count,
        question_count,
        has_ip,
        is_https
    ]).reshape(1, -1)

# ===================== PREDICTION FUNCTION =====================
def predict_url(user_input, encryption_module):
    # Detect if input is encrypted (Fernet tokens start with "gAAAA")
    if isinstance(user_input, str) and user_input.startswith("gAAAA"):
        user_input = user_input.encode("utf-8")
        decrypted_url = encryption_module.try_decrypt(user_input)
    else:
        decrypted_url = user_input  # Plain URL

    print("\nURL used for prediction:", decrypted_url)

    features = extract_features_from_url(decrypted_url)
    prediction = rf_model.predict(features)[0]

    return "⚠️ Phishing URL" if prediction == 1 else "✅ Legitimate URL"

# ===================== INITIALIZE ENCRYPTION MODULE =====================
encryption_module = EncryptionModule(FIXED_KEY)
print("\nEncryption Key (SAVE this to reuse encrypted URLs):")
print(encryption_module.get_key())

if __name__ == "__main__":
    # ===================== USER INPUT =====================
    print("\nEnter URL (plain OR encrypted):")
    # Using input() but in some headless environments this might fail
    try:
        user_input = input().strip()
        # ===================== FINAL PREDICTION =====================
        result = predict_url(user_input, encryption_module)
        print("\nFinal Prediction:", result)

        # ===================== OPTIONAL: ENCRYPT PLAIN URL =====================
        if not user_input.startswith("gAAAA"):
            encrypted_url = encryption_module.encrypt(user_input)
            print("\nEncrypted URL (save this for later use):")
            print(encrypted_url.decode())
    except EOFError:
        print("No input provided.")
