# =======================
# QR PHISHING CNN MODEL (STABLE)
# =======================

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# ---- Path Configuration ----
# Updated to use local paths relative to the project structure
BASE_PATH = "archive"
LEGIT_PATH = os.path.join(BASE_PATH, "benign")
PHISHING_PATH = os.path.join(BASE_PATH, "malicious")

IMG_SIZE = 128
SAMPLES_PER_CLASS = 5000  # ✅ FIXED
MODEL_FILENAME = "qr_phishing_cnn.h5"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, MODEL_FILENAME)

def load_images(folder, label, limit):
    X, y = [], []
    if not os.path.exists(folder):
        print(f"Directory not found: {folder}")
        return X, y
    
    count = 0
    # Handle both direct images and subfolders (like benign/benign)
    items = os.listdir(folder)
    for item in items:
        item_path = os.path.join(folder, item)
        if os.path.isdir(item_path):
            # Recurse if there's a nested folder
            sub_X, sub_y = load_images(item_path, label, limit - count)
            X.extend(sub_X)
            y.extend(sub_y)
            count += len(sub_X)
        else:
            if count >= limit:
                break
            img = cv2.imread(item_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                X.append(img)
                y.append(label)
                count += 1
    return X, y

def train_model():
    print("🚀 Loading datasets...")
    X_legit, y_legit = load_images(LEGIT_PATH, 0, SAMPLES_PER_CLASS)
    X_phish, y_phish = load_images(PHISHING_PATH, 1, SAMPLES_PER_CLASS)

    X = np.array(X_legit + X_phish, dtype=np.float32) / 255.0
    y = to_categorical(np.array(y_legit + y_phish), 2)

    if len(X) == 0:
        print("❌ No images loaded. Check your dataset paths.")
        return

    X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    # CNN Model
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        MaxPooling2D(2,2),

        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train
    print("⏳ Training CNN model...")
    model.fit(
        X_train, y_train,
        epochs=5,
        batch_size=32,
        validation_split=0.1
    )

    # Evaluate
    loss, acc = model.evaluate(X_test, y_test)
    print(f"\n✅ Final Test Accuracy: {acc*100:.2f}%")

    # Save model
    model.save(MODEL_PATH)
    print(f"💾 Model saved to {MODEL_PATH}")

def predict_qr(img_path):
    # Clean up input path (remove quotes and extra whitespace)
    img_path = img_path.strip().strip('"').strip("'")
    
    if not os.path.exists(MODEL_PATH):
        return f"❌ Model file {MODEL_PATH} not found."

    if not os.path.exists(img_path):
        return f"❌ File not found: {img_path}"

    model = load_model(MODEL_PATH)
    print(f"📂 Predicting for: {img_path}")

    # Read image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return "❌ Invalid image: Unable to read file as grayscale"

    # Preprocess
    img_processed = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_processed = img_processed.reshape(1, IMG_SIZE, IMG_SIZE, 1) / 255.0

    # Predict
    prediction = model.predict(img_processed)
    class_id = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    # Result
    if class_id == 0:
        return f"✅ LEGITIMATE QR  ({confidence:.2f}% confidence)"
    else:
        return f"⚠️ MALICIOUS / PHISHING QR  ({confidence:.2f}% confidence)"

def select_and_predict_qr():
    """Helper to simulate file upload/selection in a local environment"""
    import tkinter as tk
    from tkinter import filedialog
    
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window
    root.attributes("-topmost", True) # Bring to front
    
    print("\n📂 Opening file selection dialog...")
    file_path = filedialog.askopenfilename(
        title="Select QR Image",
        filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")]
    )
    root.destroy()
    
    if not file_path:
        return "❌ No file selected."
    
    return predict_qr(file_path)

def detect_qr(image_path=None):
    """Wrapper for main.py. If no path is provided, it opens a file selector."""
    if not image_path:
        result = select_and_predict_qr()
    else:
        result = predict_qr(image_path)
    
    print(f"Prediction: {result}")
    return result

if __name__ == "__main__":
    choice = input("Enter 'T' to train or 'P' to predict: ").strip().upper()
    if choice == 'T':
        train_model()
    elif choice == 'P':
        path = input("Enter the path to the QR image: ").strip()
        predict_qr(path)
    else:
        print("Invalid choice.")
