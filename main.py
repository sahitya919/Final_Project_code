print("=== PHISHING DETECTION SYSTEM ===")
print("1. URL")
print("2. Email")
print("3. QR Code")

choice = input("Select input type (1/2/3): ").strip()

if choice == "1":
    from url_detector import detect_url
    url = input("Enter URL: ")
    result = detect_url(url)
    print(f"\nFinal Result: {result}")

elif choice == "2":
    from email_detector import detect_email
    print("Paste email content (Type 'END' on a new line and press Enter when finished):")
    lines = []
    while True:
        line = input()
        if line.strip().upper() == "END":
            break
        lines.append(line)
    email_text = "\n".join(lines)
    result = detect_email(email_text)
    print(f"\nFinal Result: {result}")

elif choice == "3":
    import os
    from qr_detector import detect_qr, train_model
    if not os.path.exists("qr_phishing_cnn.h5"):
        print("\n🚀 QR Model not found. Starting training (this may take a few minutes)...")
        train_model()
    image_path = input("Enter QR image path: ")
    result = detect_qr(image_path)
    print(f"\nFinal Result: {result}")

else:
    print("Invalid choice")
    exit()
