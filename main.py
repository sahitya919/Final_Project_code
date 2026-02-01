print("=== PHISHING DETECTION SYSTEM ===")
print("1. URL")
print("2. Email")
print("3. QR Code")

choice = input("Select input type (1/2/3): ").strip()

if choice == "1":
    from url_detector import detect_url
    url = input("Enter URL: ")
    result = detect_url(url)

elif choice == "2":
    from email_detector import detect_email
    email_text = input("Paste email content: ")
    result = detect_email(email_text)

elif choice == "3":
    from qr_detector import detect_qr
    image_path = input("Enter QR image path: ")
    result = detect_qr(image_path)

else:
    print("Invalid choice")
    exit()
