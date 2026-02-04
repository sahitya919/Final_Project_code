import pandas as pd
import numpy as np
import os

# ===================== LOAD DATASET =====================
# Path updated to point to the local dataset location
dataset_path = "PhiUSIIL_Phishing_URL_Dataset.csv"
if not os.path.exists(dataset_path):
    # Try looking in the same directory as the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "PhiUSIIL_Phishing_URL_Dataset.csv")

if not os.path.exists(dataset_path):
    print(f"❌ Error: Dataset file '{dataset_path}' not found.")
    exit(1)

df = pd.read_csv(dataset_path)

# ===================== CREATE REQUIRED FEATURES =====================
df_reduced = pd.DataFrame()

df_reduced['url'] = df['URL']
df_reduced['url_length'] = df['URLLength']
df_reduced['dot_count'] = df['URL'].str.count(r'\.')
df_reduced['hyphen_count'] = df['URL'].str.count('-')
df_reduced['digit_count'] = df['NoOfDegitsInURL']
df_reduced['equal_count'] = df['NoOfEqualsInURL']
df_reduced['question_count'] = df['NoOfQMarkInURL']
df_reduced['has_ip'] = df['IsDomainIP']
df_reduced['is_https'] = df['IsHTTPS']
df_reduced['label'] = df['label']

# ===================== SAVE MODIFIED DATASET =====================
df_reduced.to_csv("phishing_reduced_dataset.csv", index=False)

print("✅ Dataset reduced successfully!")
print("Shape:", df_reduced.shape)
print(df_reduced.head())
