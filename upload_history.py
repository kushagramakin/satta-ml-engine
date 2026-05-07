import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore
import os
import json

print("Starting Data Migration to Firebase...")

# 1. Connect to Firebase
firebase_secret = os.environ.get('FIREBASE_CREDENTIALS')
cred = credentials.Certificate(json.loads(firebase_secret))
firebase_admin.initialize_app(cred)
db = firestore.client()

# 2. Read the CSV
df = pd.read_csv('satta_disawar_historical_data.csv')
df['Date'] = pd.to_datetime(df['Date'])

# 3. Take the last 180 days (6 months) to populate the historical audit table
df_recent = df.tail(180)
print(f"Uploading {len(df_recent)} historical draws...")

# 4. Upload to the 'historical_draws' collection
for index, row in df_recent.iterrows():
    date_obj = row['Date']
    date_str = date_obj.strftime('%Y-%m-%d')
    
    doc_ref = db.collection('historical_draws').document(date_str)
    doc_ref.set({
        'date': date_obj,
        'winning_number': int(row['Winning_Number']),
        'predicted_number': None, # AI didn't exist yet
        'is_hit': False
    })

print("SUCCESS: Pure historical data migrated to Firebase!")
