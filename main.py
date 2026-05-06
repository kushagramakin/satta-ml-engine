import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
import os
import json
import firebase_admin
from firebase_admin import credentials, firestore
import requests
from bs4 import BeautifulSoup

# --- 1. THE WEB SCRAPER (Automatic Data Update) ---
def fetch_latest_result(csv_path):
    print("Attempting to fetch today's result from Satta King Fast...")
    url = "https://satta-king-fast.com/desawar/satta-result-chart/ds/" 
    
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        ds_row = soup.find('tr', id='DS')
        
        if not ds_row:
            raise ValueError("Could not find the Desawar row in the HTML.")
            
        today_str = ds_row.find('td', class_='today-number').find('h3').text.strip()
        
        if not today_str.isdigit():
            raise ValueError(f"Today's number is not yet available. Found: '{today_str}'")
            
        todays_number = int(today_str)
        ist_time = datetime.utcnow() + timedelta(hours=5, minutes=30)
        today_date = ist_time.strftime('%Y-%m-%d')
        
        df = pd.read_csv(csv_path)
        
        if today_date in df['Date'].values:
            print(f"Data for {today_date} already exists in the CSV. Skipping append.")
        else:
            new_row = pd.DataFrame({'Date': [today_date], 'Number': [todays_number]})
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(csv_path, index=False)
            print(f"Added today's result ({todays_number}) to the CSV.")
            
        return df
    
    except Exception as e:
        print(f"Scraping failed: {e}. Proceeding with existing CSV data.")
        return pd.read_csv(csv_path)

# --- 2. THE FEATURE ENGINEER (Preparing the Data) ---
def prepare_data(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek

    df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['Day_Sin'] = np.sin(2 * np.pi * df['Day'] / 31)
    df['Day_Cos'] = np.cos(2 * np.pi * df['Day'] / 31)

    df['Lag_1'] = df['Number'].shift(1)
    df['Lag_2'] = df['Number'].shift(2)
    df['Lag_3'] = df['Number'].shift(3)
    df['Lag_7'] = df['Number'].shift(7) 

    df['Rolling_Mean_3'] = df['Number'].shift(1).rolling(window=3).mean()
    return df

# --- 3. FIREBASE UPLOAD (Updating the Website) ---
def push_to_firebase(predicted_number):
    print("Connecting to Firebase...")
    firebase_secret = os.environ.get('FIREBASE_CREDENTIALS')
    
    if not firebase_secret:
        print("ERROR: Could not find FIREBASE_CREDENTIALS in GitHub Secrets!")
        return

    try:
        creds_dict = json.loads(firebase_secret)
        cred = credentials.Certificate(creds_dict)
        
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)

        db = firestore.client()
        
        ist_time = datetime.utcnow() + timedelta(hours=5, minutes=30)
        tomorrow = (ist_time + timedelta(days=1)).strftime('%Y-%m-%d')
        
        doc_ref = db.collection('predictions').document('latest_prediction')
        doc_ref.set({
            'date': tomorrow,
            'predicted_number': predicted_number,
            'updated_at': firestore.SERVER_TIMESTAMP
        })
        print(f"SUCCESS: Pushed prediction ({predicted_number}) to live website database!")
        
    except Exception as e:
        print(f"Firebase Upload Failed: {e}")

# --- 4. THE MASTER FUNCTION (Tying it all together) ---
def train_and_predict():
    # THIS IS THE LINE WITH THE CORRECT FILE NAME!
    csv_path = 'satta_disawar_historical_data.csv'
    
    # 1. Update data with today's real scraped result
    df_raw = fetch_latest_result(csv_path)
    
    # 2. Add the math columns
    df = prepare_data(df_raw)

    features = ['Lag_1', 'Lag_2', 'Lag_3', 'Lag_7', 'Month_Sin', 'Month_Cos', 'Day_Sin', 'Day_Cos', 'Rolling_Mean_3']
    df_clean = df.dropna().copy()

    X = df_clean[features]
    Y = df_clean['Number']

    # 3. Train the Model
    print("Training the AI Model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, Y)

    # 4. Predict Tomorrow's Number
    latest_clues = df_clean.tail(1)[features]
    raw_prediction = model.predict(latest_clues)[0]
    final_prediction = int(round(raw_prediction))
    
    print(f"Prediction complete. Tomorrow's number is: {final_prediction}")

    # 5. Send to live website
    push_to_firebase(final_prediction)

if __name__ == "__main__":
    train_and_predict()
