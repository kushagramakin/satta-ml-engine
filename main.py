import os
import json
import warnings
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

import firebase_admin
from firebase_admin import credentials, firestore
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import optuna

# Suppress deprecation and user warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# --- 0. FIREBASE CONNECTION MANAGER ---
def init_firebase():
    """Ensures Firebase is connected before attempting read/write operations."""
    if not firebase_admin._apps:
        firebase_secret = os.environ.get('FIREBASE_CREDENTIALS')
        if firebase_secret:
            creds_dict = json.loads(firebase_secret)
            cred = credentials.Certificate(creds_dict)
            firebase_admin.initialize_app(cred)
            return True
        else:
            print("ERROR: FIREBASE_CREDENTIALS not found in environment!")
            return False
    return True

# --- 1. AUDIT & METRICS SYNC ---
def sync_recent_audit(df):
    if not init_firebase(): return
    db = firestore.client()
    recent_df = df.tail(7)
    
    for _, row in recent_df.iterrows():
        date_str = str(row['Date'])
        winning_number = int(row['Winning_Number'])
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        
        update_data = {
            'date': date_obj,
            'winning_number': winning_number
        }
        
        pred_ref = db.collection('daily_predictions').document(date_str).get()
        if pred_ref.exists:
            pred_data = pred_ref.to_dict()
            predicted_number = pred_data.get('top_prediction')
            
            top_5_list = [predicted_number]
            for i in range(1, 5):
                runner_up = pred_data.get(f'runner_up_{i}')
                if runner_up:
                    top_5_list.append(runner_up.get('number'))

            update_data['predicted_number'] = predicted_number
            update_data['top_5_predictions'] = top_5_list
            update_data['is_hit'] = (predicted_number == winning_number)
            update_data['is_top_5_hit'] = (winning_number in top_5_list)

        db.collection('historical_draws').document(date_str).set(update_data, merge=True)
        
    print("SUCCESS: Recent historical audit verified and synced to Firebase.")

def sync_monthly_metrics():
    if not init_firebase(): return
    db = firestore.client()
    
    ist_time = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
    current_month_str = ist_time.strftime('%Y-%m')
    start_of_month = ist_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    docs = db.collection('historical_draws').where('date', '>=', start_of_month).stream()
    
    total_signals = 0
    total_hits = 0
    
    for doc in docs:
        data = doc.to_dict()
        if 'predicted_number' in data and data['predicted_number'] is not None:
            total_signals += 1
            if data.get('is_hit') is True:
                total_hits += 1
                
    if total_signals > 0:
        accuracy_rate = total_hits / total_signals
        db.collection('monthly_metrics').document(current_month_str).set({
            'month_year': current_month_str,
            'accuracy_rate': accuracy_rate,
            'total_signals': total_signals,
            'total_hits': total_hits
        }, merge=True)
        print(f"SUCCESS: Auto-updated metrics for {current_month_str} (Top-1 Accuracy: {accuracy_rate*100:.1f}%)")

def fetch_latest_result(csv_path):
    print("Attempting to fetch today's result from Satta King Fast...")
    url = "https://satta-king-fast.com/desawar/satta-result-chart/ds/"
    
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        ds_row = soup.find('tr', id='DS')
        
        if not ds_row:
            raise ValueError("Could not find the Desawar row in the HTML.")

        today_str = ds_row.find('td', class_='today-number').find('h3').text.strip()
        
        if not today_str.isdigit():
            raise ValueError(f"Today's number is not yet available. Found: '{today_str}'")
            
        todays_number = int(today_str)
        ist_time = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
        today_date = ist_time.strftime('%Y-%m-%d')
        
        df = pd.read_csv(csv_path)

        if today_date in df['Date'].values:
            print(f"Data for {today_date} already exists in CSV. Skipping append.")
        else:
            dt_obj = datetime.strptime(today_date, '%Y-%m-%d')
            new_row = pd.DataFrame([{
                'Date': today_date,
                'Year': float(dt_obj.year),
                'Month': float(dt_obj.month),
                'Month_Name': dt_obj.strftime('%B'),
                'Day': float(dt_obj.day),
                'Winning_Number': float(todays_number)
            }])
            
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(csv_path, index=False)
            print(f"Added today's result ({todays_number}) to CSV.")
            
        sync_recent_audit(df)
        sync_monthly_metrics()
        return df
    
    except Exception as e:
        print(f"Scraping notice/failure: {e}. Proceeding with existing CSV data.")
        df = pd.read_csv(csv_path)
        sync_recent_audit(df)
        sync_monthly_metrics()
        return df

# --- 2. CULTURAL SEASONALITY ---
def apply_cultural_seasonality(df):
    festival_map = {
        2022: ['2022-01-14', '2022-03-18', '2022-05-03', '2022-08-11', '2022-10-24'],
        2023: ['2023-01-14', '2023-03-08', '2023-04-22', '2023-08-30', '2023-11-12'],
        2024: ['2024-01-14', '2024-03-25', '2024-04-11', '2024-08-19', '2024-10-31'],
        2025: ['2025-01-14', '2025-03-14', '2025-03-31', '2025-08-09', '2025-10-20'],
        2026: ['2026-01-14', '2026-03-03', '2026-03-20', '2026-08-28', '2026-11-08'],
        2027: ['2027-01-14', '2027-03-22', '2027-03-10', '2027-08-17', '2027-10-29']
    }

    all_festivals = [d for dates in festival_map.values() for d in dates]
    fest_dates = pd.to_datetime(all_festivals)
    
    def days_to_nearest(current_date):
        future_fests = fest_dates[fest_dates >= current_date]
        return (future_fests[0] - current_date).days if not future_fests.empty else 30
        
    df['Days_To_Festival'] = df['Date'].apply(days_to_nearest)
    df['Festival_Mode'] = (df['Days_To_Festival'] <= 3).astype(int)
    return df

# --- 3. FEATURE ENGINEERING ---
def prepare_data(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    
    df['Lag_1'] = df['Winning_Number'].shift(1)
    df['Is_High'] = (df['Winning_Number'] >= 50).astype(int)
    df['Is_Even'] = (df['Winning_Number'] % 2 == 0).astype(int)
    
    df['Lag_1_Is_High'] = df['Is_High'].shift(1)
    df['Lag_1_Is_Even'] = df['Is_Even'].shift(1)

    df['Rolling_Std_14'] = df['Winning_Number'].shift(1).rolling(window=14).std()
    df['Rolling_Mean_30'] = df['Winning_Number'].shift(1).rolling(window=30).mean()
    df['Z_Score_30'] = (df['Lag_1'] - df['Rolling_Mean_30']) / df['Rolling_Std_14'].replace(0, 1.0)
    
    def get_dominant_frequency(series):
        if series.isna().any(): return 0
        fft_vals = np.fft.fft(series.values)
        return float(np.abs(fft_vals)[1])

    df['FFT_Pulse_14d'] = df['Winning_Number'].shift(1).rolling(window=14).apply(get_dominant_frequency, raw=False)
    df = apply_cultural_seasonality(df)
    df = df.fillna(0)
    return df

# --- 4. FIREBASE UPLOAD ---
def push_to_firebase(top_preds, signals_data):
    print("Uploading multi-prediction matrix to Firebase...")
    if not init_firebase(): return

    try:
        db = firestore.client()
        ist_time = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
        target_date_obj = ist_time if ist_time.hour < 5 else ist_time + timedelta(days=1)
        pure_date_obj = target_date_obj.replace(hour=0, minute=0, second=0, microsecond=0)
        target_str = pure_date_obj.strftime('%Y-%m-%d')

        prediction_data = {
            "target_date": pure_date_obj,
            "top_prediction": top_preds[0]['number'],
            "top_probability_percent": top_preds[0]['prob'], 
            "runner_up_1": {"number": top_preds[1]['number'], "probability": top_preds[1]['prob']},
            "runner_up_2": {"number": top_preds[2]['number'], "probability": top_preds[2]['prob']},
            "runner_up_3": {"number": top_preds[3]['number'], "probability": top_preds[3]['prob']},
            "runner_up_4": {"number": top_preds[4]['number'], "probability": top_preds[4]['prob']},
            "signals": signals_data,
            "timestamp": firestore.SERVER_TIMESTAMP
        }
        db.collection('daily_predictions').document(target_str).set(prediction_data)
        
        latest_data = {
            'date': target_str,  
            'predicted_number': top_preds[0]['number'],
            'updated_at': firestore.SERVER_TIMESTAMP
        }
        db.collection('predictions').document('latest_prediction').set(latest_data)
        print("SUCCESS: Synced live prediction matrix to Firebase.")
        
    except Exception as e:
        print(f"Firebase Upload Failed: {e}")

# --- 5. MASTER TRAINING & PREDICTION ENGINE ---
def train_and_predict():
    csv_path = 'satta_disawar_historical_data.csv'
    df_raw = fetch_latest_result(csv_path)

    ist_time = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
    target_date_obj = ist_time if ist_time.hour < 5 else ist_time + timedelta(days=1)
    target_str = target_date_obj.strftime('%Y-%m-%d')
    
    if target_str not in df_raw['Date'].values:
        dummy_row = pd.DataFrame({'Date': [target_str], 'Winning_Number': [np.nan]})
        df_raw = pd.concat([df_raw, dummy_row], ignore_index=True)
    
    df = prepare_data(df_raw)

    initial_features = [
        'Lag_1', 'Month_Sin', 'Month_Cos', 'Lag_1_Is_High', 'Lag_1_Is_Even', 
        'Rolling_Std_14', 'Z_Score_30', 'Days_To_Festival', 'Festival_Mode',
        'FFT_Pulse_14d'
    ]

    train_df = df[df['Date'] < pd.to_datetime(target_str)].copy()
    X_full = train_df[initial_features]
    Y_raw = train_df['Winning_Number'].astype(int)
    
    le = LabelEncoder()
    Y = pd.Series(le.fit_transform(Y_raw), index=train_df.index)

    # Time decay sample weighting
    max_date = train_df['Date'].max()
    train_df['Days_Old'] = (max_date - train_df['Date']).dt.days
    time_decay_weights = 1 / (1 + (train_df['Days_Old'] / 365.0))

    # --- DIMENSION 1: FEATURE SELECTION ---
    print("Evaluating initial feature ranking...")
    primer = XGBClassifier(
        n_estimators=50, 
        max_depth=4,
        random_state=42, 
        eval_metric='mlogloss',
        objective='multi:softprob'
    )
    primer.fit(X_full, Y)
    
    importances = primer.feature_importances_
    sorted_features = [f for f, _ in sorted(zip(initial_features, importances), key=lambda x: x[1], reverse=True)]

    # --- DIMENSION 2: OPTUNA HYPERPARAMETER TUNING ---
    print("\nRunning Optuna Tuning with TimeSeriesSplit & Log-Loss...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    unique_classes = np.sort(Y.unique())
    
    def objective(trial):
        prune_ratio = trial.suggest_float('prune_ratio', 0.5, 1.0)
        keep_count = max(3, int(len(sorted_features) * prune_ratio))
        current_features = sorted_features[:keep_count]
        
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 6),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 5.0, log=True)
        }
        
        # Temporal CV using TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=3)
        losses = []
        
        for train_idx, val_idx in tscv.split(train_df):
            X_tr, X_val = train_df[current_features].iloc[train_idx], train_df[current_features].iloc[val_idx]
            y_tr, y_val = Y.iloc[train_idx], Y.iloc[val_idx]
            w_tr = time_decay_weights.iloc[train_idx]
            
            clf = XGBClassifier(**params, random_state=42, eval_metric='mlogloss', objective='multi:softprob')
            clf.fit(X_tr, y_tr, sample_weight=w_tr)
            
            val_probs = clf.predict_proba(X_val)
            # Evaluate log loss against classes present in training split
            loss = log_loss(y_val, val_probs, labels=clf.classes_)
            losses.append(loss)
            
        return np.mean(losses)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=15)
    
    best_params = study.best_params
    best_prune = best_params.pop('prune_ratio')
    keep_count = max(3, int(len(sorted_features) * best_prune))
    top_features = sorted_features[:keep_count]
    
    print(f"Optimal Pruning: Retaining {len(top_features)} of {len(initial_features)} features.")
    print(f"Optimal Parameters: {best_params}")

    # --- DIMENSION 3: FIT FINAL MODEL ---
    print("\nTraining Final Tuned Model...")
    final_model = XGBClassifier(**best_params, random_state=42, eval_metric='mlogloss', objective='multi:softprob')
    final_model.fit(train_df[top_features], Y, sample_weight=time_decay_weights)

    tomorrow_clues = df.tail(1)[top_features].copy()
    
    # --- DIMENSION 4: TARGETED MONTE CARLO SIMULATION ---
    print("\nRunning Targeted Monte Carlo Simulations (100 runs)...")
    mc_predictions = []
    
    # Identify continuous features safe for variance perturbation
    continuous_features = [f for f in ['Rolling_Std_14', 'Z_Score_30', 'FFT_Pulse_14d'] if f in top_features]
    
    for _ in range(100):
        perturbed_clues = tomorrow_clues.copy()
        if continuous_features:
            stds = train_df[continuous_features].std().replace(0, 0.01).values
            noise = np.random.normal(0, stds * 0.05, size=(1, len(continuous_features)))
            perturbed_clues[continuous_features] += noise
            
        probs = final_model.predict_proba(perturbed_clues)[0]
        mc_predictions.append(probs)
    
    mean_probabilities = np.mean(mc_predictions, axis=0)
    model_classes = le.inverse_transform(final_model.classes_)
    
    top_5_indices = np.argsort(mean_probabilities)[-5:][::-1]
    top_preds = [
        {"number": int(model_classes[idx]), "prob": float(round(mean_probabilities[idx] * 100, 2))}
        for idx in top_5_indices
    ]
    
    # --- LIVE SIGNAL METRICS ---
    ist_now = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
    top_feat_idx = np.argmax(final_model.feature_importances_)
    top_feat_name = top_features[top_feat_idx].upper()
    fest_days = int(df.tail(1)['Days_To_Festival'].values[0])
    
    live_signals = [
        {"time": ist_now.strftime('%H:%M:%S'), "signal": "MONTE_CARLO_CONSENSUS", "confidence": f"{top_preds[0]['prob']:.2f}%", "status": "ACTIVE"},
        {"time": ist_now.strftime('%H:%M:%S'), "signal": f"PRIMARY_FEATURE: {top_feat_name}", "confidence": f"{int(final_model.feature_importances_[top_feat_idx] * 100)}% WGT", "status": "ACTIVE"},
        {"time": ist_now.strftime('%H:%M:%S'), "signal": f"CULTURAL_PROXIMITY: {fest_days}D", "confidence": "CALENDAR_SYNC", "status": "ACTIVE"}
    ]
    
    push_to_firebase(top_preds, live_signals)

if __name__ == "__main__":
    train_and_predict()
